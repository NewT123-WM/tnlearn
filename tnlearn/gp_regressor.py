"""
Genetic Programming Symbolic Regressors.

This module provides two symbolic regression implementations:
1. VecSymRegressor (legacy): Original vectorized symbolic regression.
2. GPSymRegressor (new): Enhanced GP with InnerProduct, advanced functions, and mode selection.

Copyright (c) 2024 Meng WANG. All Rights Reserved.
Copyright (c) 2026 Tieyun LI. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
from sympy import sympify, expand
import operator
from random import randint, random, shuffle, seed
from copy import deepcopy
from tqdm import tqdm
import re

# InnerProduct support for GPSymRegressor (advanced/base modes)
from .operator.inner_product import InnerProduct, inner_product, convert_pretty_to_innerproduct, _simplify_expr


# ---------- Helper for random seed ----------
def _set_random_state(rs):
    """Set random seed for both random and numpy."""
    seed(rs)
    np.random.seed(rs)


# =============================================================================
# GPSymRegressor
# =============================================================================
class GPSymRegressor:
    """
    Genetic Programming Symbolic Regressor with InnerProduct and advanced modes.
    Supports 'base' (basic ops), 'advanced' (trig, exp, log, power), and 'legacy'
    (which delegates to VecSymRegressor).

    Parent selection strategies:
        - 'rank' (default): Rank-based weighted roulette (higher rank = higher chance).
        - 'tournament': Tournament selection using `tournament_size`.
    """

    def __init__(self,
                 random_state=100,
                 pop_size=5000,
                 max_generations=20,
                 tournament_size=10,
                 coefficient_range=None,
                 x_pct=0.7,
                 xover_pct=0.3,
                 save=False,
                 operations=None,
                 max_depth=6,
                 complexity_penalty=0.0,
                 node_penalty_coef=0.001,
                 immigration_rate=0.1,
                 elite_ratio=0.1,
                 mode='base',
                 maxpower=5,
                 parent_selection='rank', 
                 debug=False):
        """
        Args:
            random_state (int): Seed.
            pop_size (int): Population size.
            max_generations (int): Generations.
            tournament_size (int): Tournament size (used when parent_selection='tournament').
            coefficient_range (list): Range for random constants.
            x_pct (float): Probability of variable leaf.
            xover_pct (float): Crossover probability.
            save (bool): Save log.
            operations (tuple): Custom operations (overrides default).
            max_depth (int): Maximum tree depth.
            complexity_penalty (float): Penalty per InnerProduct.
            node_penalty_coef (float): Penalty per node.
            immigration_rate (float): Rate of random immigrant.
            elite_ratio (float): Fraction of elites kept.
            mode (str): 'base', 'advanced', or 'legacy'.
            maxpower (int): Max exponent magnitude for power operator.
            parent_selection (str): Parent selection strategy: 'rank' or 'tournament'.
            debug (bool): Print debug info.
        """
        _set_random_state(random_state)
        self.random_state = random_state
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.tournament_size = tournament_size

        self.x_pct = x_pct
        self.xover_pct = xover_pct
        self.save = save
        self.max_depth = max_depth
        self.complexity_penalty = complexity_penalty
        self.node_penalty_coef = node_penalty_coef
        self.immigration_rate = immigration_rate
        self.elite_ratio = elite_ratio
        self.debug = debug

        self.global_best = float("inf")
        self.best_prog = None
        self.neuron = None

        # Parent selection strategy
        self.parent_selection = parent_selection
        if self.parent_selection not in ('rank', 'tournament'):
            raise ValueError("parent_selection must be 'rank' or 'tournament'")
        # Ensure tournament_size has a sensible default if not provided
        if self.parent_selection == 'tournament' and self.tournament_size is None:
            self.tournament_size = 10

        if coefficient_range is None:
            self.coefficient_range = [-1, 1]
        else:
            self.coefficient_range = coefficient_range

        self.mode = mode
        self.maxpower = maxpower

        # For legacy mode, we skip operation initialization
        if mode == 'legacy':
            self.operations = None
            self.power_prob = 0.0
            # We won't use these but set to avoid attribute errors
            self._simp_cache = {}
            return

        # Default operations (base)
        base_ops = (
            {"func": operator.add, "arg_count": 2, "format_str": "({} + {})"},
            {"func": operator.sub, "arg_count": 2, "format_str": "({} - {})"},
            {"func": operator.mul, "arg_count": 2, "format_str": "({} * {})"},
            {"func": operator.neg, "arg_count": 1, "format_str": "-({})"},
            {"func": lambda a, b: InnerProduct(a, b), "arg_count": 2,
             "format_str": "InnerProduct({}, {})"},
        )

        if mode == 'base':
            if operations is not None:
                self.operations = operations
            else:
                self.operations = base_ops
            self.power_prob = 0.0
        elif mode == 'advanced':
            advanced_ops = (
                {"func": np.sin, "arg_count": 1, "format_str": "sin({})"},
                {"func": np.cos, "arg_count": 1, "format_str": "cos({})"},
                {"func": np.exp, "arg_count": 1, "format_str": "exp({})"},
                {"func": np.log, "arg_count": 1, "format_str": "log({})"},
                {"func": np.tan, "arg_count": 1, "format_str": "tan({})"},
            )
            if operations is not None:
                self.operations = operations
            else:
                self.operations = base_ops + advanced_ops
            self.power_prob = 1.0 / (len(self.operations) + 1)
        else:
            raise ValueError("mode must be 'base', 'advanced', or 'legacy'")

        self._simp_cache = {}

    def render_prog(self, node):
        if "children" not in node:
            return node["feature_name"]
        return node["format_str"].format(*[self.render_prog(c) for c in node["children"]])

    def get_tree_depth(self, node):
        if "children" not in node:
            return 1
        return 1 + max([self.get_tree_depth(c) for c in node["children"]])

    def simp(self, tree):
        key = self.render_prog(tree)
        if key in self._simp_cache:
            return self._simp_cache[key]
        expr_str = key
        try:
            sym_expr = sympify(expr_str, locals={'InnerProduct': InnerProduct})
            sym_expr = expand(sym_expr)
            sym_expr = _simplify_expr(sym_expr)
            result = str(sym_expr)
        except Exception:
            result = expr_str
            if self.debug:
                print(f"[WARNING] Simplification failed for: {expr_str}")
        self._simp_cache[key] = result
        return result

    def evaluate(self, expr_str, x_data):
        N = x_data.shape[0]
        def batch_inner(a, b):
            return inner_product(a, b, N=N)
        safe_dict = {
            'x': x_data,
            'InnerProduct': batch_inner,
            'np': np,
            'sin': np.sin,
            'cos': np.cos,
            'exp': np.exp,
            'sqrt': np.sqrt,
            'log': np.log,
            'tan': np.tan
        }
        pred = eval(expr_str, safe_dict)
        return expr_str, pred

    def compute_fitness(self, expr_str, pred, label, node_count):
        if expr_str.count('x') <= 1:
            return float("inf")
        if 'x' not in expr_str:
            return float("inf")
        if expr_str.strip() in ('x', '1', '-1'):
            return float("inf")

        if pred.ndim == 2:
            if pred.shape[1] == 1:
                pred = pred.flatten()
            else:
                pred = np.sum(pred, axis=1)
        elif pred.ndim != 1:
            return float("inf")

        if len(pred) != len(label):
            return float("inf")

        mse = np.mean(np.square(pred - label))

        if self.complexity_penalty > 0:
            ip_count = expr_str.count('InnerProduct')
            mse += self.complexity_penalty * ip_count

        if self.node_penalty_coef > 0:
            mse += self.node_penalty_coef * node_count

        return mse

    def rand_w(self):
        low, high = self.coefficient_range
        eps = 1e-4
        if low < 0 < high:
            if random() < 0.5:
                val = np.random.uniform(low, -eps)
            else:
                val = np.random.uniform(eps, high)
        else:
            val = np.random.uniform(low, high)
        return str(val)

    def _rand_power(self):
        low = -self.maxpower
        high = self.maxpower
        eps = 1e-4
        if low < 0 < high:
            if random() < 0.5:
                val = np.random.uniform(low, -eps)
            else:
                val = np.random.uniform(eps, high)
        else:
            val = np.random.uniform(low, high)
        return str(val)

    def random_prog(self, depth=0, max_depth=None):
        if max_depth is None:
            max_depth = self.max_depth

        if self.mode == 'advanced' and depth < max_depth:
            if random() < self.power_prob:
                child = self.random_prog(depth + 1, max_depth)
                power = self._rand_power()
                power_leaf = {"feature_name": power}
                return {
                    "func": operator.pow,
                    "children": [child, power_leaf],
                    "format_str": "({} ** {})",
                }

        if depth >= max_depth:
            return {'feature_name': 'x'} if random() < self.x_pct else {'feature_name': self.rand_w()}
        op = self.operations[randint(0, len(self.operations) - 1)]
        return {
            "func": op["func"],
            "children": [self.random_prog(depth + 1, max_depth) for _ in range(op["arg_count"])],
            "format_str": op["format_str"],
        }

    def select_random_node(self, selected, parent, depth):
        if "children" not in selected:
            return parent
        if randint(0, 10) < 2 * depth:
            return selected
        child_count = len(selected["children"])
        return self.select_random_node(
            selected["children"][randint(0, child_count - 1)],
            selected, depth + 1)

    def do_mutate(self, selected):
        for attempt in range(10):
            offspring = deepcopy(selected)
            mutate_point = self.select_random_node(offspring, None, 0)
            child_count = len(mutate_point["children"])
            mutate_point["children"][randint(0, child_count - 1)] = self.random_prog(0, max_depth=2)
            if self.get_tree_depth(offspring) <= self.max_depth:
                return offspring
        return self.random_prog(0)

    def do_xover(self, selected1, selected2):
        for attempt in range(10):
            offspring = deepcopy(selected1)
            xover_point1 = self.select_random_node(offspring, None, 0)
            xover_point2 = self.select_random_node(selected2, None, 0)
            child_count = len(xover_point1["children"])
            xover_point1["children"][randint(0, child_count - 1)] = xover_point2
            if self.get_tree_depth(offspring) <= self.max_depth:
                return offspring
        return self.random_prog(0)

    def get_random_parent(self, popu, fitne):
        """
        Select a parent based on the chosen parent_selection strategy.
        """
        # ---------- Tournament selection ----------
        if self.parent_selection == 'tournament':
            k = self.tournament_size if self.tournament_size else 10
            N = len(popu)
            if k > N:
                k = N
            indices = np.random.choice(N, size=k, replace=False)
            best_idx = min(indices, key=lambda i: fitne[i])
            return popu[best_idx]

        # ---------- Rank‑based weighted selection (default) ----------
        else:  # 'rank'
            sorted_indices = np.argsort(fitne)
            N = len(fitne)
            weights = np.arange(N, 0, -1)
            total = weights.sum()
            probs = weights / total
            idx = np.random.choice(sorted_indices, p=probs)
            return popu[idx]

    def get_offspring(self, popula, ftns):
        if random() < self.immigration_rate:
            return self.random_prog(0)
        tempt = random()
        parent1 = self.get_random_parent(popula, ftns)
        if tempt < self.xover_pct:
            parent2 = self.get_random_parent(popula, ftns)
            return self.do_xover(parent1, parent2)
        elif self.xover_pct <= tempt < 0.9:
            return self.do_mutate(parent1)
        else:
            return parent1

    def node_count(self, x):
        if "children" not in x:
            return 1
        return sum([self.node_count(c) for c in x["children"]])

    def _format_pretty(self, expr_str):
        def repl(match):
            args = match.group(1)
            parts = args.split(',', 1)
            if len(parts) == 2:
                return f"<{parts[0].strip()}, {parts[1].strip()}>"
            return match.group(0)
        while True:
            new_expr = re.sub(r'InnerProduct\(([^()]*)\)', repl, expr_str)
            if new_expr == expr_str:
                break
            expr_str = new_expr
        return expr_str

    def fit(self, X, y):
        """Fit the regressor to data."""
        _set_random_state(self.random_state)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # ----- Legacy mode: delegate to VecSymRegressor -----
        if self.mode == 'legacy':
            vec_reg = VecSymRegressor(
                random_state=self.random_state,
                pop_size=self.pop_size,
                max_generations=self.max_generations,
                tournament_size=self.tournament_size,
                coefficient_range=self.coefficient_range,
                x_pct=self.x_pct,
                xover_pct=self.xover_pct,
                save=self.save,
                operations=None   # use default ops
            )
            vec_reg.fit(X, y)
            # Copy results
            self.best_score = vec_reg.best_score
            self.best_program = vec_reg.best_program
            self.neuron = vec_reg.neuron
            self.global_best = self.best_score
            self.best_prog = self.best_program
            return self

        # ----- Base / Advanced modes -----
        self._simp_cache = {}
        self.population = [self.random_prog() for _ in range(self.pop_size)]
        self.box = {}

        if self.save:
            file = open("log.txt", 'w')

        for gen in tqdm(range(self.max_generations), desc="Fitting Progress"):
            fitness = []
            for prog in self.population:
                expr_str = self.simp(prog)
                if self.mode == 'advanced':
                    if any(bad in expr_str for bad in ('I', 'zoo', 'nan', 'inf', 'E')):
                        fitness.append(float("inf"))
                        continue
                try:
                    _, prediction = self.evaluate(expr_str, X)
                except Exception:
                    fitness.append(float("inf"))
                    continue
                n_nodes = self.node_count(prog)
                score = self.compute_fitness(expr_str, prediction, y, n_nodes)
                fitness.append(score)

                if score < self.global_best:
                    self.global_best = score
                    self.best_prog = expr_str

                if len(self.box) < self.pop_size * 0.05:
                    self.box[score] = prog
                else:
                    key_sort = sorted(self.box)
                    if score < key_sort[-1]:
                        self.box.pop(key_sort[-1])
                        self.box[score] = prog

            if self.debug:
                finite_count = sum(np.isfinite(fitness))
                print(f"\n[DEBUG] Generation {gen+1}/{self.max_generations}")
                print(f"  Best fitness: {self.global_best:.6f}")
                print(f"  Best expression: {self.best_prog}")
                if fitness:
                    median = np.median(fitness)
                    mean = np.mean(fitness)
                    inf_count = sum(np.isinf(fitness))
                    print(f"  Pop median fitness: {median:.6f}, mean: {mean:.6f}, inf count: {inf_count}")
                    print(f"  Finite fitness count: {finite_count} / {len(fitness)}")

            if self.save:
                file.write(
                    "Generation: {:d}\nBest Score: {:.4f}\nMedian score: {:.4f}\nBest program: {:s}\n\n"
                    .format(gen+1, self.global_best, np.median(np.array(fitness)), str(self.best_prog))
                )

            elite_limit = max(1, int(self.pop_size * self.elite_ratio))
            sorted_elites = sorted(self.box.items(), key=lambda x: x[0])[:elite_limit]
            elites = [prog for _, prog in sorted_elites]

            num_offspring = self.pop_size - len(elites)
            if num_offspring > 0:
                shuffle(self.population)
                offspring = [self.get_offspring(self.population, fitness) for _ in range(num_offspring)]
            else:
                offspring = []

            self.population = (offspring + elites)[:self.pop_size]

        self.best_score = self.global_best
        self.best_program = self.best_prog
        if self.save:
            file.write("Best score: %f\n" % self.best_score)
            file.write("Best program: %s\n" % str(self.best_prog))
            file.close()

        self.neuron = self._format_pretty(self.best_prog) if self.best_prog else None
        return self


# =============================================================================
# VecSymRegressor (Legacy)
# =============================================================================
class VecSymRegressor:
    """
    Legacy vectorized symbolic regression.

    Original implementation by Meng WANG.
    """

    def __init__(self,
                 random_state=100,
                 pop_size=5000,
                 max_generations=20,
                 tournament_size=10,
                 coefficient_range=None,
                 x_pct=0.7,
                 xover_pct=0.3,
                 save=False,
                 operations=None):
        """
        Args:
            random_state (int): Seed for random number generation.
            pop_size (int): Population size.
            max_generations (int): Maximum generations.
            tournament_size (int): Size of tournament selection.
            coefficient_range (list): Range for random constants.
            x_pct (float): Probability of selecting a variable node.
            xover_pct (float): Crossover probability.
            save (bool): Whether to save log.
            operations (tuple): Custom operations (default uses +, -, *, neg).
        """
        _set_random_state(random_state)
        self.random_state = random_state
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.tournament_size = tournament_size or round(pop_size * 0.03)

        self.x_pct = x_pct
        self.xover_pct = xover_pct
        self.save = save

        self.global_best = float("inf")
        self.best_prog = None
        self.neuron = None

        if coefficient_range is None:
            self.coefficient_range = [-1, 1]
        else:
            self.coefficient_range = coefficient_range

        self.operations = operations or (
            {"func": operator.add, "arg_count": 2, "format_str": "({} + {})"},
            {"func": operator.sub, "arg_count": 2, "format_str": "({} - {})"},
            {"func": operator.mul, "arg_count": 2, "format_str": "({} * {})"},
            {"func": operator.neg, "arg_count": 1, "format_str": "-({})"},
        )

    def render_prog(self, node):
        if "children" not in node:
            return node["feature_name"]
        return node["format_str"].format(*[self.render_prog(c) for c in node["children"]])

    def simp(self, tree):
        return str(expand(sympify(self.render_prog(tree)))).replace("*", "@").replace('@@', '**')

    def evaluate(self, expr, x_data):
        x = x_data
        temp = re.split(' ', expr)
        for n, i_exp in enumerate(temp):
            if '@' in i_exp:
                index = i_exp.find('@')
                tem = list(i_exp)
                tem[index - 1] = str((eval(''.join(i_exp[0:index])) * np.ones((1, x_data.shape[0]))).tolist())
                del tem[0:index - 1]
                temp[n] = ''.join(tem)
        ex = ''.join(temp)
        return expr, eval(ex)

    def rand_w(self):
        return str(np.random.randint(low=self.coefficient_range[0], high=self.coefficient_range[1]))

    def random_prog(self, depth=0):
        n_d = depth
        op = self.operations[randint(0, len(self.operations) - 1)]
        if randint(0, 10) >= depth and n_d <= 6:
            n_d += 1
            return {
                "func": op["func"],
                "children": [self.random_prog(depth + 1) for _ in range(op["arg_count"])],
                "format_str": op["format_str"],
            }
        else:
            return {"feature_name": 'x'} if random() < self.x_pct else {"feature_name": self.rand_w()}

    def select_random_node(self, selected, parent, depth):
        if "children" not in selected:
            return parent
        if randint(0, 10) < 2 * depth:
            return selected
        child_count = len(selected["children"])
        return self.select_random_node(
            selected["children"][randint(0, child_count - 1)],
            selected, depth + 1)

    def do_mutate(self, selected):
        offspring = deepcopy(selected)
        mutate_point = self.select_random_node(offspring, None, 0)
        child_count = len(mutate_point["children"])
        mutate_point["children"][randint(0, child_count - 1)] = self.random_prog(0)
        return offspring

    def do_xover(self, selected1, selected2):
        offspring = deepcopy(selected1)
        xover_point1 = self.select_random_node(offspring, None, 0)
        xover_point2 = self.select_random_node(selected2, None, 0)
        child_count = len(xover_point1["children"])
        xover_point1["children"][randint(0, child_count - 1)] = xover_point2
        return offspring

    def get_random_parent(self, popu, fitne):
        tournament_members = [
            randint(0, self.pop_size - 1) for _ in range(self.tournament_size)]
        member_fitness = [(fitne[i], popu[i]) for i in tournament_members]
        return min(member_fitness, key=lambda x: x[0])[1]

    def get_offspring(self, popula, ftns):
        tempt = random()
        parent1 = self.get_random_parent(popula, ftns)
        if tempt < self.xover_pct:
            parent2 = self.get_random_parent(popula, ftns)
            return self.do_xover(parent1, parent2)
        elif self.xover_pct <= tempt < 0.9:
            return self.do_mutate(parent1)
        else:
            return parent1

    def node_count(self, x):
        if "children" not in x:
            return 1
        return sum([self.node_count(c) for c in x["children"]])

    def compute_fitness(self, func, pred, label):
        m = func.count('x')
        if m == 0 or m == 1:
            return float("inf")
        else:
            mse = np.mean(np.square(pred - label))
            return mse

    def fit(self, X, y):
        X = X.T
        y = y.T
        self.population = [self.random_prog() for _ in range(self.pop_size)]
        self.box = {}

        if self.save:
            file = open("log.txt", 'w')

        for gen in tqdm(range(self.max_generations), desc="Fitting Progress"):
            fitness = []
            for prog in self.population:
                func, prediction = self.evaluate(self.simp(prog), X)
                score = self.compute_fitness(func, prediction, y)
                fitness.append(score)

                if score < self.global_best:
                    self.global_best = score
                    self.best_prog = func

                if len(self.box) < self.pop_size * 0.05:
                    self.box[score] = prog
                else:
                    key_sort = sorted(self.box)
                    if score < key_sort[-1]:
                        self.box.pop(key_sort[-1])
                        self.box[score] = prog

            if self.save:
                file.write(
                    "Generation: {:d}\nBest Score: {:.4f}\nMedian score: {:.4f}\nBest program: {:s}\n\n"
                    .format(gen+1, self.global_best, np.median(np.array(fitness)), str(self.best_prog))
                )

            lst = list(self.box.values())
            self.population += lst
            shuffle(self.population)
            population_new = [self.get_offspring(self.population, fitness) for _ in range(self.pop_size)]
            self.population = population_new + lst

        self.best_score = self.global_best
        self.best_program = self.best_prog

        if self.save:
            file.write("Best score: %f\n" % self.best_score)
            file.write("Best program: %s\n" % self.best_program)
            file.close()

        self.neuron = self.best_program