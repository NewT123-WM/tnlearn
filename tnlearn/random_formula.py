"""Seeded random neuron expressions, without fitting or changing global RNGs.

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

from copy import deepcopy
from numbers import Integral
import random
import re

import numpy as np
from sympy import expand, sympify

from .operator.inner_product import InnerProduct, _simplify_expr

__all__ = ["RandomFormulaGenerator", "generate_for_combo", "count_terms"]


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _seed(value):
    if value is None:
        return None
    value = _integer("random_state", value)
    if value >= 2**32:
        raise ValueError("random_state must be less than 2**32")
    return value


class RandomFormulaGenerator:
    """Generate polynomial/inner-product expression trees without using data.

    Parameters
    ----------
    max_depth : int, default=4
        Maximum construction depth, with the root at depth zero.
    coefficient_range : pair of float, default=(-1, 1)
        Interval for numerical leaves. When it straddles zero, sampling
        excludes (-1e-4, 1e-4).
    x_pct : float, default=0.7
        Probability of choosing x rather than a constant at a leaf.
    random_state : int or None, default=None
        Seed for private Python and NumPy RNGs. Equal seeds and calls produce
        equal expressions in the same Python/NumPy/SymPy environment.

    Notes
    -----
    Operators are addition, subtraction, multiplication, negation and inner
    product. Expression trees are varied through mutation and crossover.
    Expressions are strings accepted by the base MLP neuron implementation.
    Numerical stability or useful predictive performance is not guaranteed.
    """

    _OPERATIONS = (("({} + {})", 2), ("({} - {})", 2),
                   ("({} * {})", 2), ("-({})", 1),
                   ("InnerProduct({}, {})", 2))

    def __init__(self, max_depth=4, coefficient_range=(-1, 1), x_pct=0.7,
                 random_state=None):
        self.max_depth = _integer("max_depth", max_depth, 1)
        if len(coefficient_range) != 2:
            raise ValueError("coefficient_range must contain two bounds")
        low, high = map(float, coefficient_range)
        if not np.isfinite([low, high]).all() or not low < high:
            raise ValueError("coefficient_range must be finite and increasing")
        if low < 0 < high and not (low <= -1e-4 and high >= 1e-4):
            raise ValueError("a range spanning zero must include [-1e-4, 1e-4]")
        if not np.isfinite(x_pct) or not 0 <= x_pct <= 1:
            raise ValueError("x_pct must be between 0 and 1")
        self.coefficient_range = (low, high)
        self.x_pct = float(x_pct)
        self.random_state = _seed(random_state)
        self._random = random.Random(self.random_state)
        # Keep a private MT19937 stream for coefficient sampling.
        self._numpy = np.random.RandomState(self.random_state)

    def _coefficient(self):
        low, high = self.coefficient_range
        if low < 0 < high:
            if self._random.random() < 0.5:
                high = -1e-4
            else:
                low = 1e-4
        return str(self._numpy.uniform(low, high))

    def _tree(self, depth=0, max_depth=None):
        limit = self.max_depth if max_depth is None else max_depth
        if depth >= limit:
            return {"leaf": "x" if self._random.random() < self.x_pct
                    else self._coefficient()}
        template, arity = self._OPERATIONS[self._random.randint(0, 4)]
        return {"template": template,
                "children": [self._tree(depth + 1, limit) for _ in range(arity)]}

    def _depth(self, node):
        return (1 + max(self._depth(child) for child in node["children"])
                if "children" in node else 1)

    def _select(self, node, parent=None, depth=0):
        if "children" not in node:
            return parent
        if self._random.randint(0, 10) < 2 * depth:
            return node
        return self._select(node["children"][self._random.randint(
            0, len(node["children"]) - 1)], node, depth + 1)

    def _mutate(self, tree):
        for _ in range(10):
            offspring = deepcopy(tree)
            point = self._select(offspring)
            point["children"][self._random.randint(0, len(point["children"]) - 1)] = (
                self._tree(max_depth=2)
            )
            if self._depth(offspring) <= self.max_depth:
                return offspring
        return self._tree()

    def _crossover(self, left, right):
        for _ in range(10):
            offspring = deepcopy(left)
            point, donor = self._select(offspring), self._select(right)
            point["children"][self._random.randint(0, len(point["children"]) - 1)] = deepcopy(donor)
            if self._depth(offspring) <= self.max_depth:
                return offspring
        return self._tree()

    def _render(self, tree):
        if "leaf" in tree:
            return tree["leaf"]
        return tree["template"].format(*(self._render(c) for c in tree["children"]))

    def generate_formula(self, mutations=3, crossovers=2):
        """Return one simplified expression and advance this instance's RNGs.

        mutations and crossovers must be nonnegative integers.
        No scores or data enter generation.
        """
        mutations = _integer("mutations", mutations)
        crossovers = _integer("crossovers", crossovers)
        tree = self._tree()
        for _ in range(mutations):
            tree = self._mutate(tree)
        for _ in range(crossovers):
            tree = self._crossover(tree, self._tree())
        expression = str(_simplify_expr(expand(sympify(
            self._render(tree), locals={"InnerProduct": InnerProduct}))))
        while "InnerProduct(" in expression:
            updated = re.sub(r"InnerProduct\(([^()]*)\)", r"<\1>", expression)
            if updated == expression:
                raise ValueError("Unable to render nested inner-product expression")
            expression = updated
        return expression


def count_terms(formula):
    """Count sign-delimited terms outside inner-product brackets.

    Signs inside <...> are ignored; parentheses outside <...> are not special.
    This is a string complexity measure, not the number of expanded monomials.
    """
    if not isinstance(formula, str):
        raise TypeError("formula must be a string")
    terms, current, depth = [], "", 0
    for char in formula.strip():
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced inner-product brackets")
        elif depth == 0 and char in "+-" and current.strip():
            terms.append(current)
            current = ""
        current += char
    if depth:
        raise ValueError("unbalanced inner-product brackets")
    return len(terms) + bool(current.strip())


def generate_for_combo(term_target, x_target, count=1, max_attempts=10000,
                       random_state=0):
    """Return candidates matching term and literal-x counts.

    Candidate seeds are derived deterministically from random_state.
    Duplicates are retained; callers may deduplicate candidates explicitly.
    The search returns fewer than count candidates if its budget is exhausted.
    No Python or NumPy global random state is modified.
    """
    term_target = _integer("term_target", term_target, 1)
    x_target = _integer("x_target", x_target)
    count = _integer("count", count, 1)
    max_attempts = _integer("max_attempts", max_attempts)
    seed = _seed(random_state)
    if seed is None:
        raise TypeError("generate_for_combo requires an integer random_state")
    formulas = []
    for attempt in range(max_attempts):
        generator = RandomFormulaGenerator(
            random_state=(seed + 100 * (attempt + len(formulas))) % 2**32)
        formula = generator.generate_formula(mutations=10, crossovers=4)
        if count_terms(formula) == term_target and formula.count("x") == x_target:
            formulas.append(formula)
            if len(formulas) == count:
                break
    return formulas
