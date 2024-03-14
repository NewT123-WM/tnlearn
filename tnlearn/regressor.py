"""
Program name: Vectorized Symbolic Regression with VecSymRegressor Class
Purpose description: This script implements the symbolic regression algorithm through
                     the VecSymRegressor class, enabling the evolution of mathematical expressions
                     to fit given data. The class provides methods for generating random expressions,
                     evaluating their fitness, and evolving expressions through mutation and crossover.
                     It aims to find the best-fitting mathematical model for a given dataset.
Last revision date: February 27, 2024
Known Issues: None identified at the time of the last revision.
Note: This overview assumes that the VecSymRegressor class and all its dependencies are properly installed
      and functional.
"""

import numpy as np
from sympy import sympify, expand
import operator
from random import randint, random, shuffle
from copy import deepcopy
import re
from tqdm import tqdm
from tnlearn.utils import random_seed


class VecSymRegressor:
    def __init__(self,
                 random_state=100,
                 pop_size=5000,
                 max_generations=20,
                 tournament_size=10,
                 x_pct=0.7,
                 xover_pct=0.3,
                 save=False,
                 operations=None):

        """
        # Set default values for various parameters if not provided
        # Parameters:
        - random_state: Seed for random number generation
        - pop_size: Population size for genetic algorithm
        - max_generations: Maximum generations for genetic algorithm
        - tournament_size: Size of tournament selection
        - x_pct: Probability of selecting a variable node during random program generation
        - xover_pct: Crossover probability during offspring generation
        - save: Flag for saving
        - operations: Set of operations to be used in program generation
        """

        random_seed(random_state)
        self.random_state = random_state
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.tournament_size = tournament_size or round(pop_size * 0.03)
        self.x_pct = x_pct
        self.xover_pct = xover_pct
        self.save = save

        # Initializing the global best fitness value to infinity
        self.global_best = float("inf")
        self.best_prog = None
        self.neuron = None

        # Defining mathematical operations for the algorithm, if not provided
        self.operations = operations or (
            {"func": operator.add, "arg_count": 2, "format_str": "({} + {})"},
            {"func": operator.sub, "arg_count": 2, "format_str": "({} - {})"},
            {"func": operator.mul, "arg_count": 2, "format_str": "({} * {})"},
            {"func": operator.neg, "arg_count": 1, "format_str": "-({})"},
        )

    # Method to render a program based on a given tree structure
    # Recursively converts the tree into a string expression
    def render_prog(self, node):

        if "children" not in node:
            return node["feature_name"]
        return node["format_str"].format(*[self.render_prog(c) for c in node["children"]])

    # Method to simplify a tree-based expression into a string
    # Uses sympify and expand functions to simplify and convert the expression
    def simp(self, tree):
        return str(expand(sympify(self.render_prog(tree)))).replace("*", "@").replace('@@', '**')

    # Method to evaluate an expression using input data
    # Parses the expression and evaluates it using input x_data
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

    # Static method to generate a random weight within a specified range
    @staticmethod
    def rand_w():
        return str(round(np.random.uniform(-1, 1), 2))
        # return str(np.random.randint(low=-20, high=20))

    # Method to generate a random program/tree structure
    # Uses recursion to generate a random tree based on specified depth and operations
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

    # Method to select a random node within a given tree structure
    # Uses recursion to randomly select a node, favoring nodes near the root
    def select_random_node(self, selected, parent, depth):
        if "children" not in selected:
            return parent
            # favor nodes near the root
        if randint(0, 10) < 2 * depth:
            return selected
        child_count = len(selected["children"])
        return self.select_random_node(
            selected["children"][randint(0, child_count - 1)],
            selected, depth + 1)

    # Method to perform mutation on a selected tree structure
    # Deep copies the selected tree and mutates a random node within it
    def do_mutate(self, selected):  # 修改成实例方法
        offspring = deepcopy(selected)
        mutate_point = self.select_random_node(offspring, None, 0)
        child_count = len(mutate_point["children"])
        mutate_point["children"][randint(0, child_count - 1)] = self.random_prog(0)
        return offspring

    # Method to perform crossover between two selected tree structures
    # Deep copies the first selected tree and swaps a random node with a node from the second selected tree
    def do_xover(self, selected1, selected2):
        offspring = deepcopy(selected1)
        xover_point1 = self.select_random_node(offspring, None, 0)
        xover_point2 = self.select_random_node(selected2, None, 0)
        child_count = len(xover_point1["children"])
        xover_point1["children"][randint(0, child_count - 1)] = xover_point2
        return offspring

    # Method to select a random parent from a population based on fitness
    # Uses tournament selection to select random members and returns the one with the lowest fitness
    def get_random_parent(self, popu, fitne):
        # randomly select population members for the tournament
        tournament_members = [
            randint(0, self.pop_size - 1) for _ in range(self.tournament_size)]
        # select tournament member with best fitness
        member_fitness = [(fitne[i], popu[i]) for i in tournament_members]
        return min(member_fitness, key=lambda x: x[0])[1]

    # Method to generate offspring based on the given population and fitness
    # Randomly selects parents and performs crossover or mutation based on probabilities
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

    # Method to count the number of nodes in a given tree structure
    # Recursively counts the nodes within the tree
    def node_count(self, x):
        if "children" not in x:
            return 1
        return sum([self.node_count(c) for c in x["children"]])

    # Method to compute fitness based on the function, predictions, and labels
    # Calculates mean squared error (MSE) as fitness metric
    def compute_fitness(self, func, pred, label):
        m = func.count('x')
        if m == 0 or m == 1:
            return float("inf")
        else:
            mse = np.mean(np.square(pred - label))
            return mse

    # The symbolic regression algorithm is performed to fit the data.
    def fit(self, X, y):
        X = X.T
        y = y.T
        self.population = [self.random_prog() for _ in range(self.pop_size)]
        self.box = {}

        # If save option is enabled, open a file to log the progress
        if self.save:
            file = open("log.txt", 'w')

        # Iterate over the specified number of generations
        for gen in tqdm(range(self.max_generations), desc="Fitting Progress"):
            # Store the fitness scores of each program
            fitness = []
            # Evaluate each program and calculate its fitness score
            for prog in self.population:
                func, prediction = self.evaluate(self.simp(prog), X)
                score = self.compute_fitness(func, prediction, y)
                fitness.append(score)

                # Update global best program if the current one performs better
                if score < self.global_best:
                    self.global_best = score
                    self.best_prog = func

                # Keep track of the best performing programs in the box
                if len(self.box) < self.pop_size * 0.05:
                    self.box[score] = prog
                else:
                    key_sort = sorted(self.box)
                    if score < key_sort[-1]:
                        self.box.pop(key_sort[-1])
                        self.box[score] = prog

            # If saving, write the generation number and best program to the log file
            if self.save:
                file.write(
                    "Generation: {:d}\nBest Score: {:.4f}\nMedian score: {:.4f}\nBest program: {:s}\n\n"
                    .format(gen+1, self.global_best, np.median(np.array(fitness)), str(self.best_prog))
                )

            # Add a selected portion of best programs to the population
            lst = list(self.box.values())
            self.population += lst

            # Shuffle the population to introduce randomness
            shuffle(self.population)

            # Generate new programs as offspring
            population_new = [self.get_offspring(self.population, fitness) for _ in range(self.pop_size)]

            # Update the population with the new offspring and the best programs
            self.population = population_new + lst

        # Store the best score and best program
        self.best_score = self.global_best
        self.best_program = self.best_prog

        # If saving, write the best program to the log file and close it
        if self.save:
            # 记录最终的最优解
            file.write("Best score: %f\n" % self.best_score)
            file.write("Best program: %s\n" % self.best_program)
            file.close()

        # Set the attribute 'neuron' to the best program
        self.neuron = self.best_program

