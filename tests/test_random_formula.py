"""Regression and isolation tests for the random neuron public API."""

import random
import unittest
from unittest.mock import patch

import numpy as np

from tnlearn import RandomFormulaGenerator, generate_for_combo
from tnlearn.random_formula import count_terms


class RandomFormulaTests(unittest.TestCase):
    def test_private_random_streams_and_repeatability(self):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        left = RandomFormulaGenerator(random_state=42)
        right = RandomFormulaGenerator(random_state=42)
        self.assertEqual([left.generate_formula() for _ in range(3)],
                         [right.generate_formula() for _ in range(3)])
        self.assertEqual(python_state, random.getstate())
        current = np.random.get_state()
        self.assertEqual(numpy_state[0], current[0])
        np.testing.assert_array_equal(numpy_state[1], current[1])
        self.assertEqual(numpy_state[2:], current[2:])

    def test_candidate_filtering_and_early_stop(self):
        with patch('tnlearn.random_formula.RandomFormulaGenerator') as generator:
            generate = generator.return_value.generate_formula
            generate.side_effect = ['x', 'x + 1', '<x, x> + 1', 'x - x']
            candidates = generate_for_combo(2, 2, count=2, max_attempts=6,
                                            random_state=17)
        self.assertEqual(candidates, ['<x, x> + 1', 'x - x'])
        self.assertEqual(generate.call_count, 4)

    def test_candidate_budget_and_duplicates(self):
        with patch('tnlearn.random_formula.RandomFormulaGenerator') as generator:
            generate = generator.return_value.generate_formula
            generate.return_value = 'x + 1'
            candidates = generate_for_combo(2, 1, count=3, max_attempts=2)
        self.assertEqual(candidates, ['x + 1', 'x + 1'])
        self.assertEqual(generate.call_count, 2)

    def test_zero_budget_skips_generation(self):
        with patch('tnlearn.random_formula.RandomFormulaGenerator') as generator:
            self.assertEqual(generate_for_combo(1, 1, max_attempts=0), [])
        generator.assert_not_called()

    def test_term_counting(self):
        for formula, expected in [('', 0), ('-x', 1), ('x + 1', 2),
                                  ('<x - 1, <x, x>> + x', 2)]:
            with self.subTest(formula=formula):
                self.assertEqual(count_terms(formula), expected)

    def test_bounds_and_invalid_inputs(self):
        self.assertEqual(generate_for_combo(999, 999, max_attempts=1), [])
        with self.assertRaises(ValueError):
            RandomFormulaGenerator(max_depth=0)
        with self.assertRaises(ValueError):
            RandomFormulaGenerator(x_pct=1.1)
        with self.assertRaises(ValueError):
            RandomFormulaGenerator(coefficient_range=(1, -1))
        with self.assertRaises(TypeError):
            generate_for_combo(4, 5, random_state=None)
        with self.assertRaises(TypeError):
            generate_for_combo(4, 5, count=True)
        with self.assertRaises(ValueError):
            count_terms('<x, x')


if __name__ == '__main__':
    unittest.main()
