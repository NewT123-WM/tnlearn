"""
Program name: VecSymRegressor Class Testing
Purpose description: This program is for unit testing the VecSymRegressor class, ensuring the correctness
                     of its evaluation, simplification, and random weight generation methods. The VecSymRegressor
                     class utilizes evolutionary algorithms for regression tasks, and this testing suite
                     aims to verify functional integrity across its methods using predefined conditions and parameters.
Tests: This test suite covers the main functional aspects of the VecSymRegressor class using simple assertions
       and is not reliant on synthetic datasets from sklearn's make_regression.
Note: This testing program assumes the correct implementation of a hypothetical VecSymRegressor class which is
      not part of the sklearn library. It specifically checks that the methods behave as expected when
      called with reasonable inputs.
"""

import unittest
import numpy as np
from tnlearn import VecSymRegressor


# Custom VecSymRegressor testing class inheriting from TestCase in the unittest module
class TestRegressor(unittest.TestCase):

    # Set up function to initialize the regressor object before each test
    def setUp(self):
        # The VecSymRegressor is initialized with various hyperparameters
        self.regressor = VecSymRegressor(random_state=100,
                                         pop_size=5000,
                                         max_generations=30,
                                         tournament_size=10,
                                         x_pct=0.7,
                                         xover_pct=0.3,
                                         save=False)

    # Test to evaluate the functionality of the evaluate method
    def test_evaluate(self):
        expr, _ = self.regressor.evaluate("x**2", np.array([1, 2, 3, 4, 5]))
        # Check if the returned expression equals the expected value
        self.assertEqual(expr, "x**2")

    # Test to evaluate the simplification of expressions
    def test_simp(self):
        tree = {"feature_name": 'x'}
        simp_expr = self.regressor.simp(tree)
        self.assertEqual(simp_expr, "x")

    # Test to check return type of random weight generation function
    def test_rand_w(self):
        self.assertIsInstance(self.regressor.rand_w(), str)


# Standard boilerplate to run unittest test suite when executing the script
if __name__ == '__main__':
    unittest.main()
