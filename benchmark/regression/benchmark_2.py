# Copyright 2024 Meng WANG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
    The expressions of nonlinear neurons have been obtained in advance
    by vectorized symbolic regression (VecSymRegressor Class).
"""

from dataset import DatasetLoader
from tnlearn import MLPRegressor
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn import neural_network

dataset_loader = DatasetLoader()
X, X_test, y, y_test = dataset_loader.load_noise()

print('=' * 50)
print('Data loading complete')
print('=' * 50)

# =================================================================

# TNLearn Regression
print('TNLearn Regression:')

params = {
    "neurons": '0.06@x**2 - 0.03@x - 0.08',
    "layers_list": [10, 20, 10, 10],
    "activation_funcs": 'sigmoid',
    "loss_function": 'mse',
    "max_iter": 300,
    "batch_size": 10,
    "lr": 0.01,
}

clf = MLPRegressor(**params)
clf.fit(X, y)
print("Mean squared error: %.4f" % mean_squared_error(y_test, clf.predict(X_test)))
print("Coefficient of determination: %.3f" % clf.score(X_test, y_test))
# ===================================================================

# Linear Regression
print('=' * 50)
print('Linear Regression:')

linear_reg = linear_model.LinearRegression()
linear_reg.fit(X, y)
print("Mean squared error: %.4f" % mean_squared_error(y_test, linear_reg.predict(X_test)))
print("Coefficient of determination: %.3f" % linear_reg.score(X_test, y_test))
# ===================================================================

# Ridge Regression
print('=' * 50)
print('Ridge Regression:')


ridge_reg = linear_model.Ridge(alpha=.5)
ridge_reg.fit(X, y)
print("Mean squared error: %.4f" % mean_squared_error(y_test, ridge_reg.predict(X_test)))
print("Coefficient of determination: %.3f" % ridge_reg.score(X_test, y_test))
# ===================================================================

# Support Vector Regression
print('=' * 50)
print('Support Vector Regression:')

# Support Vector Regression
svm_reg = svm.SVR()
svm_reg.fit(X, y.ravel())
print("Mean squared error: %.4f" % mean_squared_error(y_test, svm_reg.predict(X_test)))
print("Coefficient of determination: %.3f" % svm_reg.score(X_test, y_test))
# ===================================================================

# Decision tree regression
print('=' * 50)
print('Decision tree regression:')

tree_reg = DecisionTreeRegressor(max_depth=6)
tree_reg.fit(X, y)
print("Mean squared error: %.4f" % mean_squared_error(y_test, tree_reg.predict(X_test)))
print("Coefficient of determination: %.3f" % tree_reg.score(X_test, y_test))
# ===================================================================

# Gradient Boosting regression
print('=' * 50)
print('Gradient Boosting regression:')

params = {
    "n_estimators": 500,
    "max_depth": 5,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
}

boosting_reg = GradientBoostingRegressor(**params)
boosting_reg.fit(X, y.ravel())
print("Mean squared error: %.4f" % mean_squared_error(y_test, boosting_reg.predict(X_test)))
print("Coefficient of determination: %.3f" % boosting_reg.score(X_test, y_test))
# ===================================================================

# Random forest regression
print('=' * 50)
print('Random forest regression:')

forest_reg = RandomForestRegressor(n_estimators=100, max_depth=5)
forest_reg.fit(X, y.ravel())
print("Mean squared error: %.4f" % mean_squared_error(y_test, forest_reg.predict(X_test)))
print("Coefficient of determination: %.3f" % forest_reg.score(X_test, y_test))
# ===================================================================

# MLP Regression
print('=' * 50)
print('MLP Regression:')

params = {
    "random_state": 1,
    "max_iter": 500,
    "solver": 'adam',
    "learning_rate_init": 0.01,
    "batch_size": 10,
}

regr = neural_network.MLPRegressor(**params)
regr.fit(X, y.ravel())
print("Mean squared error: %.4f" % mean_squared_error(y_test, regr.predict(X_test)))
print("Coefficient of determination: %.3f" % regr.score(X_test, y_test))
