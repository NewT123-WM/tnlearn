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
from tnlearn import MLPClassifier
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import neural_network
from sklearn.neighbors import KNeighborsClassifier

dataset_loader = DatasetLoader()
X, X_test, y, y_test = dataset_loader.load_eye()

print('=' * 50)
print('Data loading complete')
print('=' * 50)

# =================================================================

# TNLearn Classification
print('TNLearn Classification:')

params = {
    "neurons": '-0.02@x**3 - 0.01@x**2',
    "layers_list": [100, 50],
    "activation_funcs": 'sigmoid',
    "loss_function": 'crossentropy',
    "random_state": 1,
    "max_iter": 400,
    "batch_size": 128,
    "lr": 0.01,

}

clf = MLPClassifier(**params)
clf.fit(X, y)
print("Accuracy: %.3f" % clf.score(X_test, y_test))
# ===================================================================

# Logistic regression Classification
print('=' * 50)
print('Logistic regression Classification:')

log_reg = linear_model.LogisticRegression()
log_reg.fit(X, y)
print("Accuracy: %.3f" % log_reg.score(X_test, y_test))
# ===================================================================

# K-Nearest Neighbors Classification
print('=' * 50)
print('K-Nearest Neighbors Classification:')

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
print("Accuracy: %.3f" % knn.score(X_test, y_test))
# ===================================================================

# Support Vector Machines Classification
print('=' * 50)
print('Support Vector Machines Classification:')

svm = svm.SVC()
svm.fit(X, y)
print("Accuracy: %.3f" % svm.score(X_test, y_test))
# ===================================================================

# Decision Tree Classification
print('=' * 50)
print('Decision Tree Classification:')

tree_cls = DecisionTreeClassifier(max_depth=5)
tree_cls.fit(X, y)
print("Accuracy: %.3f" % tree_cls.score(X_test, y_test))
# ===================================================================

# MLP Classification
print('=' * 50)
print('MLP Classification:')

params = {
    "random_state": 1,
    "max_iter": 500,
    "solver": 'adam',
    "learning_rate_init": 0.01,
    "batch_size": 128,
}

mlp_cls = neural_network.MLPClassifier(**params)
mlp_cls.fit(X, y)
print("Accuracy: %.3f" % mlp_cls.score(X_test, y_test))
