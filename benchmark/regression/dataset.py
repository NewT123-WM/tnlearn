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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn import datasets


class DatasetLoader:
    def load_space(self, test_size=0.2, seed=100):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        ori_data = datasets.fetch_openml('space_ga', version=5)  # (3107, 6)
        X_all = np.array(ori_data.data, dtype=float)
        y_all = np.array(ori_data.target, dtype=float).reshape((-1, 1))
        X_all = scaler.fit_transform(X_all)
        y_all = scaler.fit_transform(y_all)
        X, X_test, y, y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=seed)
        return X, X_test, y, y_test

    def load_noise(self, test_size=0.2, seed=100):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        ori_data = datasets.fetch_openml('airfoil_self_noise', version=1)  # (1503, 5)
        X_all = np.array(ori_data.data, dtype=float)
        y_all = np.array(ori_data.target, dtype=float).reshape((-1, 1))
        X_all = scaler.fit_transform(X_all)
        y_all = scaler.fit_transform(y_all)
        X, X_test, y, y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=seed)
        return X, X_test, y, y_test

    def load_housing(self, test_size=0.2, seed=100):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        ori_data = datasets.fetch_california_housing()  # (20640, 8)
        X_all = ori_data.data
        y_all = ori_data.target.reshape(-1, 1)
        X_all = scaler.fit_transform(X_all)
        y_all = scaler.fit_transform(y_all)
        X, X_test, y, y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=seed)
        return X, X_test, y, y_test
