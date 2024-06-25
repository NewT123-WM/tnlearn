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
import pandas as pd


class DatasetLoader:
    def load_eye(self, test_size=0.2, seed=100):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        ori_data = datasets.fetch_openml('eye_movements', version=7)  # (7608, 20, 2)
        ori_data.target = pd.factorize(ori_data.target)[0].astype(float)
        X_all = np.array(ori_data.data, dtype=float)
        X_all = scaler.fit_transform(X_all)
        y_all = ori_data.target
        X, X_test, y, y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=seed)
        return X, X_test, y, y_test

    def load_oranges(self, test_size=0.2, seed=100):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        ori_data = datasets.fetch_openml('Oranges-vs.-Grapefruit', version=1)  # (10000, 5, 2)
        data = ori_data.data.iloc[:, 1:]
        target = pd.factorize(ori_data.data.iloc[:, 0])[0].astype(float)
        X_all = np.array(data, dtype=float)
        X_all = scaler.fit_transform(X_all)
        y_all = target
        X, X_test, y, y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=seed)
        return X, X_test, y, y_test

    def load_electricity(self, test_size=0.2, seed=100):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        ori_data = datasets.fetch_openml('electricity', version=13)  # (38474, 8, 2)
        ori_data.target = pd.factorize(ori_data.target)[0].astype(float)
        X_all = np.array(ori_data.data, dtype=float)
        X_all = scaler.fit_transform(X_all)
        y_all = ori_data.target
        X, X_test, y, y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=seed)
        return X, X_test, y, y_test




