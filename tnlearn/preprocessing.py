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

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


class DataPreprocessor:
    def __init__(self, num_features=None, cat_features=None, num_scaler='standard', feature_range=(0, 1)):
        r"""Initialize the preprocessor with options for numerical and categorical feature processing.

        Args:
            num_features: Numerical features.
            cat_features: Categorical features.
            num_scaler: Method to standardize data.
            feature_range: The range of data normalization.
        """
        # Numerical features to scale
        self.num_features = num_features
        # Categorical features to encode
        self.cat_features = cat_features
        # Scaler type: 'standard' for StandardScaler or 'minmax' for MinMaxScaler
        self.num_scaler = num_scaler
        # Feature range for MinMaxScaler
        self.feature_range = feature_range
        # List to store transformers
        self.transformers = []

        # Add numerical transformer if numerical features are specified
        if self.num_features:
            if self.num_scaler == 'minmax':
                scaler = MinMaxScaler(feature_range=self.feature_range)
            else:
                scaler = StandardScaler()
            self.transformers.append(("num", scaler, self.num_features))

        # Add categorical transformer if categorical features are specified
        if self.cat_features:
            self.transformers.append(("cat", OneHotEncoder(), self.cat_features))

        self.column_transformer = ColumnTransformer(transformers=self.transformers)

    def fit_transform(self, X, y=None):
        r"""Fit to data, then transform it.

        Args:
            X (array-like of shape (n_samples, n_features)): Input samples.
            y (array-like of shape (n_samples,) or (n_samples, n_outputs), default=None): Target values (None for unsupervised transformations).

        Returns:
            The fit data.
        """
        return self.column_transformer.fit_transform(X, y)

    def transform(self, X):
        r"""Perform standardization by centering and scaling.

        Args:
            X ({array-like, sparse matrix} of shape (n_samples, n_features)): The data used to scale along the features' axis.


        Returns:
            Transformed array.
        """
        return self.column_transformer.transform(X)


