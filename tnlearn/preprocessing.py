"""
Program name: Data Preprocessing Utilities
Purpose description: This script provides a robust and flexible DataPreprocessor class capable of
                     handling both numerical and categorical data transformations. With features for
                     standardizing or normalizing numerical features and encoding categorical features,
                     this class is an essential utility for preparing raw data for machine learning model
                     training. It consolidates preprocessing steps into a reusable framework, ensuring
                     consistency in data transformations across different machine learning workflows.
                     Users can customize the type of scalers and the feature range for normalization,
                     as well as selectively apply transformations to subsets of features.
Last revision date: February 21, 2024
Known Issues: None identified at the time of the last revision.
Note: This class is designed to work seamlessly with the sklearn pipeline, assuming that the required
      transformers from sklearn.preprocessing (such as StandardScaler, MinMaxScaler, and OneHotEncoder)
      are properly installed and imported before using this DataPreprocessor class.
"""


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


class DataPreprocessor:
    # Initialize the preprocessor with options for numerical and categorical feature processing
    def __init__(self, num_features=None, cat_features=None, num_scaler='standard', feature_range=(0, 1)):
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

    # Fit the transformers to the data and also transform the data
    def fit_transform(self, X, y=None):
        return self.column_transformer.fit_transform(X, y)

    # Transform the data using the already fitted transformers
    def transform(self, X):
        return self.column_transformer.transform(X)


