"""
Program name: DataPreprocessor Class Testing
Purpose description: This program is designed to verify the functionality of the DataPreprocessor
                     class, which is a custom utility created to streamline preprocessing of
                     numerical and categorical data for machine learning models. The program
                     checks the preprocessing steps such as scaling for numerical features and
                     one-hot encoding for categorical features.
                     The dataset used consists of randomly generated numerical and categorical data.
Last revision date: February 18, 2024
Tests: The test suite ensures that the transformed data has the appropriate scaled values and
       one-hot encoded vectors across different scaler types—StandardScaler and MinMaxScaler.
       Additionally, tests verify that the transformed train and test datasets have consistent
       dimensions and expected statistical properties after transformation.
Known Issues: None identified within the scope of numeric and categorical feature preprocessing.
Note: The testing script presumes the proper functioning of the DataPreprocessor class, its integration
      with scikit-learn utilities like train_test_split, and correct setup of the test environment
      including dependency management for numpy, pandas, and scikit-learn.
"""

import numpy as np
import pandas as pd
from tnlearn import DataPreprocessor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


def test_data_preprocessor():
    # For reproducibility of random values
    np.random.seed(0)

    # Generate synthetic numerical feature data
    num_features = ['num_feature1', 'num_feature2']  # Names of numerical features
    cat_features = ['cat_feature']  # Names of categorical features
    num_data = np.random.rand(100, 2)  # Random numerical data for 100 samples and 2 features
    cat_data = np.random.choice(['cat1', 'cat2', 'cat3'], size=(100, 1))  # Categorical data for 100 samples

    # Combine numerical and categorical data
    features_data = np.concatenate([num_data, cat_data], axis=1)
    features_df = pd.DataFrame(features_data, columns=num_features + cat_features)  # Create DataFrame for features

    # Generate synthetic target variable
    target_data = np.random.choice(['label1', 'label2', 'label3'], size=(100, 1))  # Random target labels
    target_df = pd.DataFrame(target_data, columns=['target'])  # DataFrame for labels

    # Combine feature DataFrame and target DataFrame
    df = pd.concat([features_df, target_df], axis=1)

    # Split DataFrame into features and target
    features = df[num_features + cat_features]
    target = df['target']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

    # Test DataPreprocessor class with different scalers for numerical features
    for scaler_type in ['standard', 'minmax']:
        # Initialize DataPreprocessor with numerical and categorical features and scaler type
        preprocessor = DataPreprocessor(num_features=num_features, cat_features=cat_features,
                                        num_scaler=scaler_type)

        # Fit the preprocessor to the training data and transform the training data
        X_train_preprocessed = preprocessor.fit_transform(X_train)

        """
        If the label is also a string type, it can be encoded.
        
        from sklearn.preprocessing import LabelEncoder
        # Initialize LabelEncoder
        label_encoder = LabelEncoder()

        # Fit LabelEncoder on the target data and transform training and test target data
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)        
        """

        # Assert checks if conditions are met, if not, raises an AssertionError
        # Check that the number of features after transforming matches the expected number
        expected_num_features = len(num_features)  # Expected number of numerical features after scaling
        expected_cat_features = len(np.unique(cat_data))  # Expected number of one-hot encoded features
        total_expected_features = expected_num_features + expected_cat_features
        assert X_train_preprocessed.shape[1] == total_expected_features

        # Retrieve the numerical scaler used in the preprocessor and perform checks
        num_scaler = preprocessor.column_transformer.transformers_[0][1]

        if scaler_type == 'standard':
            # For StandardScaler, check if the scaler is of correct instance type
            assert isinstance(num_scaler, StandardScaler)
            # Check if the training data mean is approximately zero after scaling
            assert np.allclose(X_train_preprocessed.mean(axis=0)[:expected_num_features], 0, atol=1e-2)
        else:
            # For MinMaxScaler, check if the scaler is of correct instance type
            assert isinstance(num_scaler, MinMaxScaler)

            # Check if the training data range is within the specified feature range after scaling
            features_min, features_max = preprocessor.feature_range
            assert np.all(X_train_preprocessed.min(axis=0)[:expected_num_features] >= features_min - 1e-2)
            assert np.all(X_train_preprocessed.max(axis=0)[:expected_num_features] <= features_max + 1e-2)

        # Check if a OneHotEncoder is used for categorical features
        assert isinstance(preprocessor.column_transformer.transformers_[-1][1], OneHotEncoder)

        # Confirm that one-hot encoding on the training categorical data produces expected results
        encoder = preprocessor.column_transformer.transformers_[-1][1]
        cat_encoded_actual = encoder.transform(X_train[cat_features]).toarray()
        assert np.allclose(X_train_preprocessed[:, -expected_cat_features:], cat_encoded_actual)

        # Transform the test data using the already fitted preprocessor
        X_test_preprocessed = preprocessor.transform(X_test)

        # Ensure the preprocessed test data has the correct number of features
        assert X_test_preprocessed.shape[1] == total_expected_features

        # For the transformed test data, we check that the data still conforms to the expectations from the scalers
        if scaler_type == 'standard':
            # For standard scaling, we expect the test data means to be close to zero, but not exactly zero
            assert np.allclose(X_test_preprocessed.mean(axis=0)[:expected_num_features], 0, atol=1)
        else:
            # For min-max scaling, we expect the test data range to be approximately within [0, 1]
            features_min, features_max = preprocessor.feature_range
            assert np.all(X_test_preprocessed.min(axis=0)[:expected_num_features] >= features_min - 1)
            assert np.all(X_test_preprocessed.max(axis=0)[:expected_num_features] <= features_max + 1)

        # Confirm that one-hot encoding on the test categorical data produces expected results
        cat_encoded_actual_test = encoder.transform(X_test[cat_features]).toarray()
        assert np.allclose(X_test_preprocessed[:, -expected_cat_features:], cat_encoded_actual_test)


# Execute the test function
test_data_preprocessor()
