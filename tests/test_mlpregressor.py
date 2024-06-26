"""
Program name: MLPRegressor Class Testing
Purpose description: This program is intended to test the capabilities of the MLPRegressor
                     class from the 'tnlearn' library on synthetic regression data
                     generated by the 'make_regression' function from 'sklearn'.
                     It includes data preprocessing using a custom DataPreprocessor class.
Tests: This script has been tested on synthetic datasets generated by `make_regression` in sklearn.
Note: This script assumes that the tnlearn library's MLPRegressor and DataPreprocessor classes
      are properly implemented. The sklearn library is used to split the data and scale the features.
"""


from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from tnlearn import MLPRegressor
from tnlearn import DataPreprocessor

# Generate a synthetic regression dataset with 100 samples and 3 features
X, y = make_regression(n_samples=100, n_features=3, noise=0.1)
y = y.reshape(-1, 1)  # Reshape 'y' to have the shape (n_samples, n_targets)

# Split the dataset into a training set (80%) and a test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Indices of numerical features in the dataset
num_features = [0, 1, 2]  # Indexes corresponding to the columns in 'X'

# Create an instance of the DataPreprocessor class for preprocessing
preprocessor = DataPreprocessor(num_features=num_features)

# Fit the preprocessor on the training data and transform the training  set
X_train = preprocessor.fit_transform(X_train)

# Transform test data only
X_test = preprocessor.transform(X_test)

# Instantiate the MLPRegressor class with specified parameters
mlp_regressor = MLPRegressor(neurons='x**2',
                             layers_list=[50, 30, 10],
                             activation_funcs='sigmoid',
                             loss_function='mse',
                             optimizer_name='adam',
                             max_iter=200,
                             batch_size=16,
                             )

# Train the MLP regressor model on the preprocessed training data
mlp_regressor.fit(X_train, y_train)

# Evaluate the model's performance on the test set
test_score = mlp_regressor.score(X_test, y_test)

# Perform predictions using the trained MLP regressor on the test data
predictions = mlp_regressor.predict(X_test)
