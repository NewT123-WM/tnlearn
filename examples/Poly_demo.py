"""Search a NeuronSeek structure and pass its expression to the base MLP."""

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tnlearn import MLPRegressor, PolyTensorRegressor
from tnlearn.operator.inner_product import neuronseek_config_to_string


if __name__ == '__main__':
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    search = PolyTensorRegressor(rank=3, poly_order=3, num_epochs=30, random_state=1)
    search.fit(X_train, y_train)
    neuron = neuronseek_config_to_string(search.structure_) or '0'
    assert neuron == search.neuron
    print('Structure:', search.structure_)
    print('Neuron:', neuron)

    mlp = MLPRegressor(neuron, layers_list=[10], max_iter=100)
    mlp.fit(X_train, y_train)
    print('Test R2:', mlp.score(X_test, y_test))
