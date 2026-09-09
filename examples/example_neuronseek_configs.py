"""Fit the four manual configurations using MLPRegressor's default settings."""

import numpy as np
from sklearn.datasets import make_regression

from tnlearn import MLPRegressor
from tnlearn.operator.inner_product import neuronseek_config_to_string


if __name__ == '__main__':
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=1)
    configs = [
        dict(type='neuronseek', pure_indices=[1, 2], interact_indices=[2, 3],
             interaction_form='cp_inner_product', rank=8),
        dict(pure_indices=[1, 3, 5], interact_indices=[]),
        dict(pure_indices=[], interact_indices=[2, 4]),
        dict(pure_indices=[1, 2], interact_indices=[2], periodic=True),
    ]
    for index, config in enumerate(configs, 1):
        neuron = neuronseek_config_to_string(config)
        print(f'Neuron {index}: {neuron}', flush=True)
        mlp = MLPRegressor(neuron)
        mlp.fit(X, y)
        predictions = mlp.predict(X)
        assert predictions.shape == (100,)
        assert np.isfinite(predictions).all()
        assert np.isfinite(mlp.losses).all()
        print(f'Configuration {index}: {len(mlp.losses)} iterations completed', flush=True)
