"""
Example: Using RLRegressor to discover a symbolic expression and building an MLP with it.

Data is generated as: y = 4*x^2 + 3*x - 5 + sin(x) + noise.
RLRegressor learns to select a subset of polynomial and trigonometric basis functions.
The discovered expression is then used as a neuron formula in MLPRegressor.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from tnlearn.rl_regressor import RLRegressor
from tnlearn import MLPRegressor

# ============================================================================
# 1. Generate synthetic data (mixed polynomial and trigonometric)
# ============================================================================
np.random.seed(1)
n_samples = 500
X = np.random.uniform(-2, 2, (n_samples, 1))
y = 4 * X[:, 0] ** 2 + 3 * X[:, 0] - 5 + np.sin(X[:, 0]) + 0.1 * np.random.randn(n_samples)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# ============================================================================
# 2. Train RLRegressor to discover the underlying expression
# ============================================================================
rl = RLRegressor(
    random_state=42,
    max_episodes=200,          # Keep low for demonstration; increase for better results
    max_power=3,
    max_terms=3,
    use_trigonometric=True,
    val_split=0.2,
    verbose=True,
)
rl.fit(X_train, y_train)

expr = rl.get_neuron()
print(f"\nDiscovered expression: {expr}")

# ============================================================================
# 3. Build an MLP using the discovered neuron formula
# ============================================================================
mlp = MLPRegressor(
    neurons=expr,
    layers_list=[50, 30, 10],
    activation_funcs='sigmoid',
    max_iter=1000,
    batch_size=64,
    lr=0.001,
    random_state=1,
    visual=False,
)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"\nMLP with discovered neuron - Test R²: {r2:.4f}")