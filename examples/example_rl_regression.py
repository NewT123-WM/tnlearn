import numpy as np
from tnlearn.rl_regressor import RLSymRegressor

np.random.seed(42)
n_samples, n_features = 300, 5
X = np.random.randn(n_samples, n_features)
sum_x = np.sum(X, axis=1)
y = np.sum(X**2, axis=1) + 0.5 * sum_x**2 + 2 * X[:, 0] + 0.1 * np.random.randn(n_samples)

reg = RLSymRegressor(
    max_power=4,
    max_terms_psi=3,
    alpha=0.1,
    random_state=42,
    max_episodes=50,
    val_split=0.2,
    lr_rl=1e-3,
    gamma=0.99,
    hidden_dim=64,
    max_terms_total=4,
    verbose=True,
)

reg.fit(X, y)
print("\n最佳表达式 (pretty):", reg.neuron)
print("验证 R²:", reg.best_score)