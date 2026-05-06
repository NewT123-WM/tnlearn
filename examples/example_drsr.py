"""
Example: Using DrSR (Deep Symbolic Regression) to discover mathematical equations from data.

This example demonstrates how to use LLMSymRegressor to automatically discover
the underlying equation from noisy observations.

The true data generating process is: y = 3*x0 + 0.5*sin(x1) + noise
DrSR should output an equation like: return params[0]*x0 + params[1]*np.sin(x1) + params[2]
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tnlearn import LLMSymRegressor

# ============================================================================
# 1. Generate synthetic data
# ============================================================================
np.random.seed(42)
n_samples = 200
X = np.random.randn(n_samples, 2)
y = 3.0 * X[:, 0] + 0.5 * np.sin(X[:, 1]) + np.random.randn(n_samples) * 0.1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ============================================================================
# 2. Configure LLM (supports multiple providers)
# ============================================================================
# Option A: DeepSeek (set environment variable DEEPSEEK_API_KEY)
llm_config = {
    'model': 'deepseek/deepseek-chat',
    'temperature': 0.6,
    'base_url': 'https://api.deepseek.com',   # optional
}
# 添加环境变量，例如：export DEEPSEEK_API_KEY=...实际key字符

# ============================================================================
# 3. Train DrSR model
# ============================================================================
reg = LLMSymRegressor(
    llm_config=llm_config,
    max_iterations=8,           # Number of evolutionary iterations
    samples_per_iteration=4,    # Number of candidates per iteration
    verbose=True                # Print progress
)

print("Starting DrSR training...")
reg.fit(X_train, y_train)

# ============================================================================
# 4. Results
# ============================================================================
print("\n" + "="*60)
print("Best discovered equation body:")
print(reg.best_equation_)
if hasattr(reg, 'best_params_') and reg.best_params_ is not None:
    print("\nOptimized parameters:")
    for i, p in enumerate(reg.best_params_):
        print(f"  params[{i}] = {p:.6f}")
print("="*60)

# ============================================================================
# 5. Evaluate on test set
# ============================================================================
y_pred = reg.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"\nTest R² score: {r2:.6f}")

# Simple baseline comparison (linear regression using only x0)
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train[:, 0:1], y_train)
y_pred_lr = lr.predict(X_test[:, 0:1])
r2_lr = r2_score(y_test, y_pred_lr)
print(f"Linear regression (x0 only) R²: {r2_lr:.6f}")
