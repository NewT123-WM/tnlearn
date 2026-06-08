"""
Example: Using LLMSymRegressor to discover a polynomial neuron and building an MLP with it.
Data: y = 3 * x^2 + 2 * x + noise
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tnlearn import LLMSymRegressor, MLPRegressor

# ============================================================================
# 1. Generate synthetic data (univariate, polynomial with noise)
# ============================================================================
np.random.seed(42)
n_samples = 300
X = np.random.uniform(-3, 3, n_samples).reshape(-1, 1)
y = 3.0 * X[:, 0]**2 + 2.0 * X[:, 0] + 0.2 * np.random.randn(n_samples)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ============================================================================
# 2. Train LLMSymRegressor to discover the polynomial formula
# ============================================================================
llm_config = {
    'model': 'deepseek/deepseek-chat',   
    'temperature': 0.6,
}
# Ensure DEEPSEEK_API_KEY is set (export DEEPSEEK_API_KEY=YOUR_API_KEY)

reg = LLMSymRegressor(
    llm_config=llm_config,
    max_iterations=3,
    samples_per_iteration=4,
    verbose=1,          # 0=quiet, 1=basic progress, 2=debug
    extra_prompt='Polynomial is preferred'
)
print("Training LLMSymRegressor to discover the underlying formula...")
reg.fit(X_train, y_train)

# Display discovered equation
print("\n" + "="*60)
print("Discovered equation body (with param placeholders):")
print(reg.best_equation_)
print("\nNeuron formula (with optimized coefficients):")
print(reg.get_neuron_formula())
print("="*60)

# ============================================================================
# 3. Build MLPRegressor using the discovered neuron formula
# ============================================================================
mlp_custom = MLPRegressor(
    neurons=reg.get_neuron_formula(),   # Use the discovered neuron
    layers_list=[20, 10],               # Two hidden layers
    activation_funcs='relu',
    max_iter=300,
    batch_size=64,
    lr=0.001,
)
print("\nTraining MLP with the discovered neuron...")
mlp_custom.fit(X_train, y_train)
y_pred_custom = mlp_custom.predict(X_test)
r2_custom = r2_score(y_test, y_pred_custom)

# ============================================================================
# 4. Results
# ============================================================================
print("\n" + "="*60)
print(f"MLP with discovered neuron R² = {r2_custom:.6f}")
print("="*60)