import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tnlearn.rl_regressor import RLRegressor

# ==================== 定义目标方程 ====================
# 用户可在此修改方程（一维函数，x 为自变量）
def target_equation(x):
    return 2 * x**2 + 3 * x + 1

# 生成数据范围及样本量
n_samples = 500
noise_std = 0.1                     # 噪声标准差

X = np.random.uniform(0.5, 5, (500, 1))
y = 2 * X.flatten()**2 + 3 * X.flatten() + 1 + np.random.normal(0, 0.1, 500)

# ==================== 划分训练/测试集 ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==================== 训练 RL 回归器 ====================
rl_reg = RLRegressor(
    max_length=15,
    population_size=200,
    n_iter=800,
    risk_factor=0.05,         # 激进地选择 top 5% 表达式
    learning_rate=0.0005,
    hidden_size=128,
    random_state=42,          
    verbose=1,
    patience=100,
    optimize_constants=True,
)
rl_reg.fit(X_train, y_train)

# ==================== 评估与输出 ====================
y_pred = rl_reg.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"\nR² 分数 (测试集): {r2:.4f}")
print(f"发现的表达式: {rl_reg.get_expression()}")