import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import minimize
import sympy as sp
import re
from typing import Optional, List, Tuple

# ------------------------------
# 安全数学函数（支持标量和数组）
# ------------------------------
def safe_log(x):
    return np.log(np.abs(x) + 1e-8) * (x > 0).astype(float)

def safe_sqrt(x):
    return np.sqrt(np.abs(x)) * (x >= 0).astype(float)

def safe_div(x, y):
    return x / (y + 1e-8)

def safe_exp(x):
    return np.exp(np.clip(x, -50, 50))

class RLRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        max_length: int = 20,
        population_size: int = 50,
        n_iter: int = 200,
        learning_rate: float = 0.001,
        hidden_size: int = 64,
        risk_factor: float = 0.1,
        device: Optional[str] = None,
        random_state: Optional[int] = None,
        verbose: int = 1,
        patience: int = 20,
        optimize_constants: bool = True,
    ):
        self.max_length = max_length
        self.population_size = population_size
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.risk_factor = risk_factor
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self.patience = patience
        self.optimize_constants = optimize_constants

        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)

        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # 定义语法
        self.tokens = [
            'x', 'const',
            'add', 'sub', 'mul', 'div',
            'square', 'cube',
            'log', 'sqrt', 'exp', 'sin', 'cos', 'tanh'
        ]
        self.vocab_size = len(self.tokens)
        self.token_to_idx = {t: i for i, t in enumerate(self.tokens)}
        self.idx_to_token = {i: t for i, t in enumerate(self.tokens)}

        self.arity = {
            'x': 0, 'const': 0,
            'square': 1, 'cube': 1, 'log': 1, 'sqrt': 1, 'exp': 1, 'sin': 1, 'cos': 1, 'tanh': 1,
            'add': 2, 'sub': 2, 'mul': 2, 'div': 2,
        }

    def _build_policy_network(self):
        self.embedding = nn.Embedding(self.vocab_size, 32).to(self.device)
        self.lstm = nn.LSTM(32, self.hidden_size, batch_first=True).to(self.device)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size).to(self.device)
        self.optimizer = optim.Adam(
            list(self.embedding.parameters()) +
            list(self.lstm.parameters()) +
            list(self.fc.parameters()),
            lr=self.learning_rate
        )

    def _is_valid_prefix_seq(self, tokens: List[str]) -> bool:
        need = 1
        for t in tokens:
            ar = self.arity[t]
            if ar == 0:
                need -= 1
            else:
                need += ar - 1
            if need < 0:
                return False
        return need == 0

    def _prefix_to_infix(self, tokens: List[str]) -> str:
        stack = []
        for t in reversed(tokens):
            ar = self.arity[t]
            if ar == 0:
                stack.append(t)
            else:
                args = [stack.pop() for _ in range(ar)]
                if ar == 1:
                    if t == 'square':
                        subtree = f"({args[0]}**2)"
                    elif t == 'cube':
                        subtree = f"({args[0]}**3)"
                    else:
                        subtree = f"{t}({args[0]})"
                else:
                    op = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}[t]
                    subtree = f"({args[0]} {op} {args[1]})"
                stack.append(subtree)
        return stack[0] if stack else 'x'

    def _generate_expression(self) -> Tuple[str, List[torch.Tensor]]:
        tokens = []
        log_probs = []
        hidden = None
        need = 1
        input_token = torch.tensor([[self.token_to_idx['x']]], device=self.device)

        for step in range(self.max_length):
            emb = self.embedding(input_token)
            out, hidden = self.lstm(emb, hidden)
            logits = self.fc(out[:, -1, :])

            mask = torch.ones(self.vocab_size, dtype=torch.bool, device=self.device)
            for i, token in enumerate(self.tokens):
                ar = self.arity[token]
                if need == 0 and ar > 0:
                    mask[i] = False
            logits = logits.masked_fill(~mask, -1e9)

            logp = torch.log_softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(logits=logits)
            token_idx = dist.sample()
            token = self.idx_to_token[token_idx.item()]
            tokens.append(token)
            log_probs.append(logp[0, token_idx])

            ar = self.arity[token]
            if ar == 0:
                need -= 1
            else:
                need += ar - 1

            input_token = token_idx.view(1, 1)
            if need == 0:
                break

        if need != 0:
            return 'x', [torch.tensor(0.0, device=self.device)]
        expr = self._prefix_to_infix(tokens)
        return expr, log_probs

    def _evaluate_expression(self, expr: str, X: np.ndarray) -> np.ndarray:
        """安全评估表达式，保证不返回 NaN 或 Inf，无效表达式返回很大的常数预测"""
        # 替换 const 占位符
        const_symbols = []
        const_values = []

        def replace_const(match):
            name = f'c{len(const_symbols)}'
            const_symbols.append(name)
            const_values.append(1.0)
            return name

        expr_const = re.sub(r'\bconst\b', replace_const, expr)
        x_sym = sp.Symbol('x')
        sym_dict = {f'c{i}': sp.Symbol(f'c{i}') for i in range(len(const_symbols))}
        sym_dict['x'] = x_sym

        try:
            expr_sym = sp.sympify(expr_const, locals=sym_dict)
            if expr_sym.has(sp.zoo, sp.oo, sp.nan):
                return np.full(X.shape[0], 1e10)
        except:
            return np.full(X.shape[0], 1e10)

        # 安全函数模块
        safe_module = {
            'log': safe_log,
            'sqrt': safe_sqrt,
            'exp': safe_exp,
            'sin': np.sin,
            'cos': np.cos,
            'tanh': np.tanh,
            'add': lambda a, b: a + b,
            'sub': lambda a, b: a - b,
            'mul': lambda a, b: a * b,
            'div': safe_div,
            'square': lambda x: x**2,
            'cube': lambda x: x**3
        }

        # 常数优化（仅训练阶段）
        if self.optimize_constants and const_symbols and hasattr(self, 'y_fit'):
            def objective(cvals):
                subs = {sym_dict[f'c{i}']: cvals[i] for i in range(len(cvals))}
                try:
                    expr_eval = expr_sym.subs(subs)
                    if expr_eval.has(sp.zoo, sp.oo, sp.nan):
                        return 1e10
                    f_lambda = sp.lambdify(x_sym, expr_eval, modules=[safe_module, 'numpy'])
                    y_pred = f_lambda(X.flatten())
                    if np.iscomplexobj(y_pred):
                        y_pred = np.real(y_pred)
                    y_pred = np.nan_to_num(y_pred, nan=1e10, posinf=1e10, neginf=-1e10)
                    if np.isscalar(y_pred):
                        y_pred = np.full(X.shape[0], y_pred)
                    return np.mean((y_pred - self.y_fit) ** 2)
                except:
                    return 1e10

            best_mse = np.inf
            best_c = const_values
            for _ in range(3):
                init = np.random.uniform(-5, 5, len(const_symbols))
                res = minimize(objective, init, method='L-BFGS-B', options={'maxiter': 50})
                if res.fun < best_mse:
                    best_mse = res.fun
                    best_c = res.x
            const_values = best_c

        # 最终预测
        subs = {sym_dict[f'c{i}']: const_values[i] for i in range(len(const_symbols))}
        expr_final = expr_sym.subs(subs)
        if expr_final.has(sp.zoo, sp.oo, sp.nan):
            return np.full(X.shape[0], 1e10)
        try:
            f_final = sp.lambdify(x_sym, expr_final, modules=[safe_module, 'numpy'])
            y_pred = f_final(X.flatten())
            if np.iscomplexobj(y_pred):
                y_pred = np.real(y_pred)
            y_pred = np.nan_to_num(y_pred, nan=1e10, posinf=1e10, neginf=-1e10)
            if np.isscalar(y_pred):
                y_pred = np.full(X.shape[0], y_pred)
            return y_pred.astype(np.float64)
        except:
            return np.full(X.shape[0], 1e10)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != 1:
            raise ValueError("RLRegressor currently only supports 1 input feature (x).")
        self.n_features_in_ = X.shape[1]
        self.X_fit_ = X
        self.y_fit_ = y

        self._build_policy_network()
        best_expr = None
        best_r2 = -np.inf
        best_mse = np.inf
        no_improve = 0

        for iteration in range(self.n_iter):
            exprs = []
            log_probs_list = []
            for _ in range(self.population_size):
                expr, lps = self._generate_expression()
                exprs.append(expr)
                log_probs_list.append(lps)

            rewards = []
            mses = []
            r2s = []
            for expr in exprs:
                y_pred = self._evaluate_expression(expr, X)
                if y_pred.ndim > 1:
                    y_pred = y_pred.flatten()
                # 确保无 NaN
                if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                    mse = 1e10
                    r2 = -1e10
                else:
                    mse = mean_squared_error(y, y_pred)
                    r2 = r2_score(y, y_pred)
                reward = -mse
                rewards.append(reward)
                mses.append(mse)
                r2s.append(r2)

            cur_best_idx = np.argmax(r2s)
            if r2s[cur_best_idx] > best_r2:
                best_r2 = r2s[cur_best_idx]
                best_mse = mses[cur_best_idx]
                best_expr = exprs[cur_best_idx]
                no_improve = 0
            else:
                no_improve += 1

            if self.verbose and iteration % 20 == 0:
                print(f"Iter {iteration:3d} | best R²={best_r2:.4f} | expr: {best_expr[:60]}")

            if no_improve >= self.patience and iteration > 50:
                if self.verbose:
                    print(f"Early stopping at iteration {iteration}")
                break

            # REINFORCE 更新
            k = max(1, int(self.population_size * self.risk_factor))
            top_indices = np.argsort(rewards)[-k:]

            if k > 0:
                self.optimizer.zero_grad()
                total_loss = torch.tensor(0.0, device=self.device)
                baseline = np.mean([rewards[i] for i in top_indices])
                for idx in top_indices:
                    log_prob_sum = sum(log_probs_list[idx])
                    advantage = rewards[idx] - baseline
                    loss = -log_prob_sum * advantage
                    total_loss = total_loss + loss
                if total_loss != 0:
                    total_loss.backward()
                    self.optimizer.step()

        self.best_expression_ = best_expr
        self.best_score_ = best_r2
        self.best_loss_ = best_mse
        return self

    def predict(self, X):
        if not hasattr(self, 'best_expression_'):
            raise RuntimeError("Model not fitted yet.")
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        original_opt = self.optimize_constants
        self.optimize_constants = False
        y_pred = self._evaluate_expression(self.best_expression_, X)
        self.optimize_constants = original_opt
        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()
        return y_pred

    def get_expression(self):
        return self.best_expression_