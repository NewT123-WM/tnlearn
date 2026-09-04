"""
Reinforcement Learning-based Symbolic Regressors.

This module provides two RL-based symbolic regression implementations:
1. RLSymRegressor (default, base): Discovers InnerProduct-based expressions:
       f(x) = sum_i term_i(x)
   where term_i are either phi_k = InnerProduct(w, x**k) or
   psi_r = InnerProduct(w1,x)*...*InnerProduct(wr,x).
2. RLRegressor (legacy): Discovers vectorized (homogeneous) expressions:
       f(x) = sum_k c_k * sum_j phi_k(x_j)
   with basis functions (polynomial, trigonometric, etc.).

Copyright (c) 2026 Tieyun LI. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple
from sympy import sympify, simplify, expand
from .operator.inner_product import InnerProduct
import re


# =============================================================================
# Utility functions (shared)
# =============================================================================

def simplify_expr(expr_str: str) -> str:
    """
    Simplify a symbolic expression string that contains InnerProduct.

    Parameters
    ----------
    expr_str : str
        Expression string with InnerProduct calls.

    Returns
    -------
    str
        Simplified expression string.
    """
    try:
        sym_expr = sympify(expr_str, locals={'InnerProduct': InnerProduct})
        sym_expr = expand(sym_expr)
        sym_expr = simplify(sym_expr)
        return str(sym_expr)
    except Exception:
        return expr_str


def format_pretty(expr_str: str) -> str:
    """
    Replace InnerProduct(...) with <...> for pretty printing.

    Parameters
    ----------
    expr_str : str
        Expression string with InnerProduct calls.

    Returns
    -------
    str
        Expression with InnerProduct replaced by angle brackets.
    """
    pattern = r'InnerProduct\(([^()]*)\)'
    result = expr_str
    while True:
        new_result = re.sub(pattern, r'<\1>', result)
        if new_result == result:
            break
        result = new_result
    return result



class PolicyNetwork(nn.Module):
    """Policy network that outputs a probability distribution over the basis functions."""
    def __init__(self, n_funcs: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_funcs),
            nn.Softmax(dim=-1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)



# =============================================================================
# RLSymRegressor (Base mode, default)
# =============================================================================

class RLSymRegressor:
    """
    Reinforcement Learning symbolic regressor that discovers expressions composed of
    inner-product terms (base mode).

    For base mode (default):
        f(x) = sum_i term_i(x)
    where each term_i is either:
        - phi_k   = InnerProduct(w, x**k)          (k >= 1)
        - psi_r   = InnerProduct(w1, x) * InnerProduct(w2, x) * ... * InnerProduct(wr, x)   (r >= 2)

    Each term type can be selected multiple times; each instance gets distinct weights.

    For legacy mode:
        Delegates to RLRegressor, which discovers vectorized (homogeneous) expressions.

    Parameters
    ----------
    mode : str, default='base'
        'base'  : use InnerProduct-based expression discovery (this class).
        'legacy': use the older RLRegressor (vectorized, homogeneous) implementation.
    max_power : int, default=5
        Maximum exponent for polynomial terms (k up to max_power).
    max_terms_psi : int, default=3
        Maximum number of InnerProduct factors in a psi_r term (r up to this value).
    alpha : float, default=0.1
        Regularisation strength for Ridge regression (used for evaluation only).
    random_state : int, default=42
        Random seed for reproducibility.
    max_episodes : int, default=100
        Number of training episodes.
    val_split : float, default=0.2
        Fraction of training data used as validation for reward computation.
    lr_rl : float, default=1e-3
        Learning rate for the policy network.
    gamma : float, default=0.99
        Discount factor for reward calculation.
    hidden_dim : int, default=64
        Number of neurons in the policy network's hidden layers.
    max_terms_total : int, default=4
        Maximum number of selected terms in the expression (i.e., total number of phi_k and psi_r).
    standardize : bool, default=True
        If True, standardize the features (mean=0, std=1) internally to avoid numerical overflow.
    output_mode : str, default='symbolic'
        Output mode: 'symbolic' (keep w1, w2 as symbols) or 'numeric' (replace with actual coefficients).
    verbose : bool, default=True
        If True, print progress updates during training.

    Additional legacy-only parameters (only used when mode='legacy'):
    basis_mode : str, default='trigonometric'
        Basis function set for the legacy regressor: 'polynomial', 'trigonometric', or 'all'.
    """

    def __init__(
        self,
        mode: str = 'base',
        max_power: int = 5,
        max_terms_psi: int = 3,
        alpha: float = 0.1,
        random_state: int = 42,
        max_episodes: int = 100,
        val_split: float = 0.2,
        lr_rl: float = 1e-3,
        gamma: float = 0.99,
        hidden_dim: int = 64,
        max_terms_total: int = 4,
        standardize: bool = True,
        output_mode: str = 'symbolic',
        verbose: bool = True,
        # legacy-specific
        basis_mode: str = 'trigonometric',
    ):
        self.mode = mode.lower()
        if self.mode not in ('base', 'legacy'):
            raise ValueError("mode must be 'base' or 'legacy'")

        self.max_power = max_power
        self.max_terms_psi = max_terms_psi
        self.alpha = alpha
        self.random_state = random_state
        self.max_episodes = max_episodes
        self.val_split = val_split
        self.lr_rl = lr_rl
        self.gamma = gamma
        self.hidden_dim = hidden_dim
        self.max_terms_total = max_terms_total
        self.standardize = standardize
        self.output_mode = output_mode.lower()
        if self.output_mode not in ('symbolic', 'numeric'):
            raise ValueError("output_mode must be 'symbolic' or 'numeric'")
        self.verbose = verbose
        self.basis_mode = basis_mode.lower()

        np.random.seed(random_state)
        torch.manual_seed(random_state)

        # For base mode, initialize internals; for legacy, we will instantiate RLRegressor in fit.
        if self.mode == 'base':
            self.scaler_ = None
            self.candidate_terms = []

            # 1. phi_k = InnerProduct(w, x**k), k = 1..max_power
            for k in range(1, self.max_power + 1):
                self.candidate_terms.append({
                    'type': 'phi',
                    'k': k,
                    'func_str': f'torch.sum(x**{k}, axis=1)',
                    'display': f'phi_{k}'
                })

            # 2. psi_r = InnerProduct(w1, x) * InnerProduct(w2, x) * ... (r times)
            for r in range(2, self.max_terms_psi + 1):
                self.candidate_terms.append({
                    'type': 'psi',
                    'r': r,
                    'func_str': f'(x.sum(axis=1))**{r}',
                    'display': f'psi_{r}'
                })

            self.n_terms = len(self.candidate_terms)

            self.policy = PolicyNetwork(self.n_terms, hidden_dim)
            self.optimizer = optim.Adam(self.policy.parameters(), lr=lr_rl)

            self.saved_log_probs = []
            self.rewards = []
            self.best_expr = None
            self.best_score = -float('inf')
            self.best_coeffs = None
            self.best_intercept = None
            self.best_selected_info = None
            self.neuron = None

        else:  # legacy
            # We store parameters for lazy instantiation in fit
            self._legacy_params = {
                'basis_mode': basis_mode,
                'max_terms': max_terms_total,
                'max_power': max_power,
                'alpha': alpha,
                'random_state': random_state,
                'max_episodes': max_episodes,
                'val_split': val_split,
                'lr_rl': lr_rl,
                'gamma': gamma,
                'hidden_dim': hidden_dim,
                'verbose': verbose,
            }
            self.best_expr = None
            self.best_score = -float('inf')
            self.neuron = None

    # ---------- Base mode methods ----------
    @staticmethod
    def _eval_func(func_str: str, x_tensor: torch.Tensor) -> np.ndarray:
        try:
            result = eval(func_str, {'x': x_tensor, 'torch': torch})
            if isinstance(result, torch.Tensor):
                return result.detach().numpy().flatten()
            else:
                return np.full(len(x_tensor), float(result), dtype=np.float64)
        except Exception:
            return np.full(len(x_tensor), np.nan)

    def _evaluate_selection(
        self,
        selected_indices: List[int],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[float, np.ndarray, float, List[dict]]:
        if not selected_indices:
            return -1e6, np.array([]), 0.0, []

        n_train, d = X_train.shape
        n_val = X_val.shape[0]
        K = len(selected_indices)

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)

        A_train = np.zeros((n_train, K))
        A_val = np.zeros((n_val, K))
        selected_info = []

        for k, idx in enumerate(selected_indices):
            term = self.candidate_terms[idx]
            func_str = term['func_str']
            try:
                phi_train = self._eval_func(func_str, X_train_t)
                phi_val = self._eval_func(func_str, X_val_t)
            except Exception:
                return -1e6, np.array([]), 0.0, []

            if np.any(np.isnan(phi_train)) or np.any(np.isnan(phi_val)):
                return -1e6, np.array([]), 0.0, []

            A_train[:, k] = phi_train
            A_val[:, k] = phi_val
            selected_info.append(term)

        ridge = Ridge(alpha=self.alpha, fit_intercept=True)
        ridge.fit(A_train, y_train)
        coeffs = ridge.coef_
        intercept = ridge.intercept_

        y_pred_val = ridge.predict(A_val)
        r2 = r2_score(y_val, y_pred_val)

        return r2, coeffs, intercept, selected_info

    def _build_expr_symbolic(self, selected_info: List[dict]) -> str:
        terms = []
        weight_counter = 1
        for info in selected_info:
            if info['type'] == 'phi':
                w_sym = f"w{weight_counter}"
                weight_counter += 1
                term_str = f"InnerProduct({w_sym}, x**{info['k']})"
            else:  # psi
                r = info['r']
                inner_prods = []
                for _ in range(r):
                    w_sym = f"w{weight_counter}"
                    weight_counter += 1
                    inner_prods.append(f"InnerProduct({w_sym}, x)")
                term_str = " * ".join(inner_prods)
            terms.append(term_str)
        if not terms:
            return "0"
        expr = " + ".join(terms)
        expr = expr.replace("+ -", "- ")
        return expr

    def _build_expr_numeric(self, selected_info: List[dict], coeffs: np.ndarray, intercept: float) -> str:
        terms = []
        if abs(intercept) > 1e-8:
            terms.append(f"{intercept:.6f}")
        for info, c in zip(selected_info, coeffs):
            if abs(c) < 1e-8:
                continue
            if info['type'] == 'phi':
                term_str = f"InnerProduct({c:.6f}, x**{info['k']})"
            else:
                r = info['r']
                inner_prods = " * ".join(["InnerProduct(1, x)"] * r)
                term_str = f"{c:.6f} * {inner_prods}"
            terms.append(term_str)
        if not terms:
            return "0"
        expr = " + ".join(terms)
        expr = expr.replace("+ -", "- ")
        return expr

    def _select_action(self) -> Tuple[List[int], torch.Tensor]:
        state = torch.tensor([0.0])
        probs = self.policy(state).squeeze(0)
        sampled = []
        log_prob = 0.0
        for _ in range(self.max_terms_total):
            idx = torch.multinomial(probs, 1).item()
            sampled.append(idx)
            log_prob += torch.log(probs[idx] + 1e-10)
        return sampled, log_prob

    # ---------- Public fit method ----------
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the RL agent to discover the best symbolic expression.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data (can be multi-dimensional).
        y : np.ndarray, shape (n_samples,)
            Target values.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # ---------- Legacy mode ----------
        if self.mode == 'legacy':
            # Instantiate RLRegressor with stored parameters
            legacy = RLRegressor(**self._legacy_params)
            legacy.fit(X, y)
            self.best_expr = legacy.best_expr
            self.best_score = legacy.best_score
            self.neuron = legacy.neuron
            return

        # ---------- Base mode ----------
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_split, random_state=self.random_state
        )

        if self.standardize:
            self.scaler_ = StandardScaler()
            X_train = self.scaler_.fit_transform(X_train)
            X_val = self.scaler_.transform(X_val)
            if self.verbose:
                print("  → Standardized features (mean=0, std=1)")

        for episode in range(self.max_episodes):
            selected, log_prob = self._select_action()
            self.saved_log_probs.append(log_prob)

            reward, coeffs, intercept, selected_info = self._evaluate_selection(
                selected, X_train, y_train, X_val, y_val
            )
            self.rewards.append(reward)

            if reward > self.best_score:
                self.best_score = reward
                if selected_info:
                    self.best_coeffs = coeffs
                    self.best_intercept = intercept
                    self.best_selected_info = selected_info

                    if self.output_mode == 'symbolic':
                        raw_expr = self._build_expr_symbolic(selected_info)
                    else:
                        raw_expr = self._build_expr_numeric(selected_info, coeffs, intercept)

                    simplified = simplify_expr(raw_expr)
                    self.best_expr = simplified
                else:
                    self.best_expr = "0"
                if self.verbose:
                    print(f"\rEpisode {episode+1}: found better expression '{self.best_expr}' (val R²={reward:.4f})", end="")

            # REINFORCE update
            R = 0
            returns = []
            for r in reversed(self.rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns)
            if len(returns) > 1 and returns.std() > 1e-8:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            elif len(returns) > 1:
                returns = returns - returns.mean()

            policy_loss = []
            for logp, ret in zip(self.saved_log_probs, returns):
                policy_loss.append(-logp * ret)
            self.optimizer.zero_grad()
            torch.stack(policy_loss).sum().backward()
            self.optimizer.step()

            self.saved_log_probs = []
            self.rewards = []

            if self.verbose and (episode + 1) % 10 == 0 and reward <= self.best_score:
                print(f"\rEpisode {episode+1}/{self.max_episodes}, current best R²={self.best_score:.4f}   ", end="")

        if self.verbose:
            print()
            print(f"RL search completed. Best expression: {self.best_expr}, validation R² = {self.best_score:.4f}")

        # Store pretty version in neuron attribute
        if self.best_expr:
            self.neuron = format_pretty(self.best_expr)
        else:
            self.neuron = "0"

    def get_neuron(self) -> Optional[str]:
        """Return the best discovered neuron formula in pretty format."""
        return self.neuron

    def get_raw_expr(self) -> Optional[str]:
        """Return the raw (non-pretty) simplified expression."""
        return self.best_expr


# =============================================================================
# RLRegressor (Legacy mode)
# =============================================================================

class RLRegressor:
    """
    Reinforcement Learning symbolic regressor (legacy, vectorized/homogeneous).

    Discovers expressions of the form:
        f(x) = sum_k c_k * sum_j phi_k(x_j)

    The agent selects basis functions (phi_k), and Ridge regression fits the global
    coefficients c_k. The reward is validation R².

    Parameters
    ----------
    basis_mode : str, default='trigonometric'
        Basis function set: 'polynomial', 'trigonometric', or 'all'.
    max_terms : int, default=3
        Maximum number of basis functions selected per expression.
    max_power : int, default=5
        Maximum exponent for polynomial terms.
    alpha : float, default=0.1
        Regularisation strength for Ridge regression.
    random_state : int, default=42
        Random seed.
    max_episodes : int, default=100
        Number of training episodes.
    val_split : float, default=0.2
        Fraction of training data used as validation.
    lr_rl : float, default=1e-3
        Learning rate for policy network.
    gamma : float, default=0.99
        Discount factor.
    hidden_dim : int, default=64
        Number of neurons in policy network hidden layers.
    verbose : bool, default=True
        Print progress.
    """
    def __init__(
        self,
        basis_mode: str = 'trigonometric',
        max_terms: int = 3,
        max_power: int = 5,
        alpha: float = 0.1,
        random_state: int = 42,
        max_episodes: int = 100,
        val_split: float = 0.2,
        lr_rl: float = 1e-3,
        gamma: float = 0.99,
        hidden_dim: int = 64,
        verbose: bool = True,
    ):
        self.basis_mode = basis_mode.lower()
        if self.basis_mode not in ['polynomial', 'trigonometric', 'all']:
            raise ValueError("basis_mode must be 'polynomial', 'trigonometric', or 'all'")
        self.max_terms = max_terms
        self.max_power = max_power
        self.alpha = alpha
        self.random_state = random_state
        self.max_episodes = max_episodes
        self.val_split = val_split
        self.lr_rl = lr_rl
        self.gamma = gamma
        self.verbose = verbose

        np.random.seed(random_state)
        torch.manual_seed(random_state)

        # Build candidate basis functions
        self.candidate_funcs = []
        self.candidate_funcs.append('torch.ones_like(x)')  # constant
        self.candidate_funcs.append('x')                   # linear
        for p in range(2, self.max_power + 1):
            self.candidate_funcs.append(f'x**{p}')

        if self.basis_mode in ['trigonometric', 'all']:
            self.candidate_funcs.append('torch.sin(x)')
            self.candidate_funcs.append('torch.cos(x)')
        if self.basis_mode == 'all':
            self.candidate_funcs.append('torch.exp(x)')
            self.candidate_funcs.append('torch.log(torch.abs(x) + 1e-8)')

        self.n_funcs = len(self.candidate_funcs)
        self.policy = PolicyNetwork(self.n_funcs, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr_rl)

        self.saved_log_probs = []
        self.rewards = []
        self.best_expr = None
        self.best_score = -float('inf')
        self.neuron = None

    @staticmethod
    def _eval_func(func_str: str, x_tensor: torch.Tensor) -> np.ndarray:
        try:
            result = eval(func_str, globals(), {'x': x_tensor, 'torch': torch})
            if isinstance(result, torch.Tensor):
                return result.detach().numpy().flatten()
            else:
                return np.full(len(x_tensor), float(result), dtype=np.float64)
        except Exception:
            return np.full(len(x_tensor), np.nan)

    def _evaluate_selection(
        self,
        selected_indices: List[int],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[float, Optional[np.ndarray], float, List[str]]:
        if not selected_indices:
            return -1e6, None, 0.0, []

        n_train, d = X_train.shape
        n_val = X_val.shape[0]
        K = len(selected_indices)

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)

        A_train = np.zeros((n_train, K))
        A_val = np.zeros((n_val, K))
        selected_funcs = []

        for k, idx in enumerate(selected_indices):
            func_str = self.candidate_funcs[idx]
            phi_train_sum = np.zeros(n_train)
            for i in range(d):
                f_i = self._eval_func(func_str, X_train_t[:, i])
                if np.any(np.isnan(f_i)):
                    return -1e6, None, 0.0, []
                phi_train_sum += f_i
            A_train[:, k] = phi_train_sum

            phi_val_sum = np.zeros(n_val)
            for i in range(d):
                f_i = self._eval_func(func_str, X_val_t[:, i])
                if np.any(np.isnan(f_i)):
                    return -1e6, None, 0.0, []
                phi_val_sum += f_i
            A_val[:, k] = phi_val_sum

            selected_funcs.append(func_str)

        ridge = Ridge(alpha=self.alpha, fit_intercept=True)
        ridge.fit(A_train, y_train)
        coeffs = ridge.coef_
        intercept = ridge.intercept_

        y_pred_val = ridge.predict(A_val)
        r2 = r2_score(y_val, y_pred_val)

        return r2, coeffs, intercept, selected_funcs

    def _build_expr(self, funcs: List[str], coeffs: np.ndarray, intercept: float) -> str:
        terms = []
        if abs(intercept) > 1e-8:
            if abs(intercept - round(intercept)) < 1e-8:
                terms.append(str(int(round(intercept))))
            else:
                terms.append(f"{intercept:.4f}")

        for func, c in zip(funcs, coeffs):
            if abs(c) < 1e-8:
                continue
            c_str = str(int(round(c))) if abs(c - round(c)) < 1e-8 else f"{c:.4f}"
            if func == "torch.ones_like(x)":
                terms.append(c_str)
            else:
                if any(op in func for op in ['+', '-', '*', '/', '**', 'sin', 'cos', 'exp', 'log']):
                    func = f"({func})"
                terms.append(f"{c_str}@{func}")

        if not terms:
            return "0"
        expr = " + ".join(terms)
        expr = expr.replace("+ -", "- ")
        return expr

    def _select_action(self) -> Tuple[List[int], torch.Tensor]:
        state = torch.tensor([0.0])
        probs = self.policy(state).squeeze(0)
        sampled = []
        log_prob = 0.0
        for _ in range(self.max_terms):
            idx = torch.multinomial(probs, 1).item()
            sampled.append(idx)
            log_prob += torch.log(probs[idx] + 1e-10)
        selected = list(dict.fromkeys(sampled))  # remove duplicates
        return selected, log_prob

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_split, random_state=self.random_state
        )

        for episode in range(self.max_episodes):
            selected, log_prob = self._select_action()
            self.saved_log_probs.append(log_prob)

            reward, coeffs, intercept, funcs = self._evaluate_selection(
                selected, X_train, y_train, X_val, y_val
            )
            self.rewards.append(reward)

            if reward > self.best_score:
                self.best_score = reward
                if coeffs is not None:
                    self.best_expr = self._build_expr(funcs, coeffs, intercept)
                else:
                    self.best_expr = "0"
                if self.verbose:
                    print(f"\rEpisode {episode+1}: found better expression '{self.best_expr}' (val R²={reward:.4f})", end="")

            # REINFORCE update
            R = 0
            returns = []
            for r in reversed(self.rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns)
            if len(returns) > 1 and returns.std() > 1e-8:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            elif len(returns) > 1:
                returns = returns - returns.mean()

            policy_loss = []
            for logp, ret in zip(self.saved_log_probs, returns):
                policy_loss.append(-logp * ret)
            self.optimizer.zero_grad()
            torch.stack(policy_loss).sum().backward()
            self.optimizer.step()

            self.saved_log_probs = []
            self.rewards = []

            if self.verbose and (episode + 1) % 10 == 0 and reward <= self.best_score:
                print(f"\rEpisode {episode+1}/{self.max_episodes}, current best R²={self.best_score:.4f}   ", end="")

        if self.verbose:
            print()
            print(f"RL search completed. Best expression: {self.best_expr}, validation R² = {self.best_score:.4f}")

        self.neuron = self.best_expr

    def get_neuron(self) -> Optional[str]:
        return self.neuron

