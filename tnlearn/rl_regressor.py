# Copyright 2026 Tieyun LI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Reinforcement Learning‑based Symbolic Regressor (Vectorized / Homogeneous version).

This module implements a policy gradient (REINFORCE) agent that selects a subset of
basis functions (polynomials and optionally trigonometric/other terms) to form a
symbolic expression. The expression is applied in a vectorized (homogeneous) manner:
    F(x) = sum_k c_k * sum_j phi_k(x_j)
Coefficients are fitted via Ridge regression on the training set, and the validation
R² is used as the reward. The discovered expression can be used as a neuron formula
in MLPRegressor.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from typing import List, Optional, Tuple, Union
import warnings

warnings.filterwarnings("ignore")


class PolicyNetwork(nn.Module):
    """
    Policy network that outputs a probability distribution over the basis functions.
    """

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


class RLRegressor:
    """
    Reinforcement Learning symbolic regressor that discovers a vectorized (homogeneous)
    expression of the form:
        f(x) = sum_k c_k * sum_j phi_k(x_j)

    The agent selects basis functions (phi_k), and Ridge regression fits the global
    coefficients c_k. The search is done on the training set, and the reward is the
    validation R². The discovered expression is compatible with VecSymRegressor format.

    Parameters
    ----------
    basis_mode : str, default='trigonometric'
        Determines the set of basis functions available:
        - 'polynomial' : only polynomial terms: constant, x, x**2, ..., x**max_power
        - 'trigonometric' : polynomial + sin(x), cos(x)
        - 'all' : polynomial + sin(x), cos(x), exp(x), log(|x|)
    max_terms : int, default=3
        Maximum number of basis functions selected per expression.
    max_power : int, default=5
        Maximum exponent for polynomial terms (x**2 .. x**max_power). x and constant are always included.
    alpha : float, default=0.1
        Regularisation strength for Ridge regression.
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
    verbose : bool, default=True
        If True, print progress updates during training.
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
            raise ValueError("basis_mode must be one of 'polynomial', 'trigonometric', or 'all'")

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

        # Build candidate basis functions based on mode and max_power
        self.candidate_funcs = []
        # Constant term (torch.ones_like(x))
        self.candidate_funcs.append('torch.ones_like(x)')
        # Linear term
        self.candidate_funcs.append('x')
        # Higher powers
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
        """
        Safely evaluate a basis function on a 1D torch tensor.
        Returns a numpy array of shape (n_samples,).
        """
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
        """
        Evaluate a set of selected basis functions in a vectorized (homogeneous) way.

        For each selected basis function phi_k, we compute:
            phi_k(X)  ->  sum over features (axis=1)
        Then we stack these columns and fit a Ridge regression to obtain global coefficients.
        The reward is the validation R².

        Returns:
            r2 : validation R²
            coeffs : fitted coefficients (length K)
            intercept : fitted intercept
            funcs : list of selected basis function strings (order matches coeffs)
        """
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
        """
        Build a human‑readable expression string in the format:
            intercept + c1@phi1 + c2@phi2 + ...
        where phi_k are the basis function strings.
        """
        terms = []
        # Add constant term if intercept is non‑zero
        if abs(intercept) > 1e-8:
            if abs(intercept - round(intercept)) < 1e-8:
                terms.append(str(int(round(intercept))))
            else:
                terms.append(f"{intercept:.4f}")

        for func, c in zip(funcs, coeffs):
            if abs(c) < 1e-8:
                continue
            c_str = str(int(round(c))) if abs(c - round(c)) < 1e-8 else f"{c:.4f}"
            # For constant function, we treat it as a separate term, but we already handled intercept.
            if func == "torch.ones_like(x)":
                # Skip constant term because intercept already accounts for it.
                # However, if intercept is zero and constant term has non‑zero coefficient,
                # we should add it. We'll just add it as a separate term.
                terms.append(c_str)
            else:
                # Wrap composite expressions in parentheses
                if any(op in func for op in ['+', '-', '*', '/', '**', 'sin', 'cos', 'exp', 'log']):
                    func = f"({func})"
                terms.append(f"{c_str}@{func}")

        if not terms:
            return "0"
        expr = " + ".join(terms)
        expr = expr.replace("+ -", "- ")
        return expr

    def _select_action(self) -> Tuple[List[int], torch.Tensor]:
        """Sample up to max_terms basis functions (with replacement) and remove duplicates."""
        state = torch.tensor([0.0])
        probs = self.policy(state).squeeze(0)  # (n_funcs,)
        sampled = []
        log_prob = 0.0
        for _ in range(self.max_terms):
            idx = torch.multinomial(probs, 1).item()
            sampled.append(idx)
            log_prob += torch.log(probs[idx] + 1e-10)
        selected = list(dict.fromkeys(sampled))  # remove duplicates, preserve order
        return selected, log_prob

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the RL agent to discover the best vectorized symbolic expression.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data (can be multi‑dimensional).
        y : np.ndarray, shape (n_samples,)
            Target values.
        """
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
        """Return the best discovered neuron formula."""
        return self.neuron