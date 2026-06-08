# Copyright 2024 Meng WANG. All Rights Reserved.
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
Reinforcement Learning‑based Symbolic Regressor with Polynomial and Trigonometric Basis.

This module implements a policy gradient (REINFORCE) agent that selects a subset of
basis functions (polynomials and optionally trigonometric terms) to form a symbolic
expression. Coefficients are fitted via Ridge regression on the training set, and the
validation R² is used as the reward. The discovered expression can be used as a neuron
formula in MLPRegressor.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from typing import List, Optional, Tuple, Dict, Any, Union
import warnings

warnings.filterwarnings("ignore")


class PolicyNetwork(nn.Module):
    """
    Policy network that outputs a probability distribution over the basis functions.

    Parameters
    ----------
    n_funcs : int
        Number of candidate basis functions.
    hidden_dim : int, default=64
        Number of neurons in each hidden layer.
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
        """
        Forward pass.

        Parameters
        ----------
        state : torch.Tensor, shape (1,)
            Dummy state (not used, kept for API compatibility).

        Returns
        -------
        torch.Tensor, shape (n_funcs,)
            Probability distribution over basis functions.
        """
        return self.net(state)


class RLRegressor:
    """
    Reinforcement Learning symbolic regressor that selects basis functions
    (polynomials and optionally trigonometric terms) and fits coefficients via Ridge.

    The agent learns to choose which basis functions to include, subject to a
    maximum number of terms (`max_terms`). Constant term can be forced to always
    be present. The reward is the R² score on the validation set after fitting
    coefficients with Ridge regression.

    Parameters
    ----------
    max_power : int, default=3
        Maximum exponent for polynomial terms (x**p).
    max_terms : int, default=5
        Maximum number of basis functions selected per expression.
    max_freq : int, default=2
        Maximum integer frequency for trigonometric functions (k in sin(k*x)).
    use_trigonometric : bool, default=True
        Whether to include sin(k*x), cos(k*x) and their products as candidates.
    alpha : float, default=0.1
        Regularisation strength for Ridge regression.
    force_constant : bool, default=True
        If True, the constant term (1) is always included.
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

    Attributes
    ----------
    best_expr : str
        The best discovered symbolic expression (with numeric coefficients).
    best_score : float
        The best validation R² score achieved.
    neuron : str
        Alias for `best_expr`, compatible with MLPRegressor neuron interface.
    """

    def __init__(
        self,
        max_power: int = 3,
        max_terms: int = 5,
        max_freq: int = 2,
        use_trigonometric: bool = True,
        alpha: float = 0.1,
        force_constant: bool = True,
        random_state: int = 42,
        max_episodes: int = 100,
        val_split: float = 0.2,
        lr_rl: float = 1e-3,
        gamma: float = 0.99,
        hidden_dim: int = 64,
        verbose: bool = True,
    ):
        # Reproducibility
        self.random_state = random_state
        np.random.seed(random_state)
        torch.manual_seed(random_state)

        self.max_power = max_power
        self.max_terms = max_terms
        self.max_freq = max_freq
        self.use_trigonometric = use_trigonometric
        self.alpha = alpha
        self.force_constant = force_constant
        self.max_episodes = max_episodes
        self.val_split = val_split
        self.lr_rl = lr_rl
        self.gamma = gamma
        self.verbose = verbose

        # Generate candidate basis functions
        self.candidate_funcs: List[str] = self._generate_candidates()
        self.n_funcs = len(self.candidate_funcs)
        # Locate constant term if present
        self.const_idx: Optional[int] = None
        if "torch.ones_like(x)" in self.candidate_funcs:
            self.const_idx = self.candidate_funcs.index("torch.ones_like(x)")

        self.policy = PolicyNetwork(self.n_funcs, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr_rl)

        self.saved_log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []

        self.best_expr: Optional[str] = None
        self.best_score: float = -float("inf")
        self.neuron: Optional[str] = None

    def _generate_candidates(self) -> List[str]:
        """
        Generate the list of candidate basis function strings.

        All functions are written in a form that can be evaluated with `torch` tensors.
        The constant term is represented by `torch.ones_like(x)`.

        Returns
        -------
        List[str]
            List of candidate expressions.
        """
        candidates = ["torch.ones_like(x)"]  # constant term
        # Polynomial terms
        for p in range(1, self.max_power + 1):
            candidates.append(f"x**{p}")

        if self.use_trigonometric:
            # Trigonometric terms and their products with powers of x
            for p in range(0, self.max_power + 1):
                for k in range(1, self.max_freq + 1):
                    sin_term = f"torch.sin({k}*x)"
                    cos_term = f"torch.cos({k}*x)"
                    if p == 0:
                        candidates.append(sin_term)
                        candidates.append(cos_term)
                        for k2 in range(1, self.max_freq + 1):
                            if k2 == k:
                                continue
                            candidates.append(f"{sin_term} * {cos_term}")
                    else:
                        x_pow = f"x**{p}"
                        candidates.append(f"{sin_term} * {x_pow}")
                        candidates.append(f"{cos_term} * {x_pow}")
                        for k2 in range(1, self.max_freq + 1):
                            if k2 == k:
                                continue
                            candidates.append(f"{sin_term} * {cos_term} * {x_pow}")

        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for f in candidates:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        return unique

    @staticmethod
    def _eval_func(func_str: str, x_tensor: torch.Tensor) -> np.ndarray:
        """
        Safely evaluate a basis function string on a torch tensor.

        Parameters
        ----------
        func_str : str
            Expression to evaluate (can use `x`, `torch`, `np`).
        x_tensor : torch.Tensor, shape (n_samples,)
            Input values.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Evaluated values as a numpy array.
        """
        try:
            # Provide torch and x in the evaluation namespace
            result = eval(func_str, globals(), {'x': x_tensor, 'torch': torch})
            if isinstance(result, torch.Tensor):
                return result.detach().numpy().flatten()
            else:
                return np.full(len(x_tensor), float(result), dtype=np.float64)
        except Exception as e:
            raise RuntimeError(f"Failed to evaluate function '{func_str}': {e}")

    def _evaluate_mask(
        self,
        selected_indices: List[int],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[float, Optional[np.ndarray], List[str]]:
        """
        Evaluate a selected set of basis functions.

        - If `force_constant` is True and the constant term is not in `selected_indices`,
          it is automatically added.
        - Fits coefficients using Ridge regression on the training set.
        - Computes validation R² as reward.

        Parameters
        ----------
        selected_indices : List[int]
            Indices of chosen basis functions.
        X_train, y_train : np.ndarray
            Training data (for fitting coefficients).
        X_val, y_val : np.ndarray
            Validation data (for reward calculation).

        Returns
        -------
        reward : float
            Validation R² score (or -1e6 if invalid).
        coeffs : np.ndarray or None
            Fitted coefficients for the selected functions (in order).
        funcs : List[str]
            Actual basis function strings used (including constant if forced).
        """
        # Force inclusion of constant term if required
        if self.force_constant and self.const_idx is not None:
            if self.const_idx not in selected_indices:
                selected_indices = [self.const_idx] + selected_indices

        if not selected_indices:
            return -1e6, None, []

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)

        A_train = []
        A_val = []
        selected_funcs = []

        for idx in selected_indices:
            func_str = self.candidate_funcs[idx]
            try:
                f_train = self._eval_func(func_str, X_train_t)
                f_val = self._eval_func(func_str, X_val_t)
                A_train.append(f_train)
                A_val.append(f_val)
                selected_funcs.append(func_str)
            except RuntimeError:
                # Skip problematic functions
                continue

        if not A_train:
            return -1e6, None, []

        A_train = np.column_stack(A_train)
        A_val = np.column_stack(A_val)
        ridge = Ridge(alpha=self.alpha, fit_intercept=False)
        ridge.fit(A_train, y_train)
        coeffs = ridge.coef_
        y_pred_val = A_val @ coeffs
        r2 = r2_score(y_val, y_pred_val)
        return r2, coeffs, selected_funcs

    def _build_expr(self, selected_funcs: List[str], coeffs: np.ndarray) -> str:
        """
        Build a human‑readable expression string from selected functions and coefficients.

        The output format is compatible with the `neuron` attribute of `VecSymRegressor`,
        using `@` to separate coefficient and function.

        Parameters
        ----------
        selected_funcs : List[str]
            Basis function strings.
        coeffs : np.ndarray
            Corresponding coefficients.

        Returns
        -------
        str
            Expression like "2.0@x**2 + 3.0@x + 1.0".
        """
        terms = []
        for func, c in zip(selected_funcs, coeffs):
            if abs(c) < 1e-6:
                continue
            if abs(c - round(c)) < 1e-6:
                c_str = str(int(round(c)))
            else:
                c_str = f"{c:.4f}"
            # Constant term: just the coefficient (no '@' or function)
            if func == "torch.ones_like(x)":
                terms.append(c_str)
            else:
                # Wrap composite expressions in parentheses
                if any(op in func for op in ['+', '-', '*', '/', '**', 'sin', 'cos']):
                    func = f"({func})"
                terms.append(f"{c_str}@{func}")
        if not terms:
            return "0"
        expr = " + ".join(terms)
        expr = expr.replace("+ -", "- ")
        return expr

    def _select_action(self, state: torch.Tensor) -> Tuple[List[int], torch.Tensor]:
        """
        Sample up to `max_terms` basis functions (with replacement) and remove duplicates.

        The log probability is the sum of log probabilities of each sampled index.

        Parameters
        ----------
        state : torch.Tensor, shape (1,)
            Dummy state.

        Returns
        -------
        selected : List[int]
            Unique indices of chosen basis functions.
        log_prob : torch.Tensor
            Sum of log probabilities for the sampled indices.
        """
        probs = self.policy(state).squeeze(0)  # (n_funcs,)
        sampled = []
        log_prob = 0.0
        for _ in range(self.max_terms):
            idx = torch.multinomial(probs, 1).item()
            sampled.append(idx)
            log_prob += torch.log(probs[idx] + 1e-10)
        # Remove duplicates while preserving order
        selected = list(dict.fromkeys(sampled))
        return selected, log_prob

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the RL agent to discover the best symbolic expression.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, 1)
            Input feature (must be one‑dimensional).
        y : np.ndarray, shape (n_samples,)
            Target values.

        Raises
        ------
        ValueError
            If X has more than one feature.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != 1:
            raise ValueError("RLRegressor currently only supports single‑feature input (X.shape[1] == 1).")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_split, random_state=self.random_state
        )

        for episode in range(self.max_episodes):
            state = torch.tensor([0.0])
            selected, log_prob = self._select_action(state)
            self.saved_log_probs.append(log_prob)

            reward, coeffs, funcs = self._evaluate_mask(selected, X_train, y_train, X_val, y_val)
            self.rewards.append(reward)

            # Update best expression if improved
            if reward > self.best_score:
                self.best_score = reward
                if coeffs is not None:
                    self.best_expr = self._build_expr(funcs, coeffs)
                else:
                    self.best_expr = "0"
                if self.verbose:
                    # Overwritable line for compact progress
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

            # Clear buffers for next episode
            self.saved_log_probs = []
            self.rewards = []

            # Periodic status update (only if no improvement and verbose)
            if self.verbose and (episode + 1) % 10 == 0 and reward <= self.best_score:
                print(f"\rEpisode {episode+1}/{self.max_episodes}, current best R²={self.best_score:.4f}   ", end="")

        if self.verbose:
            # Final newline after progress updates
            print()
            print(f"RL search completed. Best expression: {self.best_expr}, validation R² = {self.best_score:.4f}")

        self.neuron = self.best_expr

    def get_neuron(self) -> Optional[str]:
        """
        Return the best discovered neuron formula.

        Returns
        -------
        str or None
            Expression string or None if not fitted yet.
        """
        return self.neuron