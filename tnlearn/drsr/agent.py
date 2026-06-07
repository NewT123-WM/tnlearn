# Copyright 2024 Meng WANG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Top-level DrSR Agent for LLM-based symbolic regression using official evaluator.

This module provides a symbolic regression agent that combines large language
models (LLMs) with numerical optimization to discover mathematical equations
from data. The discovered equations can be used as neuron formulas in MLPRegressor.
"""

import numpy as np
import re
import os
from typing import Optional, List, Dict, Any, Tuple, Union

from tnlearn.base1 import BaseModel1
from tnlearn.drsr.llm import ClientFactory
from tnlearn.drsr import code_manipulation
from tnlearn.drsr import evaluator
from tnlearn.drsr import buffer as drsr_buffer
from tnlearn.drsr import config as drsr_config
from tnlearn.drsr import prompt_config as pc
from tnlearn.drsr.analyzer import DataAnalyzer
from . import evaluate_on_problems


# ---------- 简单经验缓冲类 ----------
class SimpleExperienceBuffer:
    """A simple text-based experience buffer for storing successful/failed equation patterns.

    This buffer stores short textual summaries of good or bad equation samples,
    which are later injected into the LLM prompt as human‑readable ideas.
    """

    def __init__(self):
        self.experiences = []
        self.lessons = []

    def add_experience(self, exp: str, is_success: bool = True):
        """Add an experience summary to the buffer.

        Parameters
        ----------
        exp : str
            Summary text describing the experience.
        is_success : bool, default=True
            Whether the experience came from a successful candidate.
        """
        if is_success:
            self.experiences.append(exp)
            if len(self.experiences) > 10:
                self.experiences.pop(0)
        else:
            self.lessons.append(exp)
            if len(self.lessons) > 10:
                self.lessons.pop(0)

    def get_experiences(self, max_count: int = 5) -> List[str]:
        """Retrieve the most recent successful experiences.

        Parameters
        ----------
        max_count : int, default=5
            Maximum number of experiences to return.

        Returns
        -------
        list of str
            The most recent experience summaries.
        """
        return self.experiences[-max_count:] if self.experiences else []


# ---------- 官方规范模板（与官方 DrSR 相同）----------
SPECIFICATION_TEMPLATE = '''
import numpy as np
from scipy.optimize import minimize

MAX_NPARAMS = 10

@equation.evolve
def equation({args}, params):
    """
    Mathematical function representing the target variable.
    """
    # Write your equation here
    return ...

@evaluate.run
def evaluate_run(inputs, outputs):
    """
    Evaluate the equation on given data.
    """
    def loss(params):
        y_pred = equation(*inputs.T, params)
        return np.mean((y_pred - outputs) ** 2)

    # Multi-start BFGS
    best_loss = np.inf
    best_params = None
    for _ in range(5):
        x0 = np.random.uniform(-1, 1, size=MAX_NPARAMS)
        res = minimize(loss, x0, method='BFGS', options={{'maxiter': 200, 'disp': False}})
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x
    y_pred = equation(*inputs.T, best_params)
    residuals = outputs - y_pred
    score = -best_loss
    result_data = np.column_stack((inputs, outputs, residuals))
    return score, result_data
'''


class LLMSymRegressor(BaseModel1):
    """LLM-based symbolic regressor using DrSR dual reasoning with official evaluator.

    This class discovers symbolic equations from data using an LLM-driven search
    combined with BFGS parameter optimization. It can output a neuron formula
    compatible with MLPRegressor.

    Parameters
    ----------
    llm_config : dict, optional
        Configuration for the LLM client, must contain 'model' key in 'provider/model' format.
        May also contain an 'extra_prompt' key for additional user-provided instructions.
    max_iterations : int, default=20
        Maximum number of evolutionary iterations.
    samples_per_iteration : int, default=8
        Number of candidate equations generated per iteration.
    background : str, optional
        Physical background description to guide the LLM.
    random_state : int, optional
        Seed for reproducibility.
    verbose : int or bool, default=True
        Verbosity level. If int: 0=quiet, 1=basic progress, 2=detailed debug.
        If bool: True -> 1, False -> 0.
    extra_prompt : str, optional
        Additional user-provided text appended to the prompt after the equation template.

    Attributes
    ----------
    best_equation_ : str
        The best discovered equation body (with param placeholders).
    best_score_ : float
        The best negative MSE score (higher is better).
    best_params_ : np.ndarray
        Optimized parameter values corresponding to the best equation.
    neuron : str
        Neuron formula with numeric coefficients, ready to use in MLPRegressor.
    """

    def __init__(
        self,
        llm_config: Optional[Dict[str, Any]] = None,
        max_iterations: int = 20,
        samples_per_iteration: int = 8,
        background: Optional[str] = None,
        random_state: Optional[int] = None,
        verbose: Union[bool, int] = True,
        extra_prompt: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        self.max_iterations = max_iterations
        self.samples_per_iteration = samples_per_iteration
        self.background = background
        self.random_state = random_state
        # Normalize verbose to int (0,1,2)
        if isinstance(verbose, bool):
            self.verbose = 1 if verbose else 0
        else:
            self.verbose = max(0, min(2, int(verbose)))

        self._llm_config = llm_config or {}
        self._llm_client = None
        # Extract extra_prompt from config if not explicitly passed
        if extra_prompt is None and 'extra_prompt' in self._llm_config:
            extra_prompt = self._llm_config['extra_prompt']
        self.extra_prompt = extra_prompt or ""

        self._template = None
        self._evaluator = None
        self._database = None
        self._inputs_for_eval = None

        self._simple_buffer = SimpleExperienceBuffer()

        self.best_equation_: Optional[str] = None
        self.best_score_: float = -float('inf')
        self.best_params_: Optional[np.ndarray] = None
        self.best_multivariate_func_ = None
        self.history_: List[Dict[str, Any]] = []

        self._feature_names: Optional[List[str]] = None
        self._target_name: str = 'y'
        self._X_fitted: Optional[np.ndarray] = None
        self._y_fitted: Optional[np.ndarray] = None

    # ----------------------------------------------------------------------
    # Initialization and helper methods
    # ----------------------------------------------------------------------
    def _setup_official_with_data(self, X: np.ndarray, y: np.ndarray):
        """Initialize official evaluator and experience buffer with data."""
        n_features = X.shape[1]
        args_str = ", ".join([f"x{i}" for i in range(n_features)])
        spec = SPECIFICATION_TEMPLATE.format(args=args_str)

        self._template = code_manipulation.text_to_program(spec)
        # Enhanced buffer configuration for better evolutionary diversity
        buffer_config = drsr_config.ExperienceBufferConfig(
            functions_per_prompt=3,                     # Number of examples in prompt
            num_islands=5,                              # Number of islands for diversity
            reset_period=3600,
            cluster_sampling_temperature_init=0.5,     # Higher temperature for exploration
            cluster_sampling_temperature_period=30000
        )
        self._database = drsr_buffer.ExperienceBuffer(buffer_config, self._template, 'equation')
        self._inputs_for_eval = {'data': {'inputs': X, 'outputs': y}}
        self._evaluator = evaluator.Evaluator(
            database=self._database,
            template=self._template,
            function_to_evolve='equation',
            function_to_run='evaluate_run',
            inputs=self._inputs_for_eval,
            timeout_seconds=30,
            sandbox_class=evaluator.LocalSandbox
        )
        self._evaluator._sandbox._verbose = self.verbose > 0

    def _get_llm_client(self):
        """Create or return the LLM client instance."""
        if self._llm_client is None:
            if 'model' not in self._llm_config:
                raise ValueError("llm_config must contain 'model' in format 'provider/model'")
            llm_verbose = self.verbose > 0
            self._llm_client = ClientFactory.from_config(self._llm_config, verbose=llm_verbose)
            temperature = self._llm_config.get('temperature', 0.6)
            max_tokens = self._llm_config.get('max_tokens', 1024)
            top_p = self._llm_config.get('top_p', 0.3)
            self._llm_client.kwargs.update({
                'temperature': temperature,
                'max_tokens': max_tokens,
                'top_p': top_p,
            })
        return self._llm_client

    def _create_multivariate_wrapper(self, univariate_body: str) -> str:
        """Wrap a univariate function body into a multivariate equation for evaluation.

        The wrapper applies the univariate expression to each feature column independently
        and sums the results. This allows the discovered univariate formula to be used
        on multi-feature datasets.

        Parameters
        ----------
        univariate_body : str
            Indented function body containing a return statement using variable 'x'.

        Returns
        -------
        str
            Python code defining a function `multivariate_equation(*args)` that takes
            all feature columns followed by params, and returns the aggregated prediction.
        """
        expr = univariate_body.strip()
        if expr.startswith('return'):
            expr = expr[6:].strip()
        lines = [
            "def multivariate_equation(*args):",
            "    import numpy as np",
            "    # args: all feature columns (n_features arrays) followed by params",
            "    params = args[-1]",
            "    feature_cols = args[:-1]",
            "    features = np.column_stack(feature_cols)",
            "    def _f(x):",
            f"        return {expr}",
            "    return np.sum(_f(features), axis=1)"
        ]
        return "\n".join(lines)

    def _create_safe_equation_body(self) -> str:
        """Return a safe fallback equation body: linear in the variable x."""
        return "    return x * 1.0"

    def _extract_equation_body_from_response(self, response_text: str) -> Optional[str]:
        """Extract the return statement from LLM response, ignoring 'def equation' lines."""
        code_pattern = r'```python\n(.*?)\n```'
        matches = re.findall(code_pattern, response_text, re.DOTALL)
        for match in matches:
            if 'def equation' in match:
                lines = match.splitlines()
                for line in lines:
                    if line.strip().startswith('return'):
                        return line.rstrip()
            else:
                if match.strip().startswith('return'):
                    return match.strip()
        return_pattern = r'^\s*return\s+[^;]+'
        return_match = re.search(return_pattern, response_text, re.MULTILINE)
        if return_match:
            return return_match.group(0).rstrip()
        return None

    # ----------------------------------------------------------------------
    # Core LLM interaction and candidate generation
    # ----------------------------------------------------------------------
    def _build_prompt(self, iteration: int) -> str:
        """Construct the prompt for the LLM, incorporating data insights and evolutionary experience.

        The prompt includes:
        - Statistical insights from the data.
        - Text-based experiences from simple buffer.
        - High-scoring equation examples from the official ExperienceBuffer (if available).
        - Instructions to generate a univariate equation body.
        - User-provided extra prompt (if any).
        """
        insight = DataAnalyzer.analyze(self._X_fitted, self._y_fitted, self._feature_names)
        insight_text = DataAnalyzer.format_insights_for_prompt(insight)

        # Simple text experiences
        experiences = self._simple_buffer.get_experiences()
        exp_text = ""
        if experiences:
            exp_text = pc.ideas_block_title + "\n".join(
                pc.idea_item_prefix.format(index=i+1) + e for i, e in enumerate(experiences[-5:])
            ) + "\n\n"

        # Official experience buffer (high-scoring equation skeletons)
        official_examples = ""
        if hasattr(self, '_database') and self._database is not None:
            try:
                prompt_obj = self._database.get_prompt()
                code = prompt_obj.code
                # Extract all `def equation_v...` function definitions
                matches = re.findall(
                    r'(def equation_v\d+\([^)]*\):.*?)(?=\n@equation\.evolve|\Z)',
                    code, re.DOTALL
                )
                if matches:
                    # Use the most recent two as examples
                    examples = matches[-2:] if len(matches) > 2 else matches
                    official_examples = (
                        "### Previous successful equation skeletons (for reference):\n" +
                        "\n".join(examples) + "\n"
                    )
            except Exception as e:
                if self.verbose >= 2:
                    print(f"Warning: Could not retrieve examples from official buffer: {e}")

        problem_desc = f"relation between features and {self._target_name}"
        head = pc.head_template.format(
            dependent=self._target_name,
            problem=problem_desc,
            independent="x (a single variable representing any feature)"
        )
        variables_block = (
            "Variables: Only one independent variable 'x' (the same expression applies to all features, "
            "then results are summed).\n"
        )
        background_text = f"Background: {self.background or 'Unknown physical system'}\n"
        data_insight_block = f"### Data Insight:\n{insight_text}\n"
        instruction = pc.instruction_prompt

        # Neutral instruction: no example bias, only allowed operations
        equation_template = (
            "# Write the body of the equation function (the part after the colon).\n"
            "# The function signature is: def equation(x, params):\n"
            "# Inside the function, write a return statement with proper indentation (4 spaces).\n"
            "# You may use numpy functions (np.sin, np.cos, np.tanh, np.exp, np.log, np.sqrt) and operators (+, -, *, /, **).\n"
            "# Important: In the final neuron formula, multiplication will be represented by '@' at the top level,\n"
            "#            but inside function arguments you must still use '*'. For example: params[0]*x + params[1]*np.sin(x)\n"
            "# Provide only the indented function body (no def line, no extra text):\n"
        )
        # Append user extra prompt if provided
        extra_block = ""
        if self.extra_prompt:
            extra_block = f"\n### Additional instructions:\n{self.extra_prompt}\n"

        prompt = (
            instruction + "\n" + head + "\n" + variables_block + background_text +
            data_insight_block + exp_text + official_examples + equation_template + extra_block
        )
        return prompt

    def _generate_candidates(self, iteration: int) -> List[Tuple[str, str]]:
        """Generate candidate equation bodies using the LLM."""
        prompt = self._build_prompt(iteration)
        candidates = []
        llm = self._get_llm_client()
        for _ in range(self.samples_per_iteration):
            try:
                resp = llm.chat([{"role": "user", "content": prompt}])
                content = resp.get('content', '')
                if self.verbose >= 2:
                    print("=" * 60)
                    print("RAW LLM RESPONSE:")
                    print(content)
                    print("=" * 60)

                body = self._extract_equation_body_from_response(content)
                if body is None or not body.strip():
                    if self.verbose >= 2:
                        print("No valid equation body found, using safe equation body.")
                    body = self._create_safe_equation_body()
                else:
                    # Clean up the body: replace string-indexed params, ensure return exists, add indentation.
                    body = re.sub(r"params\s*\[\s*['\"][^'\"]+['\"]\s*\]", "params[0]", body)
                    if re.search(r'params\s*[\*\+\-\/]', body) and not re.search(r'params\s*\[', body):
                        body = re.sub(r'params\b', 'params[0]', body)
                    lines = [line.rstrip() for line in body.splitlines() if line.strip()]
                    stripped_lines = [line.lstrip() for line in lines]
                    if not stripped_lines:
                        body = self._create_safe_equation_body()
                    else:
                        body = '\n'.join(stripped_lines)
                        if not body.startswith('return'):
                            body = f"return {body}"
                        body = '\n'.join('    ' + line for line in body.splitlines())
                if not body or 'return' not in body:
                    body = self._create_safe_equation_body()
                if self.verbose >= 2:
                    print("Equation body (first 200 chars):", body[:200])
                candidates.append((body, content))
            except Exception as e:
                if self.verbose >= 2:
                    print(f"LLM call failed: {e}")
        return candidates

    # ----------------------------------------------------------------------
    # Evaluation, optimization, and buffer registration
    # ----------------------------------------------------------------------
    def _evaluate_candidate(self, univariate_body: str) -> Tuple[float, str, Optional[np.ndarray], Optional[np.ndarray]]:
        """Evaluate a univariate equation body by wrapping it and using BFGS optimization.

        This function also registers the candidate into the official ExperienceBuffer
        to maintain evolutionary history.

        Returns
        -------
        score : float
            Negative MSE score (higher is better).
        error_msg : str
            Empty string on success, error message on failure.
        residual : np.ndarray or None
            Residuals for each data point.
        optimized_params : np.ndarray or None
            Optimized parameters for the equation.
        """
        if self.verbose >= 2:
            print("_evaluate_candidate received body:\n", univariate_body[:200])
        wrapper = self._create_multivariate_wrapper(univariate_body)
        if self.verbose >= 2:
            print("Generated wrapper:\n", wrapper)
        namespace = {}
        try:
            exec(wrapper, namespace)
        except Exception as e:
            if self.verbose >= 2:
                print(f"Exec wrapper failed: {e}")
            return -float('inf'), str(e), None, None
        multivariate_equation = namespace['multivariate_equation']

        inputs = self._X_fitted
        outputs = self._y_fitted
        data = {'inputs': inputs, 'outputs': outputs}

        try:
            eval_out = evaluate_on_problems.evaluate(data, multivariate_equation, verbose=False)
            if isinstance(eval_out, tuple) and len(eval_out) == 3:
                score, result_data, optimized_params = eval_out
            else:
                score, result_data = eval_out
                optimized_params = None
            residual = result_data[:, -1] if result_data is not None else None

            # Register into official ExperienceBuffer to enable evolution
            try:
                dummy_func = code_manipulation.text_to_function(f"def equation(x, params):\n{univariate_body}")
                dummy_func.score = score
                dummy_func.optimized_params = optimized_params
                self._database.register_program(dummy_func, island_id=None, scores_per_test={'data': score})
            except Exception as e:
                if self.verbose >= 2:
                    print(f"Warning: Failed to register to ExperienceBuffer: {e}")

            return score, "", residual, optimized_params
        except Exception as e:
            if self.verbose >= 2:
                print(f"Evaluation error: {e}")
                print("Problematic body:\n", univariate_body[:200])
            return -float('inf'), str(e), None, None

    # ----------------------------------------------------------------------
    # Public API: fit, predict, get_neuron_formula
    # ----------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LLMSymRegressor':
        """Train the symbolic regressor on the given data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data.
        y : np.ndarray, shape (n_samples,)
            Target values.

        Returns
        -------
        self : LLMSymRegressor
            Fitted estimator.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        y = y.ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._X_fitted = X
        self._y_fitted = y
        self._feature_names = [f'x{i}' for i in range(X.shape[1])]

        self._setup_official_with_data(X, y)

        # Register a safe initial equation to populate the experience buffer
        initial_body = self._create_safe_equation_body()
        score, _, _, _ = self._evaluate_candidate(initial_body)
        if self.verbose >= 1:
            print(f"Initial safe equation registered with score {score:.6f}")

        if self.verbose >= 1:
            print("Analyzing data...")
            insight = DataAnalyzer.analyze(X, y, self._feature_names)
            print(DataAnalyzer.format_insights_for_prompt(insight))

        for it in range(self.max_iterations):
            if self.verbose >= 1:
                print(f"\nIteration {it+1}/{self.max_iterations}")

            candidates = self._generate_candidates(it)
            if not candidates:
                if self.verbose >= 1:
                    print("No valid candidates generated")
                continue

            for body, _ in candidates:
                score, error_msg, residual, params = self._evaluate_candidate(body)
                if score is None or not isinstance(score, (int, float)):
                    if self.verbose >= 2:
                        print(f"  Score: None (evaluation failed)")
                    continue
                if self.verbose >= 1:
                    print(f"  Score: {score:.6f}")

                if score > self.best_score_:
                    self.best_score_ = score
                    self.best_equation_ = body
                    wrapper = self._create_multivariate_wrapper(body)
                    namespace = {}
                    exec(wrapper, namespace)
                    self.best_multivariate_func_ = namespace['multivariate_equation']
                    self.best_params_ = params
                    if self.verbose >= 1:
                        print(f"    New best equation body:\n{body[:200]}...")

                if score > -1e9:
                    quality = "Good" if score > self.best_score_ * 0.9 else "Bad"
                    exp_summary = f"{quality} sample with score {score:.4f}"
                    self._simple_buffer.add_experience(exp_summary, is_success=(quality == "Good"))

                self.history_.append({
                    'iteration': it,
                    'score': score,
                    'equation': body,
                    'error': error_msg
                })

        if self.verbose >= 1:
            print(f"\nDrSR completed. Best score: {self.best_score_:.6f}")
            if self.best_equation_:
                print(f"Best equation body:\n{self.best_equation_}")
            else:
                print("No valid equation found.")
        # Provide the same interface as VecSymRegressor
        self.neuron = self.get_neuron_formula()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the best discovered equation.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            Predicted values.
        """
        if not hasattr(self, 'best_multivariate_func_') or self.best_multivariate_func_ is None:
            raise RuntimeError("Model not fitted yet.")
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != self._X_fitted.shape[1]:
            raise ValueError(f"Expected {self._X_fitted.shape[1]} features, got {X.shape[1]}")
        try:
            y_pred = self.best_multivariate_func_(X, self.best_params_)
            return y_pred.flatten()
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

    def get_neuron_formula(self) -> str:
        """Return the discovered equation as a string with numeric coefficients and torch functions.

        Note: This formula is for reference only. It may not be directly compatible with MLPRegressor
        due to limitations in CustomNeuronLayer. For predictions, please use the predict() method.
        """
        if not hasattr(self, 'best_equation_') or self.best_equation_ is None:
            raise RuntimeError("Model not fitted yet.")
        body = self.best_equation_
        expr = body.strip()
        if expr.startswith('return'):
            expr = expr[6:].strip()
        if self.best_params_ is not None:
            for i, val in enumerate(self.best_params_):
                expr = re.sub(rf'params\[\s*{i}\s*\]', f'{val:.6f}', expr)
        expr = expr.replace('np.sin', 'torch.sin')
        expr = expr.replace('np.cos', 'torch.cos')
        expr = expr.replace('np.tanh', 'torch.tanh')
        expr = expr.replace('np.exp', 'torch.exp')
        expr = expr.replace('np.log', 'torch.log')
        expr = expr.replace('np.sqrt', 'torch.sqrt')
        return expr

def discover_neuron_formula(
    X: np.ndarray,
    y: np.ndarray,
    llm_config: Dict[str, Any],
    max_iterations: int = 10,
    samples_per_iteration: int = 4,
    verbose: Union[bool, int] = True,
) -> str:
    """Convenience function to discover a neuron formula using LLMSymRegressor.

    Parameters
    ----------
    X : np.ndarray
        Input data.
    y : np.ndarray
        Target values.
    llm_config : dict
        LLM configuration (must include 'model' key). May also contain 'extra_prompt'.
    max_iterations : int, default=10
        Maximum number of evolutionary iterations.
    samples_per_iteration : int, default=4
        Number of candidates per iteration.
    verbose : bool or int, default=True
        Verbosity level.

    Returns
    -------
    str
        Neuron formula ready for use in MLPRegressor.
    """
    reg = LLMSymRegressor(
        llm_config=llm_config,
        max_iterations=max_iterations,
        samples_per_iteration=samples_per_iteration,
        verbose=verbose,
    )
    reg.fit(X, y)
    return reg.get_neuron_formula()