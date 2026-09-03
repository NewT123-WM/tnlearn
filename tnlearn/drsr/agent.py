# drsr/agent.py
# Copyright 2023 DeepMind Technologies Limited
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
# This file is based on the DRSR project (https://github.com/scientific-intelligent-modelling/drsr)
# and has been modified for vectorized symbolic regression.

from __future__ import annotations
import os
import json
import re
import numpy as np
from typing import Optional, List, Dict, Any, Union

from . import pipeline
from . import config as config_lib
from . import sampler
from . import evaluator
from .template import SINGLE_VAR_TEMPLATE_BASE, SINGLE_VAR_TEMPLATE_LEGACY
from ..operator.inner_product import convert_innerproduct_to_pretty


class LLMSymRegressor:
    def __init__(
        self,
        llm_config: Dict[str, Any],
        max_iterations: int = 20,
        samples_per_iteration: int = 8,
        background: str = "",
        verbose: Union[bool, int] = True,
        max_params: int = 10,
        n_restarts: int = 5,
        bfgs_maxiter: int = 200,
        extra_prompt: str = "",
        exp_dir: Optional[str] = None,
        save: bool = False,
        mode: str = 'base',          # NEW: 'base' or 'legacy'
        **kwargs
    ):
        self.llm_config = llm_config
        self.max_iterations = max_iterations
        self.samples_per_iteration = samples_per_iteration
        self.background = background
        self.verbose = 1 if verbose else 0 if isinstance(verbose, bool) else max(0, min(2, int(verbose)))
        self.max_params = max_params
        self.n_restarts = n_restarts
        self.bfgs_maxiter = bfgs_maxiter
        self.extra_prompt = extra_prompt
        self.save = save
        self.mode = mode.lower()
        if self.mode not in ('base', 'legacy'):
            raise ValueError("mode must be 'base' or 'legacy'")

        if not save and exp_dir is None:
            self.exp_dir = None
        else:
            self.exp_dir = exp_dir or "./experiments"
        self._llm_client = None
        self._database = None

        self.best_equation_: Optional[str] = None
        self.best_score_: float = -np.inf
        self.best_params_: Optional[np.ndarray] = None
        self.best_multivariate_func_ = None

        self._X_fitted = None
        self._y_fitted = None

        self.neuron = None 

    def _build_specification(self) -> str:
        background_comment = f"# {self.background}" if self.background else ""
        if self.mode == 'base':
            spec = f"""
{background_comment}
import numpy as np
from scipy.optimize import minimize

# 标量内积函数
def IP(w, expr):
    return w * np.sum(expr, axis=1)

MAX_NPARAMS = {self.max_params}
N_RESTARTS = {self.n_restarts}
BFGS_MAXITER = {self.bfgs_maxiter}

@equation.evolve
def equation(x, params):
    \"\"\"
    Vectorized equation on full input matrix x (N×d).
    x: 2D array (N, d)
    params: 1D array of coefficients
    Returns: (N,) predictions
    \"\"\"
    return IP(params[0], x)   # placeholder

@evaluate.run
def evaluate_run(data):
    \"\"\"
    data: dict with keys 'inputs' and 'outputs'
    Returns: (score, result_data, best_params)
    \"\"\"
    inputs = data['inputs']
    outputs = data['outputs']
    return 0.0, np.zeros((inputs.shape[0], inputs.shape[1] + 2)), np.zeros(MAX_NPARAMS)
"""
        else:  # legacy
            spec = f"""
{background_comment}
import numpy as np
from scipy.optimize import minimize

MAX_NPARAMS = {self.max_params}
N_RESTARTS = {self.n_restarts}
BFGS_MAXITER = {self.bfgs_maxiter}

@equation.evolve
def equation(x, params):
    \"\"\"
    Single-variable mathematical expression.
    x: scalar or 1D array (will be vectorized)
    params: 1D array of coefficients
    \"\"\"
    return params[0] * x   # placeholder

@evaluate.run
def evaluate_run(data):
    \"\"\"
    data: dict with keys 'inputs' and 'outputs'
    Returns: (score, result_data, best_params)
    \"\"\"
    inputs = data['inputs']
    outputs = data['outputs']
    return 0.0, np.zeros((inputs.shape[0], inputs.shape[1] + 2)), np.zeros(MAX_NPARAMS)
"""
        return spec

    def _prepare_dataset(self, X: np.ndarray, y: np.ndarray) -> dict:
        return {'data': {'inputs': X, 'outputs': y}}

    def _get_best_from_database(self):
        if self._database is None:
            return None
        best_idx = np.argmax(self._database._best_score_per_island)
        if self._database._best_score_per_island[best_idx] > -np.inf:
            return self._database._best_program_per_island[best_idx]
        return None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LLMSymRegressor':
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self._X_fitted = X
        self._y_fitted = y

        specification = self._build_specification()
        dataset = self._prepare_dataset(X, y)

        from .llm import ClientFactory
        if 'model' not in self.llm_config:
            raise ValueError("llm_config must contain 'model' in format 'provider/model'")
        client = ClientFactory.from_config(self.llm_config)
        if hasattr(client, 'kwargs'):
            client.kwargs['max_tokens'] = 2048

        buffer_config = config_lib.ExperienceBufferConfig(
            functions_per_prompt=3,
            num_islands=5,
            reset_period=3600,
            cluster_sampling_temperature_init=0.8,
            cluster_sampling_temperature_period=30000
        )
        cfg = config_lib.Config(
            experience_buffer=buffer_config,
            num_samplers=1,
            num_evaluators=1,
            samples_per_prompt=self.samples_per_iteration,
            evaluate_timeout_seconds=30,
            wall_time_limit_seconds=None
        )
        class_config = config_lib.ClassConfig(
            llm_class=sampler.LocalLLM,
            sandbox_class=evaluator.LocalSandbox
        )

        max_samples = self.max_iterations * self.samples_per_iteration
        if self.mode == 'base':
            initial_body = "return IP(params[0], x)"
        else:
            initial_body = "return params[0] * x"

        log_dir = self.exp_dir if self.save else None
        if self.save and self.exp_dir:
            os.makedirs(self.exp_dir, exist_ok=True)

        self._database = pipeline.main(
            specification=specification,
            inputs=dataset,
            config=cfg,
            max_sample_nums=max_samples,
            class_config=class_config,
            log_dir=log_dir,
            llm_client=client,
            initial_body=initial_body,
            persist_all_samples=False,
            seed=None,
            extra_prompt=self.extra_prompt,
            verbose=self.verbose > 0,
            mode=self.mode,          # pass mode to pipeline
        )

        if self.save:
            samples_dir = os.path.join(self.exp_dir, 'samples')
            if not os.path.isdir(samples_dir):
                raise RuntimeError("Samples directory not found; pipeline may have failed.")
            top_files = [f for f in os.listdir(samples_dir) if f.startswith('top') and f.endswith('.json')]
            if not top_files:
                raise RuntimeError("No top files found; no equation discovered.")
            best_file = os.path.join(samples_dir, top_files[0])
            with open(best_file, 'r') as f:
                data = json.load(f)
            body = data.get('function', '')
            lines = body.splitlines()
            self.best_equation_ = None
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('return '):
                    self.best_equation_ = stripped
                    break
            if self.best_equation_ is None:
                raise RuntimeError("Could not extract equation body from saved file.")
            nmse = data.get('nmse')
            mse = data.get('mse')
            if nmse is not None:
                self.best_score_ = -nmse
            elif mse is not None:
                self.best_score_ = -mse
            else:
                self.best_score_ = -np.inf
        else:
            best_body = self._get_best_from_database()
            if best_body is None:
                raise RuntimeError("No equation found in memory.")
            self.best_equation_ = best_body

        # Refine with high-precision BFGS
        temp_eval = evaluator.Evaluator(
            database=None,
            function_to_evolve='equation',
            function_to_run='evaluate_run',
            inputs=dataset,
            timeout_seconds=60,
            sandbox_class=evaluator.LocalSandbox,
            max_params=self.max_params,
            n_restarts=20,
            bfgs_maxiter=500,
            complexity_penalty=0.0,
            mode=self.mode,          # pass mode
        )
        if self.mode == 'base':
            template = SINGLE_VAR_TEMPLATE_BASE
        else:
            template = SINGLE_VAR_TEMPLATE_LEGACY
        program = template.format(
            max_params=self.max_params,
            n_restarts=20,
            bfgs_maxiter=500,
            equation_body=self.best_equation_
        )
        sandbox = evaluator.LocalSandbox(verbose=self.verbose > 0, mode=self.mode)
        test_input = list(dataset.keys())[0]
        run_results = sandbox.run(
            program,
            'evaluate_run',
            'equation',
            dataset,
            test_input,
            timeout_seconds=60,
            seed=None
        )
        if run_results is not None and run_results[1]:
            if len(run_results) >= 4:
                score, _, _, params = run_results
            else:
                score, _ = run_results
                params = None
            if params is not None:
                self.best_params_ = np.array(params)
                self.best_score_ = score
                namespace = {}
                exec(program, namespace)
                eq_func = namespace['equation']
                def multivariate_predict(X):
                    pred = np.zeros(X.shape[0])
                    for i in range(X.shape[1]):
                        pred += eq_func(X[:, i], self.best_params_)
                    if self._y_fitted.ndim == 2 and self._y_fitted.shape[1] > 1:
                        return np.tile(pred[:, None], (1, self._y_fitted.shape[1]))
                    return pred
                self.best_multivariate_func_ = multivariate_predict
        self.neuron = self.get_neuron_formula()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.best_multivariate_func_ is None or self.best_params_ is None:
            raise RuntimeError("Model not fitted or no valid equation found.")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.best_multivariate_func_(X)

    def get_neuron_formula(self) -> str:
        if self.best_equation_ is None:
            raise RuntimeError("No equation found.")
        expr = self.best_equation_.strip()
        if expr.startswith('return '):
            expr = expr[7:].strip()
        if self.best_params_ is not None:
            for i, val in enumerate(self.best_params_):
                expr = re.sub(rf'params\s*\[\s*{i}\s*\]', f'{val:.6f}', expr)
        expr = expr.replace('np.', 'torch.')
        # In base mode we have IP; in legacy we might have * which should become @
        # But we'll apply general conversion:
        expr = re.sub(r'(?<!\*)\*(?!\*)', '@', expr)
        expr = re.sub(r'(torch\.exp\s*\()([^)]*)\)',
                      lambda m: m.group(1) + re.sub(r'@', '*', m.group(2)) + ')', expr)
        expr = re.sub(r'x\s*@\s*([\d.]+)', r'\1@x', expr)
        if self.mode == 'base':
            return convert_innerproduct_to_pretty(expr)
        else:
            return expr