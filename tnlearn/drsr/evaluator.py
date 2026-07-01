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

# drsr/evaluator.py
from __future__ import annotations
import time
import multiprocessing
import sys
import re
from typing import Any, Type, Optional, Dict
from abc import abstractmethod, ABC

from .template import SINGLE_VAR_TEMPLATE
from . import buffer


class Sandbox(ABC):
    @abstractmethod
    def run(
        self,
        program: str,
        function_to_run: str,
        function_to_evolve: str,
        inputs: Any,
        test_input: str,
        timeout_seconds: int,
        **kwargs
    ) -> tuple[Any, bool]:
        raise NotImplementedError


class LocalSandbox(Sandbox):
    def __init__(self, verbose=False, numba_accelerate=False, use_multiprocessing=None):
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate
        # 自动检测：Windows上默认禁用多进程（因为spawn开销大且易出错），其他系统默认启用
        if use_multiprocessing is None:
            self._use_multiprocessing = (sys.platform != 'win32')
        else:
            self._use_multiprocessing = use_multiprocessing

    def run(self, program: str, function_to_run: str, function_to_evolve: str,
            inputs: Any, test_input: str, timeout_seconds: int, **kwargs):
        dataset = inputs[test_input]
        if self._use_multiprocessing:
            # 使用多进程执行（用于Linux/macOS）
            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=self._compile_and_run_function,
                args=(program, function_to_run, function_to_evolve, dataset, self._numba_accelerate, result_queue, kwargs.get('seed', None))
            )
            process.start()
            process.join(timeout=timeout_seconds)
            if process.is_alive():
                process.terminate()
                process.join()
                results = (None, False)
            else:
                results = self._get_results(result_queue)
            if self._verbose:
                self._print_evaluation_details(program, results, **kwargs)
            return results
        else:
            # 直接执行（用于Windows或禁用多进程时）
            try:
                all_globals_namespace = {}
                exec(program, all_globals_namespace)
                func = all_globals_namespace[function_to_run]
                results = func(dataset)
                if not isinstance(results, (tuple, list)):
                    return (None, False)
                if len(results) == 3:
                    score, result_data, params = results
                    extras = {'params': params.tolist() if params is not None else None}
                else:
                    score, result_data = results
                    extras = {}
                if not isinstance(score, (int, float)):
                    return (None, False)
                if self._verbose:
                    print(f"Evaluation successful, score: {score}")
                return (score, True, result_data, extras.get('params'))
            except Exception as e:
                print(f"Execution Error in direct mode: {e}")
                import traceback
                traceback.print_exc()
                return (None, False)

    def _get_results(self, queue):
        for _ in range(5):
            if not queue.empty():
                return queue.get_nowait()
            time.sleep(0.1)
        return None, False

    def _print_evaluation_details(self, program, results, **kwargs):
        print('================= Evaluated Program =================')
        print(program[:200] + '...' if len(program) > 200 else program)
        print(f'Score: {results}\n=====================================================\n\n')

    def _compile_and_run_function(self, program, function_to_run, function_to_evolve,
                                  dataset, numba_accelerate, result_queue, seed=None):
        try:
            import os as _os
            import random as _random
            import numpy as _np
            if seed is not None:
                _os.environ['PYTHONHASHSEED'] = str(seed)
                _random.seed(seed)
                _np.random.seed(seed)
        except Exception:
            pass
        try:
            all_globals_namespace = {}
            exec(program, all_globals_namespace)
            func = all_globals_namespace[function_to_run]
            results = func(dataset)
            if not isinstance(results, (tuple, list)):
                result_queue.put((None, False))
                return
            if len(results) == 3:
                score, result_data, params = results
                extras = {'params': params.tolist() if params is not None else None}
            else:
                score, result_data = results
                extras = {}
            if not isinstance(score, (int, float)):
                result_queue.put((None, False))
                return
            result_queue.put((score, True, result_data, extras.get('params')))
        except Exception as e:
            print(f"Execution Error in multiprocess mode: {e}")
            import traceback
            traceback.print_exc()
            result_queue.put((None, False))


class Evaluator:
    def __init__(
        self,
        database: buffer.ExperienceBuffer,
        function_to_evolve: str,
        function_to_run: str,
        inputs: Any,
        timeout_seconds: int = 30,
        sandbox_class: Type[Sandbox] = LocalSandbox,
        max_params: int = 10,
        n_restarts: int = 5,
        bfgs_maxiter: int = 200,
        complexity_penalty: float = 0.01,
        verbose: bool = False,
    ):
        self._database = database
        self._function_to_evolve = function_to_evolve
        self._function_to_run = function_to_run
        self._inputs = inputs
        self._timeout_seconds = timeout_seconds
        self._sandbox = sandbox_class(verbose=verbose)
        self._max_params = max_params
        self._n_restarts = n_restarts
        self._bfgs_maxiter = bfgs_maxiter
        self._complexity_penalty = complexity_penalty
        self._verbose = verbose

    def analyse(
        self,
        sample: str,
        island_id: Optional[int],
        version_generated: Optional[int],
        **kwargs
    ) -> None:
        # Safety check: reject samples with params inside function arguments
        if re.search(r'np\.(exp|sin|cos|tanh|log|sqrt)\s*\(\s*params\s*\[', sample):
            print(f"Rejected sample due to params inside function argument: {sample[:100]}...")
            return

        program = SINGLE_VAR_TEMPLATE.format(
            max_params=self._max_params,
            n_restarts=self._n_restarts,
            bfgs_maxiter=self._bfgs_maxiter,
            equation_body=sample
        )

        scores_per_test = {}
        time_reset = time.time()

        for current_input in self._inputs:
            run_results = self._sandbox.run(
                program,
                self._function_to_run,
                self._function_to_evolve,
                self._inputs,
                current_input,
                self._timeout_seconds,
                **kwargs
            )
            if isinstance(run_results, tuple) and len(run_results) == 4:
                test_output, runs_ok, result_data, params = run_results
            else:
                test_output, runs_ok = run_results
                params = None

            if runs_ok and test_output is not None and isinstance(test_output, (int, float)):
                scores_per_test[current_input] = test_output

        evaluate_time = time.time() - time_reset

        if scores_per_test:
            # Apply complexity penalty
            n_terms = len(re.findall(r'params\s*\[\s*\d+\s*\]', sample))
            if n_terms == 0:
                n_terms = 1
            penalty = self._complexity_penalty * (n_terms - 1)
            penalized_scores = {k: v - penalty for k, v in scores_per_test.items()}
            if scores_per_test:
                avg_score = sum(scores_per_test.values()) / len(scores_per_test)
                if self._verbose:
                    print(f"[Evaluator] Body: {sample[:100]}... score={avg_score:.6f}")
                    
            self._database.register_program(
                sample,
                island_id,
                penalized_scores,
                **kwargs,
                evaluate_time=evaluate_time
            )
        else:
            print(f"Warning: No valid scores for sample:\n{sample}")