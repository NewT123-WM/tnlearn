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

# drsr/pipeline.py
from __future__ import annotations
from typing import Any, Tuple, Sequence, Optional
import numpy as np

from . import code_manipulation
from . import config as config_lib
from . import evaluator
from . import buffer
from . import sampler


def _extract_function_names(specification: str) -> Tuple[str, str]:
    run_functions = list(code_manipulation.yield_decorated(specification, 'evaluate', 'run'))
    if len(run_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@evaluate.run`.')
    evolve_functions = list(code_manipulation.yield_decorated(specification, 'equation', 'evolve'))
    if len(evolve_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@equation.evolve`.')
    return evolve_functions[0], run_functions[0]


def main(
    specification: str,
    inputs: Sequence[Any],
    config: config_lib.Config,
    max_sample_nums: Optional[int],
    class_config: config_lib.ClassConfig,
    **kwargs
) -> buffer.ExperienceBuffer:   # 返回 database
    function_to_evolve, function_to_run = _extract_function_names(specification)

    # Create database
    database = buffer.ExperienceBuffer(config.experience_buffer)

    extra_prompt = kwargs.get('extra_prompt', '')
    verbose = kwargs.get('verbose', False)

    # Main evaluators
    evaluators = []
    for _ in range(config.num_evaluators):
        evaluators.append(evaluator.Evaluator(
            database,
            function_to_evolve,
            function_to_run,
            inputs,
            timeout_seconds=config.evaluate_timeout_seconds,
            sandbox_class=class_config.sandbox_class,
            complexity_penalty=0.05,
            verbose=verbose,
        ))

    # Create profiler if log_dir provided
    log_dir = kwargs.get('log_dir')
    profiler = None
    if log_dir:
        from . import profile
        profiler = profile.Profiler(
            log_dir,
            samples_per_iteration=config.samples_per_prompt,
            persist_all_samples=kwargs.get('persist_all_samples', False),
            wandb_run=kwargs.get('wandb_run')
        )
        kwargs['profiler'] = profiler

    # Initial evaluation
    initial_body = kwargs.get('initial_body', "return params[0] * x")
    print(f"Initial evaluation with body: {initial_body}")
    evaluators[0].analyse(
        initial_body,
        island_id=None,
        version_generated=None,
        **kwargs
    )

    # Seed with nonlinear expressions (using high-precision evaluator)
    print("Seeding additional nonlinear expressions with high-precision BFGS...")
    seed_evaluator = evaluator.Evaluator(
        database,
        function_to_evolve,
        function_to_run,
        inputs,
        timeout_seconds=config.evaluate_timeout_seconds,
        sandbox_class=class_config.sandbox_class,
        max_params=10,
        n_restarts=10,
        bfgs_maxiter=300,
        complexity_penalty=0.05,
        verbose=verbose,
    )
    seed_bodies = [
        "return params[0]*x + params[1]*x**2",
    ]
    for seed in seed_bodies:
        seed_evaluator.analyse(
            seed,
            island_id=None,
            version_generated=None,
            **kwargs
        )

    # Print database size for debugging
    total = sum(len(island._clusters) for island in database._islands)
    print(f"After initial evaluation and seeding, database has {total} clusters across islands.")

    # Samplers
    samplers = [
        sampler.Sampler(
            database,
            evaluators,
            config.samples_per_prompt,
            config=config,
            max_sample_nums=max_sample_nums,
            llm_class=class_config.llm_class,
            llm_client=kwargs.get('llm_client'),
            extra_prompt=extra_prompt,
        )
        for _ in range(config.num_samplers)
    ]

    for s in samplers:
        s.sample(**kwargs)

    # 返回 database 以便调用方在内存中访问最佳个体
    return database