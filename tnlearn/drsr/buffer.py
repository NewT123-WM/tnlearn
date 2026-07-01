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

# drsr/buffer.py
from __future__ import annotations

import copy
import dataclasses
import time
from collections.abc import Mapping, Sequence
from typing import Any, Tuple, List, Optional
import logging
import numpy as np
import scipy

from . import config as config_lib

Signature = Tuple[float, ...]
ScoresPerTest = Mapping[Any, float]


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    if not np.all(np.isfinite(logits)):
        non_finites = set(logits[~np.isfinite(logits)])
        raise ValueError(f'`logits` contains non-finite value(s): {non_finites}')
    if not np.issubdtype(logits.dtype, np.floating):
        logits = np.array(logits, dtype=np.float32)
    result = scipy.special.softmax(logits / temperature, axis=-1)
    index = np.argmax(result)
    result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index + 1:])
    return result


def _reduce_score(scores_per_test: ScoresPerTest) -> float:
    test_scores = [scores_per_test[k] for k in scores_per_test.keys()]
    return sum(test_scores) / len(test_scores)


def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
    return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))


@dataclasses.dataclass(frozen=True)
class Prompt:
    code: str
    version_generated: int
    island_id: int


class Cluster:
    def __init__(self, score: float, implementation: str):
        self._score = score
        self._programs: List[str] = [implementation]
        self._lengths: List[int] = [len(implementation)]

    @property
    def score(self) -> float:
        return self._score

    def register_program(self, program: str) -> None:
        self._programs.append(program)
        self._lengths.append(len(program))

    def sample_program(self) -> str:
        normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (
            max(self._lengths) + 1e-6
        )
        probabilities = _softmax(-normalized_lengths, temperature=1.0)
        return np.random.choice(self._programs, p=probabilities)


class Island:
    def __init__(
        self,
        functions_per_prompt: int,
        cluster_sampling_temperature_init: float,
        cluster_sampling_temperature_period: int,
    ):
        self._functions_per_prompt = functions_per_prompt
        self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
        self._cluster_sampling_temperature_period = (
            cluster_sampling_temperature_period
        )
        self._clusters: dict[Signature, Cluster] = {}
        self._num_programs: int = 0

    def register_program(self, program: str, scores_per_test: ScoresPerTest) -> None:
        signature = _get_signature(scores_per_test)
        if signature not in self._clusters:
            score = _reduce_score(scores_per_test)
            self._clusters[signature] = Cluster(score, program)
        else:
            self._clusters[signature].register_program(program)
        self._num_programs += 1

    def get_prompt(self) -> tuple[str, int]:
        signatures = list(self._clusters.keys())
        if not signatures:
            return (
                "def equation_v0(x, params):\n    return params[0] * x\n\n"
                "def equation_v1(x, params):\n    return params[0]*x + params[1]*np.sin(x)\n\n"
                "def equation_v2(x, params):\n    return params[0]*x**2 + params[1]\n\n"
                "def equation_v3(x, params):\n    # write your return statement here\n", 3
            )

        cluster_scores = np.array(
            [self._clusters[signature].score for signature in signatures]
        )
        period = self._cluster_sampling_temperature_period
        temperature = self._cluster_sampling_temperature_init * (
            1 - (self._num_programs % period) / period
        )
        probabilities = _softmax(cluster_scores, temperature)

        functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)
        idx = np.random.choice(
            len(signatures), size=functions_per_prompt, p=probabilities
        )
        chosen_signatures = [signatures[i] for i in idx]
        implementations = []
        scores = []
        for signature in chosen_signatures:
            cluster = self._clusters[signature]
            implementations.append(cluster.sample_program())
            scores.append(cluster.score)

        indices = np.argsort(scores)
        sorted_implementations = [implementations[i] for i in indices]
        version_generated = len(sorted_implementations) + 1
        return self._generate_prompt(sorted_implementations), version_generated

    def _generate_prompt(self, implementations: Sequence[str]) -> str:
        lines = []
        for i, body in enumerate(implementations):
            if not body.startswith('    '):
                indented = '\n'.join('    ' + line for line in body.splitlines())
            else:
                indented = body
            lines.append(f"def equation_v{i}(x, params):\n{indented}")
        next_version = len(implementations)
        lines.append(f"def equation_v{next_version}(x, params):\n    # write your return statement here")
        return "\n\n".join(lines)


class ExperienceBuffer:
    def __init__(
        self,
        config: config_lib.ExperienceBufferConfig,
    ) -> None:
        self._config = config
        self._islands: list[Island] = []
        for _ in range(config.num_islands):
            self._islands.append(
                Island(
                    config.functions_per_prompt,
                    config.cluster_sampling_temperature_init,
                    config.cluster_sampling_temperature_period,
                )
            )
        self._best_score_per_island: list[float] = (
            [-float('inf')] * config.num_islands
        )
        self._best_program_per_island: list[Optional[str]] = (
            [None] * config.num_islands
        )
        self._best_scores_per_test_per_island: list[Optional[ScoresPerTest]] = (
            [None] * config.num_islands
        )
        self._last_reset_time: float = time.time()

    def get_prompt(self) -> Prompt:
        island_id = np.random.randint(len(self._islands))
        code, version_generated = self._islands[island_id].get_prompt()
        return Prompt(code, version_generated, island_id)

    def _register_program_in_island(
        self,
        program: str,
        island_id: int,
        scores_per_test: ScoresPerTest,
        **kwargs
    ) -> None:
        self._islands[island_id].register_program(program, scores_per_test)
        score = _reduce_score(scores_per_test)
        if score > self._best_score_per_island[island_id]:
            self._best_program_per_island[island_id] = program
            self._best_scores_per_test_per_island[island_id] = scores_per_test
            self._best_score_per_island[island_id] = score
            logging.info('Best score of island %d increased to %s', island_id, score)

        profiler = kwargs.get('profiler')
        if profiler:
            from . import code_manipulation
            func_obj = code_manipulation.Function(
                name='equation',
                args='x, params',
                body=program,
                score=score,
                global_sample_nums=kwargs.get('global_sample_nums'),
                sample_time=kwargs.get('sample_time'),
                evaluate_time=kwargs.get('evaluate_time')
            )
            profiler.register_function(func_obj)

    def register_program(
        self,
        program: str,
        island_id: Optional[int],
        scores_per_test: ScoresPerTest,
        **kwargs
    ) -> None:
        if island_id is None:
            for island_id in range(len(self._islands)):
                self._register_program_in_island(program, island_id, scores_per_test, **kwargs)
        else:
            self._register_program_in_island(program, island_id, scores_per_test, **kwargs)

        if time.time() - self._last_reset_time > self._config.reset_period:
            self._last_reset_time = time.time()
            self.reset_islands()

    def reset_islands(self) -> None:
        indices_sorted_by_score: np.ndarray = np.argsort(
            self._best_score_per_island +
            np.random.randn(len(self._best_score_per_island)) * 1e-6
        )
        num_islands_to_reset = self._config.num_islands // 2
        reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
        keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
        for island_id in reset_islands_ids:
            self._islands[island_id] = Island(
                self._config.functions_per_prompt,
                self._config.cluster_sampling_temperature_init,
                self._config.cluster_sampling_temperature_period,
            )
            self._best_score_per_island[island_id] = -float('inf')
            founder_island_id = np.random.choice(keep_islands_ids)
            founder = self._best_program_per_island[founder_island_id]
            founder_scores = self._best_scores_per_test_per_island[founder_island_id]
            if founder is not None and founder_scores is not None:
                self._register_program_in_island(founder, island_id, founder_scores)