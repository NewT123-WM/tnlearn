# Copyright 2023 DeepMind Technologies Limited
# Modifications Copyright 2024 Meng WANG
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

"""Configuration classes for DrSR components.

Note: This file has been simplified to remove dependency on `sampler` for use
with LLMSymRegressor. Only ExperienceBufferConfig and Config are kept.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class ExperienceBufferConfig:
    """Configures Experience Buffer parameters.

    Args:
        functions_per_prompt: Number of previous hypotheses to include in prompts.
        num_islands: Number of islands in experience buffer for diversity.
        reset_period: Seconds between weakest island resets.
        cluster_sampling_temperature_init: Initial cluster softmax sampling temperature.
        cluster_sampling_temperature_period: Period for temperature decay.
    """
    functions_per_prompt: int = 2
    num_islands: int = 10
    reset_period: int = 4 * 60 * 60
    cluster_sampling_temperature_init: float = 0.1
    cluster_sampling_temperature_period: int = 30_000


@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for DrSR experiments (sampler settings).

    Args:
        experience_buffer: Evolution multi-population settings.
        num_samplers: Number of parallel samplers.
        num_evaluators: Number of parallel evaluators.
        samples_per_prompt: Number of hypotheses per prompt.
        evaluate_timeout_seconds: Hypothesis evaluation timeout.
        results_root: Root directory for results.
        wall_time_limit_seconds: Total experiment time limit (seconds).
    """
    experience_buffer: ExperienceBufferConfig = dataclasses.field(default_factory=ExperienceBufferConfig)
    num_samplers: int = 1
    num_evaluators: int = 1
    samples_per_prompt: int = 4
    evaluate_timeout_seconds: int = 30
    results_root: str | None = None
    wall_time_limit_seconds: int | None = None