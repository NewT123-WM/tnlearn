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

# drsr/config.py
from __future__ import annotations
import dataclasses
from typing import Type, Optional

from . import sampler
from . import evaluator


@dataclasses.dataclass(frozen=True)
class ExperienceBufferConfig:
    functions_per_prompt: int = 3
    num_islands: int = 5
    reset_period: int = 4 * 60 * 60
    cluster_sampling_temperature_init: float = 0.8
    cluster_sampling_temperature_period: int = 30000


@dataclasses.dataclass(frozen=True)
class Config:
    experience_buffer: ExperienceBufferConfig = dataclasses.field(default_factory=ExperienceBufferConfig)
    num_samplers: int = 1
    num_evaluators: int = 1
    samples_per_prompt: int = 8
    evaluate_timeout_seconds: int = 30
    wall_time_limit_seconds: Optional[int] = None


@dataclasses.dataclass()
class ClassConfig:
    llm_class: Type[sampler.LLM]
    sandbox_class: Type[evaluator.Sandbox]