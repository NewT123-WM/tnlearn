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

# drsr/sampler.py (progressive stage-wise guidance)
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Collection, Sequence, Type, Optional, List
import numpy as np
import time
import re

from . import evaluator, buffer, config as config_lib


class LLM(ABC):
    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    @abstractmethod
    def draw_samples(self, prompt: str, config: config_lib.Config, best_score: float = None, progress: float = 0.0, extra_prompt: str = "") -> Collection[str]:
        pass


class LocalLLM(LLM):
    def __init__(self, samples_per_prompt: int, client=None, trim: bool = True) -> None:
        super().__init__(samples_per_prompt)
        self._client = client
        self._trim = trim

    def draw_samples(self, prompt: str, config: config_lib.Config, best_score: float = None, progress: float = 0.0, extra_prompt: str = "") -> Collection[str]:
        enhanced_prompt = self._build_enhanced_prompt(prompt, best_score, progress)
        enhanced_prompt += "\n\nIMPORTANT: When using np.exp, np.sin, np.cos, etc., the argument must be only x or -x, etc. Do NOT put params inside the function argument. For example, use np.exp(x), np.sin(x), np.cos(x), NOT np.exp(params[0]*x)."
        enhanced_prompt += "\nPrefer simpler expressions with fewer terms. A model with less than 3 terms is better than one with many terms if the performance is similar."
        if extra_prompt:
            enhanced_prompt += f"\n\n{extra_prompt}"
        return self._draw_samples_client(enhanced_prompt)

    def _build_enhanced_prompt(self, prompt: str, best_score: float, progress: float) -> str:
        if best_score is None:
            return prompt
        best_mse = -best_score  # score is negative MSE

        # Stage-wise guidance (each prompt independent, no continuity hint)
        if progress < 0.3:
            # Early stage: guide only polynomials
            if best_mse > 0.01:
                hint = "\n\nHint: Try using higher-degree polynomial terms (x**k, k=2,3,4,5, etc.) to better capture curvature."
            else:
                hint = "\n\nHint: Try polynomial forms with less than 3 terms. Keep it simple."
        elif progress < 0.7:
            # Middle stage: if fit is poor, suggest trigonometric/exponential; otherwise keep polynomial
            if best_mse > 0.1:
                hint = "\n\nHint: Polynomial fit is poor. Try using trigonometric (np.sin, np.cos) or exponential (np.exp) functions instead."
            else:
                hint = "\n\nHint: The current polynomial fit is decent. You may try slightly different combinations or add one more term if needed."
        else:
            # Late stage: emphasize simplicity, allow all types but require conciseness
            if best_mse > 0.1:
                hint = "\n\nHint: You can try any function type (polynomial, trig, exp) but keep the expression simple (≤3 terms)."
            else:
                hint = "\n\nHint: The fit is already good. Focus on simplifying the expression (e.g., remove negligible terms)."
        return prompt + hint

    def _draw_samples_client(self, prompt: str) -> List[str]:
        if self._client is None:
            raise RuntimeError("LLM client not provided.")
        all_samples = []
        for _ in range(self._samples_per_prompt):
            try:
                messages = [{"role": "user", "content": prompt}]
                result = self._client.chat(messages)
                content = result.get('content', '')
                body = None
                for line in content.splitlines():
                    stripped = line.strip()
                    if stripped.startswith('return '):
                        body = stripped
                        break
                if body is None:
                    body = "return x * params[0]"
                all_samples.append(body)
            except Exception as e:
                # check if the error message contains keywords indicating an authentication issue
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ['auth', 'api key', 'credential', '401', '403', 'permission']):
                    print(f"Authentication error: {e}")
                    print("Please check your LLM API key configuration.")
                    raise  # raise exception to stop sampling if authentication fails
                else:
                    print(f"LLM error: {e}")
                    all_samples.append("return x * params[0]")
        return all_samples


class Sampler:
    _global_samples_nums: int = 1

    def __init__(
        self,
        database: buffer.ExperienceBuffer,
        evaluators: Sequence[evaluator.Evaluator],
        samples_per_prompt: int,
        config: config_lib.Config,
        max_sample_nums: Optional[int] = None,
        llm_class: Type[LLM] = LocalLLM,
        llm_client=None,
        extra_prompt: str = "",
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        self._llm = llm_class(samples_per_prompt, client=llm_client)
        self._max_sample_nums = max_sample_nums
        self.config = config
        self._extra_prompt = extra_prompt 

    def sample(self, **kwargs):
        start_time = time.time()
        wall_limit = getattr(self.config, 'wall_time_limit_seconds', None)
        while True:
            if wall_limit is not None and (time.time() - start_time) >= wall_limit:
                print(f'Reached wall time limit: {wall_limit} seconds, stopping.')
                break
            if self._max_sample_nums and self.__class__._global_samples_nums >= self._max_sample_nums:
                break

            prompt = self._database.get_prompt()
            # Calculate search progress
            if self._max_sample_nums:
                progress = min(1.0, self.__class__._global_samples_nums / self._max_sample_nums)
            else:
                progress = 0.0  # Default to early stage when no upper limit
            best_score = max(self._database._best_score_per_island) if self._database._best_score_per_island else None
            reset_time = time.time()
            samples = self._llm.draw_samples(prompt.code, self.config, best_score=best_score, progress=progress, extra_prompt=self._extra_prompt)

            # ----- New deduplication logic -----
            unique_samples = []
            seen = set()
            for s in samples:
                if s not in seen:
                    seen.add(s)
                    unique_samples.append(s)
            if len(unique_samples) < len(samples):
                print(f"[Sampler] Deduplicated: {len(samples)} -> {len(unique_samples)}")
            samples = unique_samples
            # ------------------------------------

            sample_time = (time.time() - reset_time) / self._samples_per_prompt

            for sample in samples:
                self._global_sample_nums_plus_one()
                cur_global_sample_nums = self._get_global_sample_nums()
                chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
                chosen_evaluator.analyse(
                    sample,
                    prompt.island_id,
                    prompt.version_generated,
                    **kwargs,
                    global_sample_nums=cur_global_sample_nums,
                    sample_time=sample_time
                )

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        self.__class__._global_samples_nums += 1