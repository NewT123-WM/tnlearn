# drsr/sampler.py
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
    def __init__(self, samples_per_prompt: int, client=None, trim: bool = True, mode: str = 'base') -> None:
        super().__init__(samples_per_prompt)
        self._client = client
        self._trim = trim
        self._mode = mode

    def draw_samples(self, prompt: str, config: config_lib.Config, best_score: float = None, progress: float = 0.0, extra_prompt: str = "") -> Collection[str]:
        enhanced_prompt = self._build_enhanced_prompt(prompt, best_score, progress)
        if self._mode == 'base':
            enhanced_prompt += "\n\nIMPORTANT: When using np.exp, np.sin, np.cos, etc., the argument must be only x or -x, etc. Do NOT put params inside the function argument. For example, use np.exp(x), np.sin(x), np.cos(x), NOT np.exp(params[0]*x)."
            enhanced_prompt += "\nPrefer simpler expressions with fewer terms. A model with less than 3 terms is better than one with many terms if the performance is similar."
        else:
            enhanced_prompt += "\n\nPrefer simpler expressions with fewer terms. A model with less than 3 terms is better than one with many terms if the performance is similar."
        if extra_prompt:
            enhanced_prompt += f"\n\n{extra_prompt}"
        return self._draw_samples_client(enhanced_prompt)

    def _build_enhanced_prompt(self, prompt: str, best_score: float, progress: float) -> str:
        if best_score is None:
            return prompt
        best_mse = -best_score

        if self._mode == 'base':
            if progress < 0.3:
                if best_mse > 0.01:
                    hint = "\n\nHint: Try using higher-degree polynomial terms (IP(params[i], x**k), k=2,3,4,5, etc.) to better capture curvature."
                else:
                    hint = "\n\nHint: Try polynomial forms with less than 3 terms. Keep it simple."
            elif progress < 0.7:
                if best_mse > 0.1:
                    hint = "\n\nHint: Polynomial fit is poor. Try using cross term (IP(params[i], x)*IP(params[j], x)*...)."
                else:
                    hint = "\n\nHint: The current polynomial fit is decent. You may try slightly different combinations or add one more term if needed."
            else:
                if best_mse > 0.1:
                    hint = "\n\nHint: You can try any function type (polynomial, trig, exp, cross term) but keep the expression simple (≤3 terms)."
                else:
                    hint = "\n\nHint: The fit is already good. Focus on simplifying the expression (e.g., remove negligible terms)."
        else:  # legacy
            if progress < 0.3:
                if best_mse > 0.01:
                    hint = "\n\nHint: Try using higher-degree polynomial terms (x**k, k=2,3,4,5, etc.) to better capture curvature."
                else:
                    hint = "\n\nHint: Try polynomial forms with less than 3 terms. Keep it simple."
            elif progress < 0.7:
                if best_mse > 0.1:
                    hint = "\n\nHint: Polynomial fit is poor. Try using trigonometric (np.sin, np.cos) or exponential (np.exp) functions instead."
                else:
                    hint = "\n\nHint: The current polynomial fit is decent. You may try slightly different combinations or add one more term if needed."
            else:
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
                lines = content.splitlines()
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped.startswith('return '):
                        rest = stripped[6:].lstrip()
                        if rest.startswith('('):
                            body_lines = [line]
                            open_parens = line.count('(') - line.count(')')
                            j = i + 1
                            while j < len(lines) and open_parens > 0:
                                next_line = lines[j]
                                body_lines.append(next_line)
                                open_parens += next_line.count('(') - next_line.count(')')
                                j += 1
                            body = '\n'.join(body_lines).strip()
                        else:
                            body = stripped
                        break
                if body is None:
                    if self._mode == 'base':
                        body = "return IP(params[0], x)"
                    else:
                        body = "return x * params[0]"
                all_samples.append(body)
            except Exception as e:
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ['auth', 'api key', 'credential', '401', '403', 'permission']):
                    print(f"Authentication error: {e}")
                    print("Please check your LLM API key configuration.")
                    raise
                else:
                    print(f"LLM error: {e}")
                    if self._mode == 'base':
                        all_samples.append("return IP(params[0], x)")
                    else:
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
        mode: str = 'base',      # NEW
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        self._llm = llm_class(samples_per_prompt, client=llm_client, mode=mode)
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
            if self._max_sample_nums:
                progress = min(1.0, self.__class__._global_samples_nums / self._max_sample_nums)
            else:
                progress = 0.0
            best_score = max(self._database._best_score_per_island) if self._database._best_score_per_island else None
            reset_time = time.time()
            samples = self._llm.draw_samples(prompt.code, self.config, best_score=best_score, progress=progress, extra_prompt=self._extra_prompt)

            unique_samples = []
            seen = set()
            for s in samples:
                if s not in seen:
                    seen.add(s)
                    unique_samples.append(s)
            if len(unique_samples) < len(samples):
                print(f"[Sampler] Deduplicated: {len(samples)} -> {len(unique_samples)}")
            samples = unique_samples

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