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

# drsr/profile.py
# 实验采样/评估的简单记录器（去除 TensorBoard 依赖，仅保留 JSON 与控制台输出）
from __future__ import annotations

import os
from typing import List, Dict, Any, Optional
import logging
import json
from . import llm
from . import code_manipulation


class Profiler:
    def __init__(
            self,
            log_dir: str | None = None,
            pkl_dir: str | None = None,
            max_log_nums: int | None = None,
            samples_per_iteration: int | None = None,
            target_variance: Optional[float] = None,
            persist_all_samples: bool = False,
            wandb_run=None,
    ):
        logging.getLogger().setLevel(logging.INFO)
        self._log_dir = log_dir
        self._json_dir = os.path.join(log_dir, 'samples')
        os.makedirs(self._json_dir, exist_ok=True)
        self._best_history_dir = os.path.join(log_dir, 'best_history')
        os.makedirs(self._best_history_dir, exist_ok=True)
        self._samples_per_iteration = samples_per_iteration or 1
        self._persist_all_samples = bool(persist_all_samples)
        self._target_variance: Optional[float] = target_variance
        self._progress_json_path = os.path.join(log_dir, 'progress.json')
        self._max_log_nums = max_log_nums
        self._num_samples = 0
        self._cur_best_program_sample_order = None
        self._cur_best_program_mse: Optional[float] = None
        self._cur_best_program_nmse: Optional[float] = None
        self._cur_best_program_str = None
        self._evaluate_success_program_num = 0
        self._evaluate_failed_program_num = 0
        self._tot_sample_time = 0
        self._tot_evaluate_time = 0
        self._all_sampled_functions: Dict[int, code_manipulation.Function] = {}
        self._top_k = 10
        self._writer = None
        self._each_sample_best_program_score = []
        self._each_sample_evaluate_success_program_num = []
        self._each_sample_evaluate_failed_program_num = []
        self._each_sample_tot_sample_time = []
        self._each_sample_tot_evaluate_time = []
        self._iteration_progress: Dict[int, Dict[str, Any]] = {}
        self._wandb_run = wandb_run

    def _write_tensorboard(self):
        return

    def _compute_iteration_index(self, sample_order: int | None) -> int | None:
        if sample_order is None:
            return None
        if self._samples_per_iteration <= 0:
            return sample_order
        return int((int(sample_order) - 1) // self._samples_per_iteration) + 1

    def _build_content(self, programs: code_manipulation.Function) -> Dict:
        sample_order = programs.global_sample_nums
        sample_order = sample_order if sample_order is not None else 0
        iteration_idx = self._compute_iteration_index(sample_order)
        function_str = str(programs)
        score = programs.score
        if score is None:
            mse = None
            nmse = None
        else:
            mse = -float(score)
            if self._target_variance is not None and self._target_variance > 0:
                nmse = mse / float(self._target_variance)
            else:
                nmse = None
        params = programs.params
        content = {
            'iteration': iteration_idx,
            'sample_order': sample_order,
            'nmse': nmse,
            'mse': mse,
            'function': function_str,
            'params': params,
        }
        return content

    def _write_json(self, programs: code_manipulation.Function):
        sample_order = programs.global_sample_nums
        sample_order = sample_order if sample_order is not None else 0
        content = self._build_content(programs)
        path = os.path.join(self._json_dir, f'samples_{sample_order}.json')
        with open(path, 'w') as json_file:
            json.dump(content, json_file)

    def _write_topk_json(self):
        try:
            for fname in os.listdir(self._json_dir):
                if fname.startswith('top') and fname.endswith('.json'):
                    try:
                        os.remove(os.path.join(self._json_dir, fname))
                    except Exception:
                        continue
        except Exception:
            pass

        scored_items = []
        for order, func in self._all_sampled_functions.items():
            score = getattr(func, 'score', None)
            if score is None:
                continue
            mse = -float(score)
            if self._target_variance is not None and self._target_variance > 0:
                nmse = mse / float(self._target_variance)
            else:
                nmse = None
            scored_items.append((order, func, mse, nmse))

        if not scored_items:
            return

        def _sort_key(item):
            _, _, mse_val, nmse_val = item
            key_val = nmse_val if nmse_val is not None else mse_val
            return key_val if key_val is not None else float('inf')

        scored_items.sort(key=_sort_key)
        top_items = scored_items[: self._top_k]

        for idx, (order, func, _, _) in enumerate(top_items, start=1):
            prefix = f'top{idx:02d}_'
            content = self._build_content(func)
            filename = f'{prefix}samples_{order}.json'
            path = os.path.join(self._json_dir, filename)
            with open(path, 'w') as json_file:
                json.dump(content, json_file)

    def _save_best_history_sample(self, programs: code_manipulation.Function, sample_orders: int):
        if self._best_history_dir is None:
            return
        content = self._build_content(programs)
        filename = f'best_sample_{sample_orders}.json'
        path = os.path.join(self._best_history_dir, filename)
        with open(path, 'w') as json_file:
            json.dump(content, json_file)

    def _update_iteration_progress(self, sample_orders: int):
        if self._cur_best_program_sample_order is None:
            return
        iteration_idx = self._compute_iteration_index(sample_orders)
        if iteration_idx is None:
            return
        prev_record = self._iteration_progress.get(iteration_idx)
        record = {
            'iteration': iteration_idx,
            'best_nmse': self._cur_best_program_nmse,
            'best_mse': self._cur_best_program_mse,
            'best_sample_order': self._cur_best_program_sample_order,
        }
        try:
            tokens = llm.get_global_tokens()
        except Exception:
            tokens = {}
        try:
            total_time = llm.get_global_time()
        except Exception:
            total_time = None

        record['llm_tokens'] = tokens
        if total_time is not None:
            record['llm_time_seconds'] = round(float(total_time), 2)
        self._iteration_progress[iteration_idx] = record

        if self._wandb_run and record != prev_record:
            try:
                self._wandb_run.log(record, step=iteration_idx)
            except Exception:
                pass

        history = [self._iteration_progress[k] for k in sorted(self._iteration_progress.keys())]
        try:
            with open(self._progress_json_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def register_function(self, programs: code_manipulation.Function):
        if self._max_log_nums is not None and self._num_samples >= self._max_log_nums:
            return

        sample_orders: int = programs.global_sample_nums
        if sample_orders not in self._all_sampled_functions:
            self._num_samples += 1
            self._all_sampled_functions[sample_orders] = programs
            self._record_and_verbose(sample_orders)
            self._write_tensorboard()
            if self._persist_all_samples:
                self._write_json(programs)
            self._write_topk_json()
            self._update_iteration_progress(sample_orders)

    def _record_and_verbose(self, sample_orders: int):
        function = self._all_sampled_functions[sample_orders]
        function_str = str(function).strip('\n')
        sample_time = function.sample_time
        evaluate_time = function.evaluate_time
        score = function.score

        mse = None
        nmse = None
        if score is not None:
            mse = -float(score)
            if self._target_variance is not None and self._target_variance > 0:
                nmse = mse / float(self._target_variance)
            else:
                nmse = None

        print(f'================= Evaluated Function =================')
        print(f'{function_str}')
        print(f'------------------------------------------------------')
        print(f'MSE         : {mse}')
        print(f'NMSE        : {nmse}')
        print(f'Sample time : {str(sample_time)}')
        print(f'Evaluate time: {str(evaluate_time)}')
        print(f'Sample orders: {str(sample_orders)}')
        print(f'======================================================\n\n')

        if nmse is not None:
            if (self._cur_best_program_nmse is None) or (nmse < self._cur_best_program_nmse):
                self._cur_best_program_nmse = nmse
                self._cur_best_program_mse = mse
                self._cur_best_program_sample_order = sample_orders
                self._cur_best_program_str = function_str
                self._save_best_history_sample(function, sample_orders)
        elif mse is not None:
            if (self._cur_best_program_mse is None) or (mse < self._cur_best_program_mse):
                self._cur_best_program_mse = mse
                self._cur_best_program_nmse = None
                self._cur_best_program_sample_order = sample_orders
                self._cur_best_program_str = function_str
                self._save_best_history_sample(function, sample_orders)

        if score:
            self._evaluate_success_program_num += 1
        else:
            self._evaluate_failed_program_num += 1

        if sample_time:
            self._tot_sample_time += sample_time
        if evaluate_time:
            self._tot_evaluate_time += evaluate_time