"""轻量化实验记录：保留样本 JSON 输出，移除 TensorBoard 依赖。

Profiler 现在直接接收 results_root（实验根目录），
在 results_root/samples 下输出每个样本的 JSON 文件。
"""

from __future__ import annotations

import os.path
from typing import List, Dict
import logging
import json
from . import code_manipulation
# 移除对 TensorBoard 的依赖，避免安装额外包


class Profiler:
    def __init__(
        self,
        results_root: str | None = None,
        pkl_dir: str | None = None,
        max_log_nums: int | None = None,
        samples_per_iteration: int | None = None,
    ):
        """
        Args:
            results_root: 实验根目录（samples JSON 将保存在此目录下的 samples 子目录）。
            pkl_dir     : save the results to a pkl file.
            max_log_nums: stop logging if exceeding max_log_nums.
        """
        logging.getLogger().setLevel(logging.INFO)
        self._results_root = results_root or '.'
        # samples 输出目录：results_root/samples
        self._json_dir = os.path.join(self._results_root, 'samples')
        os.makedirs(self._json_dir, exist_ok=True)
        # 进度与历史最优目录/文件
        self._progress_path = os.path.join(self._results_root, 'progress.json')
        self._best_history_dir = os.path.join(self._results_root, 'best_history')
        os.makedirs(self._best_history_dir, exist_ok=True)

        self._max_log_nums = max_log_nums
        self._num_samples = 0
        self._cur_best_program_sample_order = None
        self._cur_best_program_score = -99999999
        self._cur_best_program_str = None
        self._evaluate_success_program_num = 0
        self._evaluate_failed_program_num = 0
        self._tot_sample_time = 0
        self._tot_evaluate_time = 0
        self._all_sampled_functions: Dict[int, code_manipulation.Function] = {}
        # 仅保留 samples 目录下分数最高的前 K 个样本 JSON
        self._keep_top_k_samples: int = 10

        # iteration 与全局最佳跟踪
        self._samples_per_iteration: int | None = (
            int(samples_per_iteration) if samples_per_iteration and samples_per_iteration > 0 else None
        )
        self._progress_records: list[dict] = []
        self._global_best_score = None
        self._global_best_sample_order = None

        # 不再创建 TensorBoard 写入器
        self._writer = None

        self._each_sample_best_program_score = []
        self._each_sample_evaluate_success_program_num = []
        self._each_sample_evaluate_failed_program_num = []
        self._each_sample_tot_sample_time = []
        self._each_sample_tot_evaluate_time = []

    def _write_tensorboard(self):
        """兼容旧接口：不再写入 TensorBoard。"""
        return

    def _write_json(self, programs: code_manipulation.Function):
        """为当前样本写入单独 JSON，并刷新 Top-K 文件。"""
        # 1) 写入单个样本文件：samples/samples_<sample_order>.json
        sample_order = programs.global_sample_nums or 0
        score = programs.score
        iteration = self._compute_iteration(int(sample_order))
        function_str = str(programs)

        # 字段顺序：iteration -> sample_order -> score -> function -> params
        content = {
            "iteration": iteration,
            "sample_order": int(sample_order),
            "score": score,
            "function": function_str,
        }
        try:
            if getattr(programs, "optimized_params", None) is not None:
                content["params"] = list(programs.optimized_params)
        except Exception:
            pass

        path = os.path.join(self._json_dir, f"samples_{sample_order}.json")
        try:
            with open(path, "w", encoding="utf-8") as json_file:
                json.dump(content, json_file, ensure_ascii=False, indent=2)
        except Exception:
            # 单个样本写失败不影响主流程
            pass

        # 2) 基于所有样本重新计算 Top-K，并写入 topXX_ 前缀文件
        try:
            self._prune_samples_dir_topk()
        except Exception:
            # 精简/重写失败不影响主流程
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
            self._write_json(programs)
            # 在写入 samples 之后，更新进度与历史最优记录
            try:
                self._update_progress_and_history(programs)
            except Exception:
                # 进度记录失败不影响主流程
                pass

    def _compute_iteration(self, sample_order: int) -> int:
        """
        根据全局 sample_order 估算 iteration 编号。
        若未提供 samples_per_iteration，则统一视为第 1 轮。
        """
        if not isinstance(sample_order, int) or sample_order <= 0:
            return 1
        if not self._samples_per_iteration or self._samples_per_iteration <= 0:
            return 1
        # 按 samples_per_iteration 分组，1-based
        return (sample_order - 1) // self._samples_per_iteration + 1

    def _update_progress_and_history(self, programs: code_manipulation.Function):
        """
        更新 progress.json 与 best_history 目录：
        - progress.json：每个 iteration 的全局最佳记录
        - best_history：每次全局最优被刷新时，保存对应 sample 的完整信息
        """
        sample_order: int = programs.global_sample_nums or 0
        score = programs.score
        if not isinstance(score, (int, float)):
            return

        iteration = self._compute_iteration(sample_order)

        # 1. 若是全局最优被刷新，则追加一条历史最优样本
        is_new_global_best = (
            self._global_best_score is None or score > self._global_best_score
        )
        if is_new_global_best:
            self._global_best_score = float(score)
            self._global_best_sample_order = int(sample_order)

            # 写入 best_history/best_sample_<sample_order>.json
            function_str = str(programs)
            content = {
                "iteration": iteration,
                "sample_order": self._global_best_sample_order,
                "score": self._global_best_score,
                "function": function_str,
            }
            try:
                if getattr(programs, "optimized_params", None) is not None:
                    content["params"] = list(programs.optimized_params)
            except Exception:
                pass

            best_path = os.path.join(
                self._best_history_dir,
                f"best_sample_{self._global_best_sample_order}.json",
            )
            try:
                with open(best_path, "w", encoding="utf-8") as f:
                    json.dump(content, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        # 2. 更新 progress.json：确保 iteration 条目存在，并写入“截至该 iteration 的全局最优”
        # 扩展进度列表到当前 iteration
        while len(self._progress_records) < iteration:
            next_iter = len(self._progress_records) + 1
            # 默认沿用当前全局最优（可能为 None）
            self._progress_records.append(
                {
                    "iteration": next_iter,
                    "best_score": self._global_best_score,
                    "best_sample_order": self._global_best_sample_order,
                }
            )

        # 当前 iteration 的记录更新为最新的全局最优
        self._progress_records[iteration - 1]["best_score"] = self._global_best_score
        self._progress_records[iteration - 1]["best_sample_order"] = (
            self._global_best_sample_order
        )

        try:
            with open(self._progress_path, "w", encoding="utf-8") as f:
                json.dump(self._progress_records, f, ensure_ascii=False, indent=2)
        except Exception:
            # 写进度失败也不终止主流程
            pass

    def _record_and_verbose(self, sample_orders: int):
        function = self._all_sampled_functions[sample_orders]
        function_str = str(function).strip('\n')
        sample_time = function.sample_time
        evaluate_time = function.evaluate_time
        score = function.score
        # log attributes of the function
        print(f'================= Evaluated Function =================')
        print(f'{function_str}')
        print(f'------------------------------------------------------')
        print(f'Score        : {str(score)}')
        print(f'Sample time  : {str(sample_time)}')
        print(f'Evaluate time: {str(evaluate_time)}')
        print(f'Sample orders: {str(sample_orders)}')
        print(f'======================================================\n\n')

        # update best function in curve
        if function.score is not None and score > self._cur_best_program_score:
            self._cur_best_program_score = score
            self._cur_best_program_sample_order = sample_orders
            self._cur_best_program_str = function_str

        # update statistics about function
        if score:
            self._evaluate_success_program_num += 1
        else:
            self._evaluate_failed_program_num += 1

        if sample_time:
            self._tot_sample_time += sample_time
        if evaluate_time:
            self._tot_evaluate_time += evaluate_time

    def _prune_samples_dir_topk(self):
        """
        仅保留分数最高的前 K 个样本，并按排名重写文件：
        - 文件名形如：top01_samples_{sample_order}.json
        - JSON 字段顺序：iteration -> sample_order -> score -> function -> params
        """
        # 1. 收集所有已有样本的分数
        entries = []
        for order, func in self._all_sampled_functions.items():
            try:
                s = getattr(func, 'score', None)
                if isinstance(s, (int, float)):
                    entries.append((order, float(s), func))
            except Exception:
                continue
        if not entries:
            return

        # 2. 按分数从高到低排序，取前 K 个
        entries.sort(key=lambda x: x[1], reverse=True)
        top_k = entries[: max(1, int(self._keep_top_k_samples))]

        # 3. 清空 samples 目录下旧的 Top-K JSON 文件（保留 samples_*.json）
        try:
            for name in os.listdir(self._json_dir):
                if not name.endswith('.json'):
                    continue
                # 只删除 top 前缀的文件，保留 samples_*.json
                if not name.startswith("top"):
                    continue
                try:
                    os.remove(os.path.join(self._json_dir, name))
                except Exception:
                    pass
        except Exception:
            pass

        # 4. 重新按排名写入 topXX_ 前缀的文件
        for rank, (order, score, func) in enumerate(top_k, start=1):
            sample_order = order if order is not None else 0
            function_str = str(func)

            iteration = self._compute_iteration(int(sample_order))

            # 按用户需求的字段顺序组织内容：
            # iteration -> sample_order -> score -> function -> params
            content = {
                "iteration": iteration,
                "sample_order": sample_order,
                "score": score,
                "function": function_str,
            }
            # 如果存在优化参数，则追加 params
            try:
                if getattr(func, 'optimized_params', None) is not None:
                    content["params"] = list(func.optimized_params)
            except Exception:
                pass

            file_name = f"top{rank:02d}_samples_{sample_order}.json"
            path = os.path.join(self._json_dir, file_name)
            try:
                with open(path, "w", encoding="utf-8") as json_file:
                    json.dump(content, json_file, ensure_ascii=False, indent=2)
            except Exception:
                # 单个样本写失败不影响其它样本
                continue
