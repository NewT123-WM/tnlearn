# Copyright 2024 Meng WANG. All Rights Reserved.
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
"""Top-level DrSR Agent for LLM-based symbolic regression using official evaluator."""

import numpy as np
import re
import os
from typing import Optional, List, Dict, Any, Tuple

from tnlearn.base1 import BaseModel1
from tnlearn.drsr.llm import ClientFactory
from tnlearn.drsr import code_manipulation
from tnlearn.drsr import evaluator
from tnlearn.drsr import buffer as drsr_buffer
from tnlearn.drsr import config as drsr_config
from tnlearn.drsr import prompt_config as pc
from tnlearn.drsr.analyzer import DataAnalyzer

# ---------- 简单经验缓冲类 ----------
class SimpleExperienceBuffer:
    def __init__(self):
        self.experiences = []
        self.lessons = []

    def add_experience(self, exp: str, is_success: bool = True):
        if is_success:
            self.experiences.append(exp)
            if len(self.experiences) > 10:
                self.experiences.pop(0)
        else:
            self.lessons.append(exp)
            if len(self.lessons) > 10:
                self.lessons.pop(0)

    def get_experiences(self, max_count: int = 5) -> List[str]:
        return self.experiences[-max_count:] if self.experiences else []

# ---------- 官方规范模板（与官方 DrSR 相同）----------
SPECIFICATION_TEMPLATE = '''
import numpy as np
from scipy.optimize import minimize

MAX_NPARAMS = 10

@equation.evolve
def equation({args}, params):
    """
    Mathematical function representing the target variable.
    """
    # Write your equation here
    return ...

@evaluate.run
def evaluate_run(inputs, outputs):
    """
    Evaluate the equation on given data.
    """
    def loss(params):
        y_pred = equation(*inputs.T, params)
        return np.mean((y_pred - outputs) ** 2)

    # Multi-start BFGS
    best_loss = np.inf
    best_params = None
    for _ in range(5):
        x0 = np.random.uniform(-1, 1, size=MAX_NPARAMS)
        res = minimize(loss, x0, method='BFGS', options={{'maxiter': 200, 'disp': False}})
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x
    y_pred = equation(*inputs.T, best_params)
    residuals = outputs - y_pred
    score = -best_loss
    result_data = np.column_stack((inputs, outputs, residuals))
    return score, result_data
'''


class LLMSymRegressor(BaseModel1):
    def __init__(
        self,
        llm_config: Optional[Dict[str, Any]] = None,
        max_iterations: int = 20,
        samples_per_iteration: int = 8,
        background: Optional[str] = None,
        random_state: Optional[int] = None,
        verbose: bool = True,
        **kwargs
    ):
        super().__init__()
        self.max_iterations = max_iterations
        self.samples_per_iteration = samples_per_iteration
        self.background = background
        self.random_state = random_state
        self.verbose = verbose

        self._llm_config = llm_config or {}
        self._llm_client = None

        self._template = None
        self._evaluator = None
        self._database = None
        self._inputs_for_eval = None

        self._simple_buffer = SimpleExperienceBuffer()

        self.best_equation_: Optional[str] = None
        self.best_score_: float = -float('inf')
        self.best_params_: Optional[Dict[str, float]] = None
        self.history_: List[Dict[str, Any]] = []

        self._feature_names: Optional[List[str]] = None
        self._target_name: str = 'y'
        self._X_fitted: Optional[np.ndarray] = None
        self._y_fitted: Optional[np.ndarray] = None

    def _setup_official_with_data(self, X: np.ndarray, y: np.ndarray):
        n_features = X.shape[1]
        args_str = ", ".join([f"x{i}" for i in range(n_features)])
        spec = SPECIFICATION_TEMPLATE.format(args=args_str)

        self._template = code_manipulation.text_to_program(spec)
        buffer_config = drsr_config.ExperienceBufferConfig(
            functions_per_prompt=2,
            num_islands=1,
            reset_period=3600,
            cluster_sampling_temperature_init=0.1,
            cluster_sampling_temperature_period=30000
        )
        self._database = drsr_buffer.ExperienceBuffer(buffer_config, self._template, 'equation')
        # 官方 evaluator 期望 inputs 是一个字典，键为字符串（如 'data'），值为数据集字典
        self._inputs_for_eval = {'data': {'inputs': X, 'outputs': y}}
        self._evaluator = evaluator.Evaluator(
            database=self._database,
            template=self._template,
            function_to_evolve='equation',
            function_to_run='evaluate_run',
            inputs=self._inputs_for_eval,
            timeout_seconds=30,
            sandbox_class=evaluator.LocalSandbox
        )
        self._evaluator._sandbox._verbose = self.verbose # 强制设置 sandbox 的 verbose 标志

    def _get_llm_client(self):
        if self._llm_client is None:
            if 'model' not in self._llm_config:
                raise ValueError("llm_config must contain 'model' in format 'provider/model'")
            # 直接使用工厂创建客户端，工厂内部会根据 provider 从环境变量或配置中获取 API Key
            self._llm_client = ClientFactory.from_config(self._llm_config, verbose=self.verbose)
            # 设置生成参数（从 llm_config 或默认值）
            temperature = self._llm_config.get('temperature', 0.6)
            max_tokens = self._llm_config.get('max_tokens', 1024)
            top_p = self._llm_config.get('top_p', 0.3)
            self._llm_client.kwargs.update({
                'temperature': temperature,
                'max_tokens': max_tokens,
                'top_p': top_p,
            })
        return self._llm_client

    def _build_prompt(self, iteration: int) -> str:
        insight = DataAnalyzer.analyze(self._X_fitted, self._y_fitted, self._feature_names)
        insight_text = DataAnalyzer.format_insights_for_prompt(insight)

        experiences = self._simple_buffer.get_experiences()
        exp_text = ""
        if experiences:
            exp_text = pc.ideas_block_title + "\n".join(
                pc.idea_item_prefix.format(index=i+1) + e for i, e in enumerate(experiences[-5:])
            ) + "\n\n"

        problem_desc = f"relation between {', '.join(self._feature_names)} and {self._target_name}"
        head = pc.head_template.format(
            dependent=self._target_name,
            problem=problem_desc,
            independent=", ".join(self._feature_names)
        )
        variables_block = f"Variables: Independents = {', '.join(self._feature_names)}, Dependent = {self._target_name}\n"
        background_text = f"Background: {self.background or 'Unknown physical system'}\n"
        data_insight_block = f"### Data Insight:\n{insight_text}\n"
        instruction = pc.instruction_prompt

        # 提示 LLM 只输出函数体（return 语句），注意需要缩进
        equation_template = f'''# Write the body of the equation function (the part after the colon).
# The function signature is: def equation({', '.join(self._feature_names)}, params):
# Inside the function, write a return statement with proper indentation (4 spaces).
# Example:
#     return {self._feature_names[0]} * 1.0 + 0.5 * np.sin({self._feature_names[1]}) + 0.2
# Provide only the indented function body (no def line, no extra text):
'''
        prompt = (instruction + "\n" + head + "\n" + variables_block + background_text +
                  data_insight_block + exp_text + equation_template)
        return prompt

    def _extract_equation_body_from_response(self, response_text: str) -> Optional[str]:
        """从 LLM 响应中提取函数体（return 语句等），不包含 def 行。"""
        # 先尝试从代码块中提取
        code_pattern = r'```python\n(.*?)\n```'
        matches = re.findall(code_pattern, response_text, re.DOTALL)
        for match in matches:
            # 如果匹配内容包含 def，则尝试提取 return 部分
            if 'def equation' in match:
                lines = match.splitlines()
                for line in lines:
                    if line.strip().startswith('return'):
                        # 保留原始缩进（如果有）
                        return line.rstrip()
                continue
            else:
                # 不包含 def，可能是纯 return 语句
                if match.strip().startswith('return'):
                    return match.strip()
        # 如果没有代码块，直接在文本中查找 return 行
        return_pattern = r'^\s*return\s+[^;]+'
        return_match = re.search(return_pattern, response_text, re.MULTILINE)
        if return_match:
            return return_match.group(0).rstrip()
        return None

    def _create_safe_equation_body(self) -> str:
        """返回一个绝对安全的方程体（线性 y = x0），注意必须带缩进"""
        return f"    return {self._feature_names[0]} * 1.0"


    def _generate_candidates(self, iteration: int) -> List[Tuple[str, str]]:
        prompt = self._build_prompt(iteration)
        candidates = []
        llm = self._get_llm_client()
        for _ in range(self.samples_per_iteration):
            try:
                resp = llm.chat([{"role": "user", "content": prompt}])
                content = resp.get('content', '')
                if self.verbose:
                    print("=" * 60)
                    print("RAW LLM RESPONSE:")
                    print(content)
                    print("=" * 60)

                body = self._extract_equation_body_from_response(content)
                if body is None or not body.strip():
                    if self.verbose:
                        print("No valid equation body found, using safe equation body.")
                    body = self._create_safe_equation_body()
                else:
                    # ========== 改进后的清洗规则 ==========
                    # 1. 将字符串键访问转为 params[0]
                    body = re.sub(r"params\s*\[\s*['\"][^'\"]+['\"]\s*\]", "params[0]", body)
                    # 2. 保留已有的整数索引（不做替换）
                    # 3. 修复错误用法：params 后直接跟运算符但没有索引
                    if re.search(r'params\s*[\*\+\-\/]', body) and not re.search(r'params\s*\[', body):
                        body = re.sub(r'params\b', 'params[0]', body)
                    # 4. 去除不必要的空白和注释（但保留缩进）
                    lines = []
                    for line in body.splitlines():
                        line = line.rstrip()
                        # 移除行内注释（可选），避免 # 号影响
                        # if '#' in line:
                        #     line = line[:line.index('#')]
                        if line.strip():
                            lines.append(line)
                    # 重新组合并统一添加缩进（如果原 body 没有统一缩进）
                    # 先清空现有缩进，再统一加 4 空格
                    stripped_lines = [line.lstrip() for line in lines if line.strip()]
                    if not stripped_lines:
                        body = self._create_safe_equation_body()
                    else:
                        body = '\n'.join(stripped_lines)
                        if not body.startswith('return'):
                            body = f"return {body}"
                        # 添加 4 空格缩进（官方要求）
                        body = '\n'.join('    ' + line for line in body.splitlines())
                    # ========== 清洗结束 ==========

                # 最终安全回退
                if not body or 'return' not in body:
                    body = self._create_safe_equation_body()

                if self.verbose:
                    print("Equation body (first 200 chars):", body[:200])
                candidates.append((body, content))
            except Exception as e:
                if self.verbose:
                    print(f"LLM call failed: {e}")
        return candidates

    def _evaluate_candidate(self, equation_body: str) -> Tuple[float, str, Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            score, error_msg, residual = self._evaluator.analyse(
                sample=equation_body,
                island_id=None,
                version_generated=None,
                profiler=None
            )
            # 获取优化后的参数（从 sandbox 中）
            params = getattr(self._evaluator._sandbox, '_last_params', None)
            if score is None:
                score = -float('inf')
            return score, error_msg, residual, params
        except Exception as e:
            if self.verbose:
                print(f"Evaluation error: {e}")
                print("Problematic body:\n", equation_body[:200])
            return -float('inf'), str(e), None, None
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LLMSymRegressor':
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._X_fitted = X
        self._y_fitted = y
        self._feature_names = [f'x{i}' for i in range(X.shape[1])]

        self._setup_official_with_data(X, y)

        if self.verbose:
            print("Analyzing data...")
            insight = DataAnalyzer.analyze(X, y, self._feature_names)
            print(DataAnalyzer.format_insights_for_prompt(insight))

        for it in range(self.max_iterations):
            if self.verbose:
                print(f"\nIteration {it+1}/{self.max_iterations}")

            candidates = self._generate_candidates(it)
            if not candidates:
                if self.verbose:
                    print("No valid candidates generated")
                continue

            for body, _ in candidates:
                score, error_msg, residual, params = self._evaluate_candidate(body)
                if score is None or not isinstance(score, (int, float)):
                    if self.verbose:
                        print(f"  Score: None (evaluation failed)")
                    continue
                if self.verbose:
                    print(f"  Score: {score:.6f}")

                if score > self.best_score_:
                    self.best_score_ = score
                    self.best_equation_ = body
                    # 提取方程体中实际使用的参数索引
                    used_indices = set()
                    import re
                    for match in re.finditer(r'params\[(\d+)\]', body):
                        used_indices.add(int(match.group(1)))
                    if used_indices:
                        max_idx = max(used_indices)
                        self.best_params_ = params[:max_idx + 1] if params is not None else None
                    else:
                        self.best_params_ = params  # fallback
                    if self.verbose:
                        print(f"    New best equation body:\n{body[:200]}...")

                if score > -1e9:
                    quality = "Good" if score > self.best_score_ * 0.9 else "Bad"
                    exp_summary = f"{quality} sample with score {score:.4f}"
                    self._simple_buffer.add_experience(exp_summary, is_success=(quality=="Good"))

                self.history_.append({
                    'iteration': it,
                    'score': score,
                    'equation': body,
                    'error': error_msg
                })

        if self.verbose:
            print(f"\nDrSR completed. Best score: {self.best_score_:.6f}")
            if self.best_equation_:
                print(f"Best equation body:\n{self.best_equation_}")
            else:
                print("No valid equation found.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.best_equation_ is None:
            raise RuntimeError("Model not fitted yet.")
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # 重建完整函数定义
        args_str = ", ".join(self._feature_names + ['params'])
        full_def = f"def equation({args_str}):\n{self.best_equation_}"
        namespace = {'np': np}
        try:
            exec(full_def, namespace)
            eq_func = namespace['equation']
        except Exception as e:
            raise RuntimeError(f"Failed to compile best equation: {e}")

        args = [X[:, i] for i in range(X.shape[1])]
        
        # 使用训练得到的最佳参数（如果存在），否则回退到零
        if self.best_params_ is not None:
            params = self.best_params_
        else:
            # 尝试从方程体中推断参数个数
            n_params = self.best_equation_.count('params[')
            n_params = max(n_params, 1)
            params = np.zeros(n_params)
            print(f"Warning: best_params_ not available, using zeros (n={n_params})")
        
        try:
            y_pred = eq_func(*args, params)
            if np.isscalar(y_pred):
                y_pred = np.full(X.shape[0], y_pred)
            return y_pred
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")


def discover_neuron_formula(
    X: np.ndarray,
    y: np.ndarray,
    llm_config: Dict[str, Any],
    max_iterations: int = 10,
    samples_per_iteration: int = 4,
    verbose: bool = True,
) -> str:
    reg = LLMSymRegressor(
        llm_config=llm_config,
        max_iterations=max_iterations,
        samples_per_iteration=samples_per_iteration,
        verbose=verbose,
    )
    reg.fit(X, y)
    return reg.best_equation_ or 'x'