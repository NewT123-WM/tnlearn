# Copyright 2024 Meng WANG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""DrSR: Deep Symbolic Regression using LLMs for mathematical equation discovery.

This module provides an LLM-driven symbolic regression agent that combines
LLM's scientific knowledge with data-driven optimization to discover
mathematical equations from data.

Example:
    >>> from tnlearn.drsr import LLMSymRegressor
    >>> reg = LLMSymRegressor(llm_config={'model': 'deepseek/deepseek-chat'})
    >>> reg.fit(X_train, y_train)
    >>> print(reg.best_equation_)
"""

# 导出核心 API (必须)
from .agent import LLMSymRegressor, discover_neuron_formula

# 导出常用辅助工具，方便用户进行自定义分析 (可选)
from .analyzer import DataAnalyzer
from .prompt_config import instruction_prompt, head_template

__all__ = [
    # 核心 API
    'LLMSymRegressor',
    'discover_neuron_formula',
    # 辅助工具
    'DataAnalyzer',
    'instruction_prompt',
    'head_template',
]