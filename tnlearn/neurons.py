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

import math
import re
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F

# 安全全局命名空间（仅用于初始化时编译表达式）
_EVAL_GLOBALS = {
    'torch': torch,
    'math': math,
    'F': F,
    # 可根据需要扩展其他常用库，如 'np': np
}


class CustomNeuronLayer(nn.Module):
    r"""Build a neural network model of a custom architecture (GPU‑efficient version)."""

    def __init__(self, in_features: int, out_features: int, symbolic_expression: str, bias: bool = True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.neuron = symbolic_expression

        # ---------- 一次性解析表达式 ----------
        self.expr_strings = self._extract_var_exprs(symbolic_expression)
        self.num_terms = len(self.expr_strings)

        # 预编译每个子表达式为可调用函数（仅执行一次 eval）
        self.funcs = []
        for expr_str in self.expr_strings:
            try:
                # 编译为 lambda 函数：输入 x，返回表达式计算结果
                func = eval(f"lambda x: {expr_str}", _EVAL_GLOBALS, {})
                self.funcs.append(func)
            except Exception as e:
                # 若表达式不合法，则占位为一个返回零张量的函数
                print(f"Warning: Failed to compile expression '{expr_str}': {e}. Using zero placeholder.")
                self.funcs.append(lambda x: torch.zeros_like(x))

        # ---------- 权重参数列表 ----------
        self.weights = nn.ParameterList([
            Parameter(torch.Tensor(out_features, in_features)) for _ in range(self.num_terms)
        ])

        # ---------- 偏置 ----------
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def _extract_var_exprs(self, expr: str) -> list:
        """
        从原始符号表达式中提取所有 '@' 之后的变量表达式（丢弃系数）。
        例如 "2@x**3 + 3@torch.sin(x) + x" -> ['x**3', 'torch.sin(x)', 'x']
        """
        # 去除空格
        expr = expr.replace(' ', '')
        # 按顶层 '+' 或 '-' 分割（注意保留括号内的符号）
        terms = self._split_terms(expr)
        var_exprs = []
        for term in terms:
            # 去掉开头的 '+' 或 '-'
            term = term.lstrip('+-')
            if '@' in term:
                # 取 '@' 之后的部分
                var_part = term.split('@', 1)[1]
                var_exprs.append(var_part)
            elif 'x' in term:
                # 没有 '@' 表示系数为 1（隐含），整个 term 就是变量表达式
                var_exprs.append(term)
            # 常数项（无 'x'）直接忽略
        return var_exprs

    def _split_terms(self, expr: str):
        """按顶层加减号分割表达式，正确保留括号内的符号。"""
        terms = []
        current = []
        paren_depth = 0
        for ch in expr:
            if ch == '(':
                paren_depth += 1
            elif ch == ')':
                paren_depth -= 1
            if paren_depth == 0 and ch in ('+', '-'):
                terms.append(''.join(current))
                current = [ch]
            else:
                current.append(ch)
        if current:
            terms.append(''.join(current))
        return terms

    def reset_parameters(self) -> None:
        """Kaiming 初始化权重，均匀初始化偏置。"""
        for w in self.weights:
            init.kaiming_uniform_(w, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        前向传播：对每个编译好的子表达式计算张量，再分别做线性变换，最后求和加偏置。
        """
        result = 0.0
        for func, weight in zip(self.funcs, self.weights):
            # 调用预编译函数，得到张量值
            value = func(x)
            # 线性变换（不额外加偏置，因为总偏置统一添加）
            result += F.linear(value, weight, None)

        if self.bias is not None:
            result += self.bias
        return result