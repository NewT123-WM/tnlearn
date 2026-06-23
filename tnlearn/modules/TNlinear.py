import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init

__all__ = ['TNLinear']

_EVAL_GLOBALS = {
    'torch': torch,
    'np': __import__('numpy'),
    'math': math,
    'F': F,
}

class TNLinear(nn.Module):
    r"""自定义全连接层，支持符号表达式定义的基函数组合。

    参数:
        in_features (int): 输入特征数。
        out_features (int): 输出特征数。
        symbolic_expression (str): 符号表达式，如 'x + sin(x) + 0.1@x**2'，'x' 代表输入。
        bias (bool): 是否使用偏置，默认为 True。
        device (torch.device, optional): 权重和偏置所在的设备。
        dtype (torch.dtype, optional): 权重和偏置的数据类型。
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 symbolic_expression: str = 'x',
                 bias: bool = True,
                 device=None,
                 dtype=None):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.symbolic_expression = symbolic_expression

        # 解析表达式，提取基函数
        self.terms = self._parse_expression(symbolic_expression)
        if not self.terms:
            self.terms = ['x']    # 回退到线性

        # 预编译基函数为 lambda，提升 forward 效率
        self.funcs = []
        for expr in self.terms:
            try:
                fn = eval('lambda x: ' + expr, _EVAL_GLOBALS)
            except Exception as e:
                print(f"Warning: eval failed for '{expr}', using identity. Error: {e}")
                fn = lambda x: x
            self.funcs.append(fn)

        # 创建权重 (每个基函数一组)
        # 权重形状: (out_features, in_features)
        self.weights = nn.ParameterList()
        for _ in self.terms:
            w = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
            self.weights.append(w)

        # 偏置 (共享一个偏置，与 nn.Linear 一致)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('funcs', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.funcs = []
        for expr in self.terms:
            try:
                fn = eval('lambda x: ' + expr, _EVAL_GLOBALS)
            except Exception as e:
                print(f"Warning: eval failed for '{expr}', using identity. Error: {e}")
                fn = lambda x: x
            self.funcs.append(fn)

    def reset_parameters(self):
        r"""初始化权重和偏置，使用 Kaiming 均匀初始化"""
        for w in self.weights:
            init.kaiming_uniform_(w, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def _split_terms(self, expr: str):
        """按顶层 '+' 和 '-' 分割表达式，保留括号嵌套"""
        terms = []
        current = []
        depth = 0
        for ch in expr:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            if depth == 0 and ch in ('+', '-'):
                terms.append(''.join(current))
                current = [ch]
            else:
                current.append(ch)
        if current:
            terms.append(''.join(current))
        return terms

    def _parse_expression(self, expr: str):
        """提取所有包含 'x' 的子表达式，忽略纯常数项"""
        expr = expr.replace(' ', '')
        terms = self._split_terms(expr)
        x_terms = []
        for t in terms:
            t = t.lstrip('+-')
            if 'x' in t:
                # 若存在 '@'，仅保留 '@' 后的部分（变量表达式）
                if '@' in t:
                    var_expr = t.split('@', 1)[1]
                else:
                    var_expr = t
                x_terms.append(var_expr)
        return x_terms

    def forward(self, x):
        r"""前向传播：对每个基函数变换后分别做线性变换，求和，加偏置"""
        result = 0.0
        for func, weight in zip(self.funcs, self.weights):
            transformed = func(x)          # 应用基函数变换，形状不变
            out = F.linear(transformed, weight, None)   # 不传 bias，自己加
            result = result + out

        if self.bias is not None:
            result = result + self.bias

        return result