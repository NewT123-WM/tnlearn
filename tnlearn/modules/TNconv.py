import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
from typing import Union, Tuple

__all__ = ['TNconv1d', 'TNconv2d', 'TNconv3d', 'TNConvTranspose1d', 'TNConvTranspose2d', 'TNConvTranspose3d']

_EVAL_GLOBALS = {
    'torch': torch,
    'np': __import__('numpy'),
    'math': math,
    'F': F,
}

class _TNConvNd(nn.Module):
    """基类：支持符号表达式的 N 维卷积层"""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[int, Tuple[int, ...]] = 0,
                 symbolic_expression: str = 'x',
                 groups: int = 1,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 padding_mode: str = 'zeros',
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 ndim: int = 2):   # ndim 指定卷积维度 1/2/3
        super().__init__()
        self.ndim = ndim
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.groups = int(groups)

        # 统一转换为元组
        self.kernel_size = self._ntuple(ndim)(kernel_size)
        self.stride = self._ntuple(ndim)(stride)
        self.padding = self._ntuple(ndim)(padding)
        self.dilation = self._ntuple(ndim)(dilation)

        self.padding_mode = padding_mode
        self.symbolic_expression = symbolic_expression

        # 验证 groups
        if in_channels % groups != 0:
            raise ValueError(f'in_channels ({in_channels}) must be divisible by groups ({groups})')

        # 解析表达式
        self.terms = self._parse_expression(symbolic_expression)
        if not self.terms:
            self.terms = ['x']

        # 预编译基函数
        self.funcs = []
        for expr in self.terms:
            try:
                fn = eval('lambda x: ' + expr, _EVAL_GLOBALS)
            except Exception as e:
                print(f"Warning: eval failed for '{expr}', using identity. Error: {e}")
                fn = lambda x: x
            self.funcs.append(fn)

        # 创建权重
        in_ch_per_group = in_channels // groups
        # 权重形状: (out_channels, in_ch_per_group, *kernel_size)
        self.weights = nn.ParameterList()
        for _ in self.terms:
            w = nn.Parameter(torch.empty(out_channels, in_ch_per_group, *self.kernel_size,
                                         device=device, dtype=dtype))
            self.weights.append(w)

        # 偏置
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, device=device, dtype=dtype))
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

    def _ntuple(self, n):
        """返回一个函数，将输入转为长度为 n 的元组"""
        def parse(x):
            if isinstance(x, (int, float)):
                return tuple([int(x)] * n)
            elif isinstance(x, (tuple, list)):
                if len(x) == n:
                    return tuple(int(v) for v in x)
                else:
                    raise ValueError(f'Expected tuple of length {n}, got {len(x)}')
            else:
                raise TypeError(f'Unsupported type: {type(x)}')
        return parse

    def reset_parameters(self):
        for w in self.weights:
            init.kaiming_uniform_(w, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def _split_terms(self, expr: str):
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
        expr = expr.replace(' ', '')
        terms = self._split_terms(expr)
        x_terms = []
        for t in terms:
            t = t.lstrip('+-')
            if 'x' in t:
                if '@' in t:
                    var_expr = t.split('@', 1)[1]
                else:
                    var_expr = t
                x_terms.append(var_expr)
        return x_terms

    def forward(self, x):
        # 处理填充模式（非零填充需手动 pad）
        if self.padding_mode != 'zeros':
            # 根据维度构建 pad 元组：F.pad 顺序为 (left, right, top, bottom, front, back) 等
            # 对于 1D: (pad_w, pad_w)
            # 对于 2D: (pad_w, pad_w, pad_h, pad_h)
            # 对于 3D: (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d)
            pad = []
            for size in reversed(self.padding):   # 从最后一个维度开始
                pad.extend([size, size])
            x = F.pad(x, pad, mode=self.padding_mode)
            conv_padding = tuple([0] * self.ndim)   # 卷积时不额外填充
        else:
            conv_padding = self.padding

        result = 0.0
        for func, weight in zip(self.funcs, self.weights):
            transformed = func(x)
            out = self._conv_forward(transformed, weight, conv_padding)
            result = result + out

        if self.bias is not None:
            # 偏置形状: (out_channels,)，需扩展维度匹配输出
            # 输出形状为 (N, C, *spatial)，偏置加在 C 维度上
            result = result + self.bias.view(1, -1, *([1] * self.ndim))

        return result

    def _conv_forward(self, x, weight, padding):
        """子类重写此方法，调用对应的 F.conv1d/2d/3d"""
        raise NotImplementedError
    

    


# ---------- 子类实现 ----------
class TNConv1d(_TNConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 symbolic_expression='x', groups=1, dilation=1, padding_mode='zeros',
                 bias=True, device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         symbolic_expression, groups, dilation, padding_mode,
                         bias, device, dtype, ndim=1)

    def _conv_forward(self, x, weight, padding):
        return F.conv1d(x, weight, None,
                        stride=self.stride,
                        padding=padding,
                        dilation=self.dilation,
                        groups=self.groups)


class TNConv2d(_TNConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 symbolic_expression='x', groups=1, dilation=1, padding_mode='zeros',
                 bias=True, device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         symbolic_expression, groups, dilation, padding_mode,
                         bias, device, dtype, ndim=2)

    def _conv_forward(self, x, weight, padding):
        return F.conv2d(x, weight, None,
                        stride=self.stride,
                        padding=padding,
                        dilation=self.dilation,
                        groups=self.groups)


class TNConv3d(_TNConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 symbolic_expression='x', groups=1, dilation=1, padding_mode='zeros',
                 bias=True, device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         symbolic_expression, groups, dilation, padding_mode,
                         bias, device, dtype, ndim=3)

    def _conv_forward(self, x, weight, padding):
        return F.conv3d(x, weight, None,
                        stride=self.stride,
                        padding=padding,
                        dilation=self.dilation,
                        groups=self.groups)
    

# ---------- 转置卷积基类 ----------
class _TNConvTransposeNd(nn.Module):
    """基类：支持符号表达式的 N 维转置卷积层"""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[int, Tuple[int, ...]] = 0,
                 output_padding: Union[int, Tuple[int, ...]] = 0,
                 symbolic_expression: str = 'x',
                 groups: int = 1,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 ndim: int = 2):
        super().__init__()
        self.ndim = ndim
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.groups = int(groups)

        # 统一转换为元组
        self.kernel_size = self._ntuple(ndim)(kernel_size)
        self.stride = self._ntuple(ndim)(stride)
        self.padding = self._ntuple(ndim)(padding)
        self.output_padding = self._ntuple(ndim)(output_padding)
        self.dilation = self._ntuple(ndim)(dilation)

        self.symbolic_expression = symbolic_expression

        # 验证 groups
        if in_channels % groups != 0:
            raise ValueError(f'in_channels ({in_channels}) must be divisible by groups ({groups})')
        if out_channels % groups != 0:
            raise ValueError(f'out_channels ({out_channels}) must be divisible by groups ({groups})')

        # 解析表达式
        self.terms = self._parse_expression(symbolic_expression)
        if not self.terms:
            self.terms = ['x']

        # 预编译基函数
        self.funcs = []
        for expr in self.terms:
            try:
                fn = eval('lambda x: ' + expr, _EVAL_GLOBALS)
            except Exception as e:
                print(f"Warning: eval failed for '{expr}', using identity. Error: {e}")
                fn = lambda x: x
            self.funcs.append(fn)

        # 创建权重（转置卷积权重形状：(in_channels, out_channels // groups, *kernel_size)）
        out_ch_per_group = out_channels // groups
        self.weights = nn.ParameterList()
        for _ in self.terms:
            w = nn.Parameter(torch.empty(in_channels, out_ch_per_group, *self.kernel_size,
                                         device=device, dtype=dtype))
            self.weights.append(w)

        # 偏置
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, device=device, dtype=dtype))
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

    def _ntuple(self, n):
        def parse(x):
            if isinstance(x, (int, float)):
                return tuple([int(x)] * n)
            elif isinstance(x, (tuple, list)):
                if len(x) == n:
                    return tuple(int(v) for v in x)
                else:
                    raise ValueError(f'Expected tuple of length {n}, got {len(x)}')
            else:
                raise TypeError(f'Unsupported type: {type(x)}')
        return parse

    def reset_parameters(self):
        for w in self.weights:
            init.kaiming_uniform_(w, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def _split_terms(self, expr: str):
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
        expr = expr.replace(' ', '')
        terms = self._split_terms(expr)
        x_terms = []
        for t in terms:
            t = t.lstrip('+-')
            if 'x' in t:
                if '@' in t:
                    var_expr = t.split('@', 1)[1]
                else:
                    var_expr = t
                x_terms.append(var_expr)
        return x_terms

    def forward(self, x):
        result = 0.0
        for func, weight in zip(self.funcs, self.weights):
            transformed = func(x)
            out = self._conv_forward(transformed, weight)
            result = result + out

        if self.bias is not None:
            result = result + self.bias.view(1, -1, *([1] * self.ndim))

        return result

    def _conv_forward(self, x, weight):
        """子类重写此方法，调用对应的 F.conv_transpose1d/2d/3d"""
        raise NotImplementedError


# ---------- 转置卷积子类 ----------
class TNConvTranspose1d(_TNConvTransposeNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, symbolic_expression='x', groups=1, dilation=1,
                 bias=True, device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         output_padding, symbolic_expression, groups, dilation,
                         bias, device, dtype, ndim=1)

    def _conv_forward(self, x, weight):
        return F.conv_transpose1d(x, weight, None,
                                  stride=self.stride,
                                  padding=self.padding,
                                  output_padding=self.output_padding,
                                  dilation=self.dilation,
                                  groups=self.groups)


class TNConvTranspose2d(_TNConvTransposeNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, symbolic_expression='x', groups=1, dilation=1,
                 bias=True, device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         output_padding, symbolic_expression, groups, dilation,
                         bias, device, dtype, ndim=2)

    def _conv_forward(self, x, weight):
        return F.conv_transpose2d(x, weight, None,
                                  stride=self.stride,
                                  padding=self.padding,
                                  output_padding=self.output_padding,
                                  dilation=self.dilation,
                                  groups=self.groups)


class TNConvTranspose3d(_TNConvTransposeNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, symbolic_expression='x', groups=1, dilation=1,
                 bias=True, device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         output_padding, symbolic_expression, groups, dilation,
                         bias, device, dtype, ndim=3)

    def _conv_forward(self, x, weight):
        return F.conv_transpose3d(x, weight, None,
                                  stride=self.stride,
                                  padding=self.padding,
                                  output_padding=self.output_padding,
                                  dilation=self.dilation,
                                  groups=self.groups)
    

