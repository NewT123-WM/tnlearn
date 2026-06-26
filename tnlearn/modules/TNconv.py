import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
from typing import Union, Tuple

__all__ = ['TNConv1d', 'TNConv2d', 'TNConv3d', 'TNConvTranspose1d', 'TNConvTranspose2d', 'TNConvTranspose3d']

_EVAL_GLOBALS = {
    'torch': torch,
    'np': __import__('numpy'),
    'math': math,
    'F': F,
}

class _TNConvNd(nn.Module):
    """Base class: N-dimensional convolution layer supporting symbolic expressions."""
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
                 ndim: int = 2):   # ndim specifies convolution dimension 1/2/3
        super().__init__()
        self.ndim = ndim
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.groups = int(groups)

        # Convert to tuples uniformly
        self.kernel_size = self._ntuple(ndim)(kernel_size)
        self.stride = self._ntuple(ndim)(stride)
        self.padding = self._ntuple(ndim)(padding)
        self.dilation = self._ntuple(ndim)(dilation)

        self.padding_mode = padding_mode
        self.symbolic_expression = symbolic_expression

        # Validate groups
        if in_channels % groups != 0:
            raise ValueError(f'in_channels ({in_channels}) must be divisible by groups ({groups})')

        # Parse the expression
        self.terms = self._parse_expression(symbolic_expression)
        if not self.terms:
            self.terms = ['x']

        # Pre-compile basis functions
        self.funcs = []
        for expr in self.terms:
            try:
                fn = eval('lambda x: ' + expr, _EVAL_GLOBALS)
            except Exception as e:
                print(f"Warning: eval failed for '{expr}', using identity. Error: {e}")
                fn = lambda x: x
            self.funcs.append(fn)

        # Create weights
        in_ch_per_group = in_channels // groups
        # Weight shape: (out_channels, in_ch_per_group, *kernel_size)
        self.weights = nn.ParameterList()
        for _ in self.terms:
            w = nn.Parameter(torch.empty(out_channels, in_ch_per_group, *self.kernel_size,
                                         device=device, dtype=dtype))
            self.weights.append(w)

        # Bias
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
        """Return a function that converts input to a tuple of length n."""
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
        # Handle padding modes (non-zero padding requires manual pad)
        if self.padding_mode != 'zeros':
            # Build pad tuple according to dimension: F.pad order is (left, right, top, bottom, front, back) etc.
            # For 1D: (pad_w, pad_w)
            # For 2D: (pad_w, pad_w, pad_h, pad_h)
            # For 3D: (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d)
            pad = []
            for size in reversed(self.padding):   # start from the last dimension
                pad.extend([size, size])
            x = F.pad(x, pad, mode=self.padding_mode)
            conv_padding = tuple([0] * self.ndim)   # no extra padding in convolution
        else:
            conv_padding = self.padding

        # Accumulate outputs from each basis function
        result = None
        for func, weight in zip(self.funcs, self.weights):
            transformed = func(x)
            out = self._conv_forward(transformed, weight, conv_padding)
            if result is None:
                result = out          # direct assignment, no copy
            else:
                result = result + out # in-place addition would be ideal, but PyTorch doesn't support in-place for tensors from different operations; this is fine

        if self.bias is not None:
            # Bias shape: (out_channels,), need to expand dimensions to match output
            # Output shape is (N, C, *spatial), bias is added on the C dimension
            result = result + self.bias.view(1, -1, *([1] * self.ndim))

        return result

    def _conv_forward(self, x, weight, padding):
        """Subclasses override this method to call the corresponding F.conv1d/2d/3d."""
        raise NotImplementedError


# ---------- Subclass implementations ----------
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


# ---------- Transposed convolution base class ----------
class _TNConvTransposeNd(nn.Module):
    """Base class: N-dimensional transposed convolution layer supporting symbolic expressions."""
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

        # Convert to tuples uniformly
        self.kernel_size = self._ntuple(ndim)(kernel_size)
        self.stride = self._ntuple(ndim)(stride)
        self.padding = self._ntuple(ndim)(padding)
        self.output_padding = self._ntuple(ndim)(output_padding)
        self.dilation = self._ntuple(ndim)(dilation)

        self.symbolic_expression = symbolic_expression

        # Validate groups
        if in_channels % groups != 0:
            raise ValueError(f'in_channels ({in_channels}) must be divisible by groups ({groups})')
        if out_channels % groups != 0:
            raise ValueError(f'out_channels ({out_channels}) must be divisible by groups ({groups})')

        # Parse the expression
        self.terms = self._parse_expression(symbolic_expression)
        if not self.terms:
            self.terms = ['x']

        # Pre-compile basis functions
        self.funcs = []
        for expr in self.terms:
            try:
                fn = eval('lambda x: ' + expr, _EVAL_GLOBALS)
            except Exception as e:
                print(f"Warning: eval failed for '{expr}', using identity. Error: {e}")
                fn = lambda x: x
            self.funcs.append(fn)

        # Create weights (transposed conv weight shape: (in_channels, out_channels // groups, *kernel_size))
        out_ch_per_group = out_channels // groups
        self.weights = nn.ParameterList()
        for _ in self.terms:
            w = nn.Parameter(torch.empty(in_channels, out_ch_per_group, *self.kernel_size,
                                         device=device, dtype=dtype))
            self.weights.append(w)

        # Bias
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
        # Accumulate outputs from each basis function
        result = None
        for func, weight in zip(self.funcs, self.weights):
            transformed = func(x)
            out = self._conv_forward(transformed, weight)
            if result is None:
                result = out
            else:
                result = result + out

        if self.bias is not None:
            result = result + self.bias.view(1, -1, *([1] * self.ndim))

        return result

    def _conv_forward(self, x, weight):
        """Subclasses override this method to call the corresponding F.conv_transpose1d/2d/3d."""
        raise NotImplementedError


# ---------- Transposed convolution subclasses ----------
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
