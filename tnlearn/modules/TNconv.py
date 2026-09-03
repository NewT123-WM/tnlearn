"""
Convolutional layers with symbolic aggregation supporting inner‑product cross terms.

This module provides 1D, 2D, 3D convolution and transposed convolution layers where
the aggregation function is defined by a symbolic expression.

Two modes are supported:
- base (default): Uses SymPy parameterization with InnerProduct support.
- legacy: Uses simple eval-based expression parsing (original implementation).

Copyright (c) 2026 Tieyun LI. All Rights Reserved.
"""

import math
import re
from typing import Union, Tuple, Optional, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Parameter
from sympy import symbols, Add, Mul, Pow, sin, cos, exp, simplify, expand, sympify, Symbol, Number

from tnlearn.operator.inner_product import (
    InnerProduct,
    convert_pretty_to_innerproduct,
    is_inner_product,
    replace_numbers,
)

__all__ = [
    'TNConv1d', 'TNConv2d', 'TNConv3d',
    'TNConvTranspose1d', 'TNConvTranspose2d', 'TNConvTranspose3d'
]

# ---------- Global eval environment for legacy mode ----------
_EVAL_GLOBALS = {
    'torch': torch,
    'np': __import__('numpy'),
    'math': math,
    'F': F,
}

# ---------- Internal helper: base mode parameterization ----------
def _process_inner_product(ip, counter):
    left = ip.left
    right = ip.right
    new_left = replace_numbers(left, counter, False)
    new_right = expand(simplify(right))
    return InnerProduct(new_left, new_right)

def parameterize_term_conv(term, counter):
    if term.is_number:
        return 0
    if is_inner_product(term):
        new_ip = _process_inner_product(term, counter)
        if term.left.is_Number:
            return new_ip
        else:
            c = symbols('c%d' % counter['c'])
            counter['c'] += 1
            return Mul(c, new_ip)
    if isinstance(term, Mul):
        inner_products = []
        non_inner = []
        created_param = False
        for factor in term.args:
            if factor.is_number:
                continue
            elif is_inner_product(factor):
                new_ip = _process_inner_product(factor, counter)
                if factor.left.is_Number:
                    created_param = True
                inner_products.append(new_ip)
            else:
                non_inner.append(factor)
        if non_inner:
            merged = Mul(*non_inner)
            old_w = counter['w']
            merged = replace_numbers(merged, counter, False)
            if counter['w'] > old_w:
                created_param = True
            w = symbols('w%d' % counter['w'])
            counter['w'] += 1
            created_param = True
            inner_products.append(InnerProduct(w, merged))
        if not inner_products:
            return 1
        result = Mul(*inner_products) if len(inner_products) > 1 else inner_products[0]
        if not created_param:
            c = symbols('c%d' % counter['c'])
            counter['c'] += 1
            result = Mul(c, result)
        return result
    new_expr = replace_numbers(term, counter, False)
    w = symbols('w%d' % counter['w'])
    counter['w'] += 1
    return InnerProduct(w, new_expr)

def parameterize_expression_conv(expr):
    if isinstance(expr, Add):
        terms = list(expr.args)
    else:
        terms = [expr]
    counter = {'w': 1, 'b': 1, 'c': 1}
    new_terms = []
    for t in terms:
        pt = parameterize_term_conv(t, counter)
        if pt != 0:
            new_terms.append(pt)
    if not new_terms:
        return 0
    return Add(*new_terms) if len(new_terms) > 1 else new_terms[0]

def _check_inner_product_legality(expr, x_sym):
    if is_inner_product(expr):
        left, right = expr.args
        if not (isinstance(left, Symbol) and str(left).startswith('w')):
            raise ValueError(f"InnerProduct left argument must be a weight symbol (w_i), got {left}")
        if not right.has(x_sym):
            raise ValueError(f"InnerProduct right argument must depend on input 'x', got {right}")
        for sym in right.free_symbols:
            if str(sym).startswith('w'):
                raise ValueError(f"InnerProduct right argument contains weight symbol {sym}, which is not allowed.")
        _check_inner_product_legality(right, x_sym)
    elif isinstance(expr, (Add, Mul, Pow)):
        for arg in expr.args:
            _check_inner_product_legality(arg, x_sym)
    elif hasattr(expr, 'args') and not isinstance(expr, (Symbol, Number)):
        for arg in expr.args:
            _check_inner_product_legality(arg, x_sym)

# ---------- Legacy mode helpers ----------
def _split_terms(expr: str):
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

def _parse_expression(expr: str):
    expr = expr.replace(' ', '')
    terms = _split_terms(expr)
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

# ---------- Base class for N‑D convolution ----------
class _TNConvNd(nn.Module):
    """
    Base class for N‑dimensional convolution with symbolic aggregation.

    Args:
        in_channels, out_channels, kernel_size, stride, padding, groups, dilation,
        padding_mode, bias, device, dtype, ndim, is_transposed, output_padding
        mode (str): 'base' (SymPy+InnerProduct) or 'legacy' (eval-based)
        symbolic_expression (str): expression defining aggregation.
    """
    __constants__ = ['in_channels', 'out_channels', 'groups', 'kernel_size',
                     'stride', 'padding', 'dilation', 'padding_mode',
                     'output_padding', 'symbolic_expression', 'ndim', 'is_transposed', 'mode']

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[int, Tuple[int, ...], str] = 0,
                 symbolic_expression: str = 'x',
                 groups: int = 1,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] = 'zeros',
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 ndim: int = 2,
                 is_transposed: bool = False,
                 output_padding: Union[int, Tuple[int, ...]] = 0,
                 mode: str = 'base'):
        super().__init__()
        self.ndim = ndim
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.groups = int(groups)
        self.is_transposed = is_transposed
        self.mode = mode.lower()
        if self.mode not in ('base', 'legacy'):
            raise ValueError("mode must be 'base' or 'legacy'")

        ntuple = self._ntuple(ndim)
        self.kernel_size = ntuple(kernel_size)
        self.stride = ntuple(stride)
        self.padding = ntuple(padding) if not isinstance(padding, str) else padding
        self.dilation = ntuple(dilation)
        self.output_padding = ntuple(output_padding) if is_transposed else None
        self.padding_mode = padding_mode
        self.symbolic_expression = symbolic_expression

        if in_channels % groups != 0:
            raise ValueError(f'in_channels ({in_channels}) must be divisible by groups ({groups})')
        if is_transposed and out_channels % groups != 0:
            raise ValueError(f'out_channels ({out_channels}) must be divisible by groups ({groups})')

        # Determine weight shape
        if is_transposed:
            out_ch_per_group = out_channels // groups
            w_shape = (in_channels, out_ch_per_group, *self.kernel_size)
        else:
            in_ch_per_group = in_channels // groups
            w_shape = (out_channels, in_ch_per_group, *self.kernel_size)

        if self.mode == 'base':
            # ---------- Base mode (SymPy) ----------
            converted_str = convert_pretty_to_innerproduct(symbolic_expression)
            raw_expr = sympify(converted_str, locals={'InnerProduct': InnerProduct})
            raw_expr = expand(simplify(raw_expr))
            self.param_expr = parameterize_expression_conv(raw_expr)

            x_sym = symbols('x')
            all_symbols = self.param_expr.free_symbols if self.param_expr != 0 else set()
            weight_symbols = [sym for sym in all_symbols if sym != x_sym]
            w_syms = [str(sym) for sym in weight_symbols if str(sym).startswith('w')]
            c_syms = [str(sym) for sym in weight_symbols if str(sym).startswith('c')]

            if self.param_expr != 0:
                _check_inner_product_legality(self.param_expr, x_sym)

            self.weights = nn.ParameterDict()
            for sym in w_syms:
                self.weights[str(sym)] = nn.Parameter(torch.empty(w_shape, device=device, dtype=dtype))

            spatial_shape = (1,) * ndim
            self.coeffs = nn.ParameterDict()
            for sym in c_syms:
                self.coeffs[str(sym)] = nn.Parameter(torch.empty(out_channels, *spatial_shape, device=device, dtype=dtype))

            self._x_sym = x_sym
            self._base_initialized = True
            self._legacy_initialized = False
        else:
            # ---------- Legacy mode (eval) ----------
            self.terms = _parse_expression(symbolic_expression)
            if not self.terms:
                self.terms = ['x']

            self.funcs = []
            for expr in self.terms:
                try:
                    fn = eval('lambda x: ' + expr, _EVAL_GLOBALS)
                except Exception as e:
                    print(f"Legacy: eval failed for '{expr}', using identity. Error: {e}")
                    fn = lambda x: x
                self.funcs.append(fn)

            self.weights = nn.ParameterList()
            for _ in self.terms:
                self.weights.append(nn.Parameter(torch.empty(w_shape, device=device, dtype=dtype)))

            self._base_initialized = False
            self._legacy_initialized = True

        # Bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

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
        if self.mode == 'base':
            for w in self.weights.values():
                init.kaiming_uniform_(w, a=math.sqrt(5))
            for c in self.coeffs.values():
                fan_in = self.in_channels
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(c, -bound, bound)
        else:  # legacy
            for w in self.weights:
                init.kaiming_uniform_(w, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        s = f'in_channels={self.in_channels}, out_channels={self.out_channels}'
        s += f', kernel_size={self.kernel_size}'
        if self.stride != (1,) * self.ndim:
            s += f', stride={self.stride}'
        if self.padding != (0,) * self.ndim and not isinstance(self.padding, str):
            s += f', padding={self.padding}'
        elif isinstance(self.padding, str):
            s += f', padding="{self.padding}"'
        if self.dilation != (1,) * self.ndim:
            s += f', dilation={self.dilation}'
        if self.is_transposed and self.output_padding != (0,) * self.ndim:
            s += f', output_padding={self.output_padding}'
        if self.groups != 1:
            s += f', groups={self.groups}'
        if self.padding_mode != 'zeros':
            s += f', padding_mode={self.padding_mode}'
        if self.bias is None:
            s += ', bias=False'
        if self.symbolic_expression != 'x':
            s += f', symbolic_expression="{self.symbolic_expression}"'
        if self.mode != 'base':
            s += f', mode="{self.mode}"'
        return s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # -------- Padding handling (same for both modes) --------
        if self.padding_mode != 'zeros':
            pad = []
            if isinstance(self.padding, str):
                if self.padding == 'valid':
                    pad = [0] * (2 * self.ndim)
                elif self.padding == 'same':
                    for d, k, s, dil in zip(range(self.ndim), self.kernel_size,
                                            self.stride, self.dilation):
                        total_pad = dil * (k - 1)
                        left = total_pad // 2
                        right = total_pad - left
                        pad = [right, left] + pad
                else:
                    raise ValueError(f"Unsupported padding string: {self.padding}")
            else:
                for size in reversed(self.padding):
                    pad.extend([size, size])
            x_padded = F.pad(x, pad, mode=self.padding_mode)
            conv_padding = tuple([0] * self.ndim)
        else:
            x_padded = x
            conv_padding = self.padding if not isinstance(self.padding, str) else (0,) * self.ndim

        # -------- Choose convolution function --------
        if self.is_transposed:
            conv_fn = getattr(F, f'conv_transpose{self.ndim}d')
            conv_kwargs = {
                'stride': self.stride,
                'padding': conv_padding,
                'output_padding': self.output_padding,
                'dilation': self.dilation,
                'groups': self.groups,
            }
        else:
            conv_fn = getattr(F, f'conv{self.ndim}d')
            conv_kwargs = {
                'stride': self.stride,
                'padding': conv_padding,
                'dilation': self.dilation,
                'groups': self.groups,
            }

        # -------- Mode-specific forward --------
        if self.mode == 'base':
            def conv_wrapper(inp, weight):
                return conv_fn(inp, weight, **conv_kwargs)

            subs = {}
            for name, param in self.weights.items():
                subs[symbols(name)] = param
            for name, param in self.coeffs.items():
                subs[symbols(name)] = param

            result = self._eval_conv_expr(self.param_expr, x_padded, subs, conv_wrapper, self.is_transposed)
        else:  # legacy
            result = None
            for func, weight in zip(self.funcs, self.weights):
                transformed = func(x_padded)
                out = conv_fn(transformed, weight, **conv_kwargs)
                if result is None:
                    result = out
                else:
                    result = result + out

        if self.bias is not None:
            result = result + self.bias.view(1, -1, *([1] * self.ndim))

        return result

    def _eval_conv_expr(self, expr, x_tensor, subs, conv_fn, is_transposed=False):
        """Base mode evaluation (recursive SymPy evaluation)."""
        if expr.is_Number:
            return torch.tensor(float(expr), device=x_tensor.device).reshape(1, 1, *([1]*self.ndim))

        if expr.is_Symbol:
            if expr == self._x_sym:
                return x_tensor
            else:
                return subs[expr]

        if expr.is_Add:
            result = self._eval_conv_expr(expr.args[0], x_tensor, subs, conv_fn, is_transposed)
            for arg in expr.args[1:]:
                result = result + self._eval_conv_expr(arg, x_tensor, subs, conv_fn, is_transposed)
            return result

        if expr.is_Mul:
            result = self._eval_conv_expr(expr.args[0], x_tensor, subs, conv_fn, is_transposed)
            for arg in expr.args[1:]:
                result = result * self._eval_conv_expr(arg, x_tensor, subs, conv_fn, is_transposed)
            return result

        if expr.is_Pow:
            base = self._eval_conv_expr(expr.base, x_tensor, subs, conv_fn, is_transposed)
            exp = self._eval_conv_expr(expr.exp, x_tensor, subs, conv_fn, is_transposed)
            return base ** exp

        if expr.func.__name__ == 'sin':
            return torch.sin(self._eval_conv_expr(expr.args[0], x_tensor, subs, conv_fn, is_transposed))
        if expr.func.__name__ == 'cos':
            return torch.cos(self._eval_conv_expr(expr.args[0], x_tensor, subs, conv_fn, is_transposed))
        if expr.func.__name__ == 'exp':
            return torch.exp(self._eval_conv_expr(expr.args[0], x_tensor, subs, conv_fn, is_transposed))

        if is_inner_product(expr):
            left = expr.left
            right = expr.right
            left_val = self._eval_conv_expr(left, x_tensor, subs, conv_fn, is_transposed)
            right_val = self._eval_conv_expr(right, x_tensor, subs, conv_fn, is_transposed)

            if isinstance(left_val, Parameter) and left_val.dim() == self.ndim + 2:
                return conv_fn(right_val, left_val)
            else:
                prod = left_val * right_val
                dims_to_sum = list(range(1, prod.dim()))
                return prod.sum(dim=dims_to_sum, keepdim=True)

        raise NotImplementedError(f"Unsupported expression node: {expr}")


# ========== Concrete convolution classes ==========
class TNConv1d(_TNConvNd):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int]],
                 stride: Union[int, Tuple[int]] = 1,
                 padding: Union[int, Tuple[int], str] = 0,
                 symbolic_expression: str = 'x',
                 groups: int = 1,
                 dilation: Union[int, Tuple[int]] = 1,
                 padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] = 'zeros',
                 bias: bool = True,
                 device=None, dtype=None,
                 mode: str = 'base'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         symbolic_expression, groups, dilation, padding_mode,
                         bias, device, dtype, ndim=1, is_transposed=False,
                         output_padding=0, mode=mode)


class TNConv2d(_TNConvNd):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int], str] = 0,
                 symbolic_expression: str = 'x',
                 groups: int = 1,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] = 'zeros',
                 bias: bool = True,
                 device=None, dtype=None,
                 mode: str = 'base'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         symbolic_expression, groups, dilation, padding_mode,
                         bias, device, dtype, ndim=2, is_transposed=False,
                         output_padding=0, mode=mode)


class TNConv3d(_TNConvNd):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int, int]],
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 padding: Union[int, Tuple[int, int, int], str] = 0,
                 symbolic_expression: str = 'x',
                 groups: int = 1,
                 dilation: Union[int, Tuple[int, int, int]] = 1,
                 padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] = 'zeros',
                 bias: bool = True,
                 device=None, dtype=None,
                 mode: str = 'base'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         symbolic_expression, groups, dilation, padding_mode,
                         bias, device, dtype, ndim=3, is_transposed=False,
                         output_padding=0, mode=mode)


# ========== Transposed convolution classes ==========
class TNConvTranspose1d(_TNConvNd):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int]],
                 stride: Union[int, Tuple[int]] = 1,
                 padding: Union[int, Tuple[int]] = 0,
                 output_padding: Union[int, Tuple[int]] = 0,
                 symbolic_expression: str = 'x',
                 groups: int = 1,
                 dilation: Union[int, Tuple[int]] = 1,
                 bias: bool = True,
                 device=None, dtype=None,
                 mode: str = 'base'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         symbolic_expression, groups, dilation, 'zeros',
                         bias, device, dtype, ndim=1, is_transposed=True,
                         output_padding=output_padding, mode=mode)


class TNConvTranspose2d(_TNConvNd):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 output_padding: Union[int, Tuple[int, int]] = 0,
                 symbolic_expression: str = 'x',
                 groups: int = 1,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 bias: bool = True,
                 device=None, dtype=None,
                 mode: str = 'base'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         symbolic_expression, groups, dilation, 'zeros',
                         bias, device, dtype, ndim=2, is_transposed=True,
                         output_padding=output_padding, mode=mode)


class TNConvTranspose3d(_TNConvNd):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int, int]],
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 padding: Union[int, Tuple[int, int, int]] = 0,
                 output_padding: Union[int, Tuple[int, int, int]] = 0,
                 symbolic_expression: str = 'x',
                 groups: int = 1,
                 dilation: Union[int, Tuple[int, int, int]] = 1,
                 bias: bool = True,
                 device=None, dtype=None,
                 mode: str = 'base'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         symbolic_expression, groups, dilation, 'zeros',
                         bias, device, dtype, ndim=3, is_transposed=True,
                         output_padding=output_padding, mode=mode)