"""
Linear layer with symbolic aggregation supporting inner‑product cross terms.

This module provides a fully connected layer where the aggregation function is defined
by a symbolic expression. Two modes are supported:
- base (default): Uses SymPy parameterization with InnerProduct support.
- legacy: Uses simple eval-based expression parsing (original implementation).

Copyright (c) 2026 Tieyun LI. All Rights Reserved.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from sympy import symbols, Add, Mul, Pow, sin, cos, exp, simplify, expand, sympify, Symbol

from tnlearn.operator.inner_product import (
    InnerProduct,
    convert_pretty_to_innerproduct,
    parameterize_expression,
    is_inner_product,
)

__all__ = ['TNLinear']

# ---------- Global eval environment for legacy mode ----------
_EVAL_GLOBALS = {
    'torch': torch,
    'np': __import__('numpy'),
    'math': math,
    'F': F,
}


# ---------- Base mode evaluation engine (SymPy) ----------
def eval_sympy_expr(expr, subs):
    """
    Recursively evaluate a SymPy expression with torch tensors.

    Args:
        expr: SymPy expression node.
        subs: Dictionary mapping Symbol -> torch.Tensor.

    Returns:
        torch.Tensor: Result of the expression.
    """
    if expr.is_Number:
        device = next(iter(subs.values())).device
        return torch.tensor(float(expr), device=device)
    if expr.is_Symbol:
        return subs[expr]
    if expr.is_Add:
        terms = [eval_sympy_expr(arg, subs) for arg in expr.args]
        result = terms[0]
        for t in terms[1:]:
            result = result + t
        return result
    if expr.is_Mul:
        result = eval_sympy_expr(expr.args[0], subs)
        for arg in expr.args[1:]:
            result = result * eval_sympy_expr(arg, subs)
        return result
    if expr.is_Pow:
        return eval_sympy_expr(expr.base, subs) ** eval_sympy_expr(expr.exp, subs)
    if expr.func.__name__ == 'sin':
        return torch.sin(eval_sympy_expr(expr.args[0], subs))
    if expr.func.__name__ == 'cos':
        return torch.cos(eval_sympy_expr(expr.args[0], subs))
    if expr.func.__name__ == 'exp':
        return torch.exp(eval_sympy_expr(expr.args[0], subs))
    if expr.func.__name__ == 'tanh':
        return torch.tanh(eval_sympy_expr(expr.args[0], subs))
    if is_inner_product(expr):
        a = eval_sympy_expr(expr.args[0], subs)
        b = eval_sympy_expr(expr.args[1], subs)
        return (a * b).sum(dim=-1, keepdim=True)
    raise NotImplementedError(f"Unsupported expression: {expr}")


def evaluate_expression(expr, x_tensor, param_dict, out_dim):
    """
    Evaluate the parameterized expression for each output dimension.

    Args:
        expr: SymPy expression (parameterized).
        x_tensor: Input tensor of shape (..., in_features).
        param_dict: nn.ParameterDict mapping symbol names to parameters.
        out_dim: Number of output features.

    Returns:
        torch.Tensor: Output of shape (..., out_dim).
    """
    results = []
    x_sym = symbols('x')
    for j in range(out_dim):
        subs = {x_sym: x_tensor}
        for sym_str, param in param_dict.items():
            sym_obj = symbols(sym_str)
            # Take the j-th row, keep dimension (1, ...)
            subs[sym_obj] = param[j:j+1]
        val = eval_sympy_expr(expr, subs)
        # Ensure the last dimension is 1 for concatenation
        if val.dim() == 1:
            val = val.unsqueeze(-1)
        results.append(val)
    # Concatenate along the last dimension (feature dimension)
    return torch.cat(results, dim=-1)


# ---------- Legacy mode helpers ----------
def _split_terms(expr: str):
    """Split expression by top-level '+' and '-' while respecting parentheses."""
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
    """Extract all sub‑expressions containing 'x', ignoring pure constant terms."""
    expr = expr.replace(' ', '')
    terms = _split_terms(expr)
    x_terms = []
    for t in terms:
        t = t.lstrip('+-')
        if 'x' in t:
            # If '@' is present, keep only the part after '@' (the variable expression)
            if '@' in t:
                var_expr = t.split('@', 1)[1]
            else:
                var_expr = t
            x_terms.append(var_expr)
    return x_terms


# ---------- TNLinear class ----------
class TNLinear(nn.Module):
    r"""
    A fully connected layer with custom symbolic aggregation.

    The layer accepts a symbolic expression using `x` as input. Two modes are available:
    - base (default): Uses SymPy parameterization with InnerProduct support.
    - legacy: Uses eval-based parsing (original implementation).

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        symbolic_expression (str): Symbolic expression using `x` as input.
            Example: 'x + sin(x) + <2, x>' becomes learnable weights.
        bias (bool): If True, adds a learnable bias (independent of expression).
        device (torch.device, optional): Device for parameters.
        dtype (torch.dtype, optional): Data type for parameters.
        mode (str): 'base' or 'legacy'. Default 'base'.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 symbolic_expression: str = 'x',
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 mode: str = 'base'):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.symbolic_expression = symbolic_expression
        self.bias_flag = bias
        self.device = device
        self.dtype = dtype
        self.mode = mode.lower()
        if self.mode not in ('base', 'legacy'):
            raise ValueError("mode must be 'base' or 'legacy'")

        if self.mode == 'base':
            # ---------- Base mode (SymPy) ----------
            # 1. Convert and parameterize expression
            converted_str = convert_pretty_to_innerproduct(symbolic_expression)
            raw_expr = sympify(converted_str, locals={'InnerProduct': InnerProduct})
            raw_expr = expand(simplify(raw_expr))
            self.param_expr = parameterize_expression(raw_expr, include_bias=False)

            # 2. Extract symbols: w_i (weights) and c_i (coefficients)
            all_symbols = self.param_expr.free_symbols if self.param_expr != 0 else set()
            x_sym = symbols('x')
            weight_symbols = [sym for sym in all_symbols if sym != x_sym]
            w_syms = [sym for sym in weight_symbols if str(sym).startswith('w')]
            c_syms = [sym for sym in weight_symbols if str(sym).startswith('c')]

            # 3. Create parameter dict
            self.param_dict = nn.ParameterDict()
            for sym in w_syms:
                name = str(sym)
                self.param_dict[name] = nn.Parameter(
                    torch.empty(out_features, in_features, device=device, dtype=dtype)
                )
            for sym in c_syms:
                name = str(sym)
                self.param_dict[name] = nn.Parameter(
                    torch.empty(out_features, 1, device=device, dtype=dtype)
                )

            self._w_syms = w_syms
            self._c_syms = c_syms
            self._x_sym = x_sym
            self._base_initialized = True
            self._legacy_initialized = False

        else:  # legacy mode
            # ---------- Legacy mode (eval) ----------
            self.terms = _parse_expression(symbolic_expression)
            if not self.terms:
                self.terms = ['x']    # fallback to linear

            # Pre-compile basis functions into lambdas
            self.funcs = []
            for expr in self.terms:
                try:
                    fn = eval('lambda x: ' + expr, _EVAL_GLOBALS)
                except Exception as e:
                    print(f"Legacy: eval failed for '{expr}', using identity. Error: {e}")
                    fn = lambda x: x
                self.funcs.append(fn)

            # Create combined weight: shape (out_features, in_features * num_terms)
            self.num_terms = len(self.terms)
            self.weight = nn.Parameter(
                torch.empty(out_features, self.in_features * self.num_terms,
                            device=device, dtype=dtype)
            )
            self._base_initialized = False
            self._legacy_initialized = True

        # Bias (shared single bias)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.mode == 'base':
            for name, param in self.param_dict.items():
                if name.startswith('w'):
                    init.kaiming_uniform_(param, a=math.sqrt(5))
                elif name.startswith('c'):
                    fan_in = self.in_features
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    init.uniform_(param, -bound, bound)
        else:  # legacy
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features).

        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """
        if self.mode == 'base':
            out = evaluate_expression(self.param_expr, x, self.param_dict, self.out_features)
        else:  # legacy
            # Compute all basis functions and concatenate (augmented features)
            augmented = torch.cat([func(x) for func in self.funcs], dim=-1)
            out = F.linear(augmented, self.weight, None)

        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self):
        mode_str = f', mode="{self.mode}"' if self.mode != 'base' else ''
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.bias is not None}, symbolic_expression="{self.symbolic_expression}"'
                f'{mode_str}')

    # ---------- Serialization support ----------
    def __getstate__(self):
        """
        Remove non‑picklable attributes (SymPy objects or lambdas) before saving.
        """
        state = self.__dict__.copy()
        # Remove SymPy expressions and symbols (base mode)
        state.pop('param_expr', None)
        state.pop('_x_sym', None)
        # Remove lambdas (legacy mode)
        state.pop('funcs', None)
        # Keep all other attributes (including parameters)
        return state

    def __setstate__(self, state):
        """
        Restore the object and rebuild SymPy expression or lambdas from stored strings.
        """
        self.__dict__.update(state)
        if self.mode == 'base':
            # Rebuild param_expr from symbolic_expression
            converted_str = convert_pretty_to_innerproduct(self.symbolic_expression)
            raw_expr = sympify(converted_str, locals={'InnerProduct': InnerProduct})
            raw_expr = expand(simplify(raw_expr))
            self.param_expr = parameterize_expression(raw_expr, include_bias=False)
            self._x_sym = symbols('x')
        else:  # legacy
            # Rebuild funcs from self.terms
            self.funcs = []
            for expr in self.terms:
                try:
                    fn = eval('lambda x: ' + expr, _EVAL_GLOBALS)
                except Exception as e:
                    print(f"Legacy: eval failed for '{expr}', using identity. Error: {e}")
                    fn = lambda x: x
                self.funcs.append(fn)