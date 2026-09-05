"""
InnerProduct operator and parameterization utilities for symbolic expressions.

This module provides:
- InnerProduct class: a symbolic representation of the inner product <a, b>.
- Functions to convert between pretty-printed and InnerProduct forms.
- Parameterization of expressions: replacing numeric constants and known symbols
  with learnable parameters (w, c, b), with an option to suppress automatic
  coefficient addition for already-parameterized expressions.

Copyright (c) 2024 Meng WANG. All Rights Reserved.
Copyright (c) 2026 Tieyun LI. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from sympy import Expr, Add, Mul, sympify, symbols, Symbol, Number, Pow, sin, cos, exp
import re
import numpy as np
import torch


class InnerProduct(Expr):
    """
    Symbolic representation of an inner product <left, right>.
    """
    def __new__(cls, left, right, **kwargs):
        left = sympify(left)
        right = sympify(right)
        obj = Expr.__new__(cls, left, right, **kwargs)
        return obj

    @property
    def left(self):
        return self.args[0]

    @property
    def right(self):
        return self.args[1]

    def __str__(self):
        return f"InnerProduct({self.left}, {self.right})"

    def _sympystr(self, printer):
        return f"InnerProduct({printer.doprint(self.left)}, {printer.doprint(self.right)})"

    def __repr__(self):
        return f"<{self.left}, {self.right}>"

    def _eval_expand_basic(self, **hints):
        left, right = self.left, self.right
        if isinstance(left, Add) and isinstance(right, Add):
            return Add(*[InnerProduct(lterm, rterm)
                         for lterm in left.args for rterm in right.args])
        if isinstance(left, Add):
            return Add(*[InnerProduct(term, right) for term in left.args])
        if isinstance(right, Add):
            return Add(*[InnerProduct(left, term) for term in right.args])
        return self

    def _eval_simplify(self, **kwargs):
        left = _simplify_expr(self.left)
        right = _simplify_expr(self.right)
        if left == 0 or right == 0:
            return 0
        expanded = InnerProduct(left, right)._eval_expand_basic(**kwargs)
        if expanded != InnerProduct(left, right):
            return _simplify_expr(expanded, **kwargs)
        return InnerProduct(left, right)


def _simplify_expr(expr, **kwargs):
    """
    Recursively simplify InnerProduct and arithmetic expressions.
    Also extracts leading minus signs from InnerProduct arguments.
    """
    from sympy import Mul, S, Add
    if isinstance(expr, InnerProduct):
        left = _simplify_expr(expr.left, **kwargs)
        right = _simplify_expr(expr.right, **kwargs)
        if left == 0 or right == 0:
            return 0

        def extract_sign(e):
            """Extract sign from expression: returns (sign, core) where sign is 1 or -1."""
            if e.is_Mul:
                coeff, rest = e.as_coeff_Mul()
                if coeff < 0:
                    return -1, Mul(-coeff, rest)
                else:
                    return 1, e
            elif e.is_Number and e < 0:
                return -1, -e
            else:
                return 1, e

        sign_left, core_left = extract_sign(left)
        sign_right, core_right = extract_sign(right)
        total_sign = sign_left * sign_right

        inner = InnerProduct(core_left, core_right)
        expanded = inner._eval_expand_basic(**kwargs)
        if expanded != inner:
            result = _simplify_expr(expanded, **kwargs)
        else:
            result = inner

        if total_sign == -1:
            return Mul(-1, result)
        else:
            return result

    elif isinstance(expr, Add):
        new_args = [_simplify_expr(arg, **kwargs) for arg in expr.args]
        new_args = [a for a in new_args if a != 0]
        return Add(*new_args) if new_args else 0

    elif isinstance(expr, Mul):
        new_args = [_simplify_expr(arg, **kwargs) for arg in expr.args]
        if any(a == 0 for a in new_args):
            return 0
        return Mul(*new_args)

    else:
        if hasattr(expr, 'args') and expr.args:
            new_args = [_simplify_expr(arg, **kwargs) for arg in expr.args]
            return expr.func(*new_args)
        return expr


def convert_pretty_to_innerproduct(expr_str):
    """Convert pretty-printed <...> expressions to InnerProduct(...) function calls."""
    pattern = r'<([^<>]+)>'
    def repl(match):
        inner = match.group(1).strip()
        return f"InnerProduct({inner})"
    while '<' in expr_str:
        expr_str = re.sub(pattern, repl, expr_str)
    return expr_str


def convert_innerproduct_to_pretty(expr_str: str) -> str:
    """
    Convert InnerProduct(...) and IP(...) to pretty <...> format.

    Example:
        "InnerProduct(w, x**2) + IP(c, x)" -> "<w, x**2> + <c, x>"
    """
    expr_str = re.sub(r'\bIP\s*\(', 'InnerProduct(', expr_str)
    pattern = r'InnerProduct\s*\(([^()]*)\)'
    while re.search(pattern, expr_str):
        expr_str = re.sub(pattern, r'<\1>', expr_str)
    return expr_str


def is_inner_product(expr):
    """Check if an expression is an InnerProduct instance."""
    return isinstance(expr, InnerProduct)


def replace_numbers(expr, counter, in_exponent=False):
    """
    Replace numeric constants with new weight symbols (w_i).
    """
    if isinstance(expr, Symbol):
        return expr
    if expr.is_Number:
        if not in_exponent:
            w = symbols('w%d' % counter['w'])
            counter['w'] += 1
            return w
        return expr
    elif expr.is_Pow:
        base, exp = expr.args
        return Pow(replace_numbers(base, counter, False),
                   replace_numbers(exp, counter, True))
    elif expr.is_Mul:
        return Mul(*[replace_numbers(arg, counter, in_exponent) for arg in expr.args])
    elif expr.is_Add:
        return Add(*[replace_numbers(arg, counter, in_exponent) for arg in expr.args])
    elif is_inner_product(expr):
        a1, a2 = expr.args
        return InnerProduct(replace_numbers(a1, counter, False),
                            replace_numbers(a2, counter, False))
    elif hasattr(expr, 'args') and not isinstance(expr, (Symbol, int, float, Number)):
        return expr.func(*[replace_numbers(arg, counter, False) for arg in expr.args])
    else:
        return expr


def parameterize_term(term, counter, include_bias=False, already_parametrized=True):
    """
    Parameterize a single term of an expression.

    Args:
        term: a SymPy expression (single term, not an Add)
        counter: dict with keys 'w', 'c', 'b' to keep track of parameter indices
        include_bias: if True, replace standalone numeric constants with b_i
        already_parametrized: if True, do NOT add an extra coefficient 'c'
            even if no new weight symbol is generated. This is useful when the
            expression already contains parameter symbols (e.g., w1, c1, b1).
            If False, the original logic applies: add 'c' only when no new
            weight symbol is introduced.
            Default: True.
    """
    if term.is_number:
        if include_bias:
            b = symbols('b%d' % counter['b'])
            counter['b'] += 1
            return b
        else:
            return 0
    if is_inner_product(term):
        a1, a2 = term.args
        old_w = counter['w']
        new_a1 = replace_numbers(a1, counter, False)
        new_a2 = replace_numbers(a2, counter, False)
        new_ip = InnerProduct(new_a1, new_a2)
        if counter['w'] == old_w:
            # No new weight generated; add 'c' only if not already_parametrized
            if not already_parametrized:
                c = symbols('c%d' % counter['c'])
                counter['c'] += 1
                return Mul(c, new_ip)
            else:
                return new_ip
        return new_ip
    if isinstance(term, Mul):
        inner_products = []
        non_inner = []
        created_param = False
        for factor in term.args:
            if factor.is_number:
                continue
            elif is_inner_product(factor):
                a1, a2 = factor.args
                old_w = counter['w']
                new_a1 = replace_numbers(a1, counter, False)
                new_a2 = replace_numbers(a2, counter, False)
                if counter['w'] > old_w:
                    created_param = True
                inner_products.append(InnerProduct(new_a1, new_a2))
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
        if not created_param and not already_parametrized:
            # Only add 'c' if no new weight was generated and we are not already parameterized
            c = symbols('c%d' % counter['c'])
            counter['c'] += 1
            result = Mul(c, result)
        return result
    # For other cases (e.g., a simple symbol or function)
    new_expr = replace_numbers(term, counter, False)
    w = symbols('w%d' % counter['w'])
    counter['w'] += 1
    # A new weight is always generated here, so no 'c' needed regardless of already_parametrized
    return InnerProduct(w, new_expr)


def parameterize_expression(expr, include_bias=False, already_parametrized=True):
    """
    Parameterize an entire expression by replacing constants and known symbols with learnable parameters.

    Args:
        expr: a SymPy expression (possibly an Add of multiple terms)
        include_bias: whether to replace standalone constants with b_i
        already_parametrized: passed to parameterize_term; controls automatic addition
            of extra coefficients 'c'. Default: True.
    Returns:
        A new SymPy expression with parameters.
    """
    if isinstance(expr, Add):
        terms = list(expr.args)
    else:
        terms = [expr]
    counter = {'w': 1, 'c': 1, 'b': 1}
    new_terms = []
    for t in terms:
        pt = parameterize_term(t, counter, include_bias, already_parametrized)
        if pt != 0:
            new_terms.append(pt)
    if not new_terms:
        return 0
    return Add(*new_terms) if len(new_terms) > 1 else new_terms[0]


def inner_product(a, b, N=None):
    """
    Compute the inner product (sum of elementwise products) for numpy or torch tensors.

    Supports broadcasting and reshaping to handle batched inputs.
    """
    is_torch = isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor)
    if is_torch:
        a = torch.as_tensor(a) if not isinstance(a, torch.Tensor) else a
        b = torch.as_tensor(b) if not isinstance(b, torch.Tensor) else b
        if N is None:
            if a.ndim >= 1:
                N = a.shape[0]
            elif b.ndim >= 1:
                N = b.shape[0]
            else:
                N = 1
        if a.ndim == 0:
            a = a.expand(N, 1)
        if b.ndim == 0:
            b = b.expand(N, 1)
        if a.ndim == 1:
            if a.shape[0] == N:
                a = a.unsqueeze(1)
            else:
                a = a.expand(N, -1)
        if b.ndim == 1:
            if b.shape[0] == N:
                b = b.unsqueeze(1)
            else:
                b = b.expand(N, -1)
        if a.ndim == 2 and b.ndim == 2:
            if a.shape[1] != b.shape[1]:
                if a.shape[1] == 1:
                    a = a.expand(-1, b.shape[1])
                elif b.shape[1] == 1:
                    b = b.expand(-1, a.shape[1])
                else:
                    raise ValueError(f"Dim mismatch: {a.shape[1]} vs {b.shape[1]}")
            return torch.sum(a * b, dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unsupported ndim: a={a.ndim}, b={b.ndim}")
    else:
        a = np.asarray(a)
        b = np.asarray(b)
        if N is None:
            if a.ndim >= 1:
                N = a.shape[0]
            elif b.ndim >= 1:
                N = b.shape[0]
            else:
                N = 1
        if a.ndim == 0:
            a = np.full((N, 1), a)
        if b.ndim == 0:
            b = np.full((N, 1), b)
        if a.ndim == 1:
            if a.shape[0] == N:
                a = a[:, np.newaxis]
            else:
                a = np.broadcast_to(a, (N, a.shape[0]))
        if b.ndim == 1:
            if b.shape[0] == N:
                b = b[:, np.newaxis]
            else:
                b = np.broadcast_to(b, (N, b.shape[0]))
        if a.ndim == 2 and b.ndim == 2:
            if a.shape[1] != b.shape[1]:
                if a.shape[1] == 1:
                    a = np.broadcast_to(a, (N, b.shape[1]))
                elif b.shape[1] == 1:
                    b = np.broadcast_to(b, (N, a.shape[1]))
                else:
                    raise ValueError(f"Dim mismatch: {a.shape[1]} vs {b.shape[1]}")
            return np.sum(a * b, axis=-1, keepdims=True)
        else:
            raise ValueError(f"Unsupported ndim: a={a.ndim}, b={b.ndim}")


def eval_sympy_expr(expr, subs):
    """
    Evaluate a SymPy expression with torch tensors as substitutions.
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
    if is_inner_product(expr):
        a = eval_sympy_expr(expr.args[0], subs)
        b = eval_sympy_expr(expr.args[1], subs)
        return (a * b).sum(dim=-1, keepdim=True)
    raise NotImplementedError(f"Unsupported expression: {expr}")


def evaluate_expression(expr, x_tensor, param_dict, out_dim):
    """
    Evaluate a parameterized expression for a batch of inputs and all output neurons.
    """
    batch = x_tensor.size(0)
    results = []
    x_sym = symbols('x')
    for j in range(out_dim):
        subs = {x_sym: x_tensor}
        for sym_str, param in param_dict.items():
            sym_obj = symbols(sym_str)
            if param.dim() == 2 and param.size(0) == out_dim:
                subs[sym_obj] = param[j:j+1]
            elif param.dim() == 2 and param.size(0) == out_dim and param.size(1) == 1:
                subs[sym_obj] = param[j:j+1]
            elif param.dim() == 1 and param.size(0) == out_dim:
                subs[sym_obj] = param[j:j+1]
            else:
                subs[sym_obj] = param
        val = eval_sympy_expr(expr, subs)
        if val.dim() == 1:
            val = val.unsqueeze(-1)
        results.append(val)
    return torch.cat(results, dim=1)


def neuronseek_config_to_string(config: dict) -> str:
    """
    Convert a NeuronSeek configuration dictionary to a human-readable mathematical expression string.

    The expression consists of a polynomial stream, an interaction stream, and an optional periodic stream.
    Pure terms are represented as <w_i, x> for k=1 and <w_i, x**k> for k>1.
    Interaction terms are represented as products of linear inner products <w_j, x>
    for each interaction order m in `interact_indices`, using a rank-1 CP decomposition.
    If `periodic` is True, a term <w_i, sin(x)> is appended.

    This function performs input validation and gracefully handles missing fields,
    invalid data types, and non-positive orders. The `rank` parameter is currently
    ignored, as only rank-1 decompositions are used for interaction terms.

    Args:
        config (dict): A dictionary containing the following optional keys:
            - pure_indices (list[int]): List of positive integers indicating the
                power orders for pure polynomial terms. Defaults to [].
            - interact_indices (list[int]): List of positive integers indicating
                the interaction orders. Each order m will produce a product of
                m linear inner products. Defaults to [].
            - interaction_form (str): Type of interaction. Only 'cp_inner_product'
                is fully supported; other values trigger a warning but still
                generate an expression using inner products. Defaults to
                'cp_inner_product'.
            - rank (int): CP rank; currently ignored.
            - periodic (bool): If True, includes a periodic term <w_i, sin(x)>.
                Defaults to False.

    Returns:
        str: A string representing the aggregation function, with terms joined by '+'.
            Returns an empty string if no valid terms are present.

    Raises:
        TypeError: If `pure_indices` or `interact_indices` is present but not a list.

    Examples:
        >>> config = {'pure_indices': [1, 2], 'interact_indices': [2, 3]}
        >>> neuronseek_config_to_string(config)
        '<w1,x>+<w2,x**2>+<w3,x>*<w4,x>+<w5,x>*<w6,x>*<w7,x>'

        >>> config = {'pure_indices': [1], 'interact_indices': [2], 'periodic': True}
        >>> neuronseek_config_to_string(config)
        '<w1,x>+<w2,x>*<w3,x>+<w4,sin(x)>'

        >>> neuronseek_config_to_string({})
        ''

        >>> neuronseek_config_to_string({'pure_indices': [0, -1, 2]})
        '<w1,x**2>'

        >>> neuronseek_config_to_string({'pure_indices': 'not a list'})
        Traceback (most recent call last):
        ...
        TypeError: 'pure_indices' must be a list
    """
    # ---- Safely retrieve and validate fields ----
    pure = config.get('pure_indices')
    if pure is None:
        pure = []
    if not isinstance(pure, list):
        raise TypeError("'pure_indices' must be a list")

    inter = config.get('interact_indices')
    if inter is None:
        inter = []
    if not isinstance(inter, list):
        raise TypeError("'interact_indices' must be a list")

    def is_valid_order(v: any) -> bool:
        """Return True if v is a positive integer."""
        try:
            return isinstance(v, int) and v > 0
        except Exception:
            return False

    pure = [k for k in pure if is_valid_order(k)]
    inter = [m for m in inter if is_valid_order(m)]

    form = config.get('interaction_form', 'cp_inner_product')
    if form != 'cp_inner_product':
        import warnings
        warnings.warn(
            f"interaction_form '{form}' is not fully supported; "
            "using cp_inner_product representation (rank 1).",
            UserWarning
        )

    parts = []
    w_idx = 1

    for k in pure:
        if k == 1:
            parts.append(f"<w{w_idx}, x>")
        else:
            parts.append(f"<w{w_idx}, x**{k}>")
        w_idx += 1

    for m in inter:
        factors = []
        for _ in range(m):
            factors.append(f"<w{w_idx}, x>")
            w_idx += 1
        parts.append("*".join(factors))

    if config.get('periodic', False):
        parts.append(f"<w{w_idx}, sin(x)>")
        w_idx += 1

    return "+".join(parts)


# Alias for convenience
IP = InnerProduct