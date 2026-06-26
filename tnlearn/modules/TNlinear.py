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
    r"""Custom fully connected layer that supports basis function combinations defined by symbolic expressions.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        symbolic_expression (str): Symbolic expression, e.g., 'x + sin(x) + 0.1@x**2', where 'x' denotes the input.
        bias (bool): Whether to use bias, default True.
        device (torch.device, optional): Device for weights and bias.
        dtype (torch.dtype, optional): Data type for weights and bias.
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

        # Parse the expression to extract basis terms
        self.terms = self._parse_expression(symbolic_expression)
        if not self.terms:
            self.terms = ['x']    # fallback to linear

        # Pre-compile basis functions into lambdas for faster forward pass
        self.funcs = []
        for expr in self.terms:
            try:
                fn = eval('lambda x: ' + expr, _EVAL_GLOBALS)
            except Exception as e:
                print(f"Warning: eval failed for '{expr}', using identity. Error: {e}")
                fn = lambda x: x
            self.funcs.append(fn)

        # Create combined weight: shape (out_features, in_features * num_terms)
        self.num_terms = len(self.terms)
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features * self.num_terms,
                        device=device, dtype=dtype)
        )

        # Bias (shared single bias)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('funcs', None)   # lambdas are not serializable
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Rebuild funcs
        self.funcs = []
        for expr in self.terms:
            try:
                fn = eval('lambda x: ' + expr, _EVAL_GLOBALS)
            except Exception as e:
                print(f"Warning: eval failed for '{expr}', using identity. Error: {e}")
                fn = lambda x: x
            self.funcs.append(fn)

    def reset_parameters(self):
        r"""Initialize weights and bias using Kaiming uniform initialization."""
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def _split_terms(self, expr: str):
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

    def _parse_expression(self, expr: str):
        """Extract all sub‑expressions containing 'x', ignoring pure constant terms."""
        expr = expr.replace(' ', '')
        terms = self._split_terms(expr)
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

    def forward(self, x):
        r"""Forward pass: single linear transformation after feature augmentation, plus bias."""
        # Compute all basis functions and concatenate (augmented features)
        augmented = torch.cat([func(x) for func in self.funcs], dim=-1)
        # Single linear transformation
        out = F.linear(augmented, self.weight, self.bias)
        return out
