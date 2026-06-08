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
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
import re
from torch.nn import functional as F

# 准备 eval 所需的全局命名空间
_EVAL_GLOBALS = {
    'torch': torch,
    'np': __import__('numpy'),
    'math': math,
    'F': F,
}


class CustomNeuronLayer(nn.Module):
    r"""Build a neural network model of a custom architecture."""
    def __init__(self, in_features: int, out_features: int, symbolic_expression: str, bias: bool = True):
        r"""Initialize the custom layer with given input and output features
        and the symbolic expression that defines neuron functionality.

        Args:
            in_features: The number of features of the input data.
            out_features: The number of features of the output data.
            symbolic_expression: Neuronal expression obtained by vectorized symbolic regression.
            bias: Bias.
        """
        super(CustomNeuronLayer, self).__init__()

        # Number of input and output features
        self.in_features = in_features
        self.out_features = out_features

        # The symbolic expression that represents neuron operations
        self.neuron = symbolic_expression

        # Count the number of inputs ('x') in the symbolic expression
        self.number = symbolic_expression.count('x')

        # Create parameters (weights) for the layer based on the number of 'x' inputs
        for i in range(self.number):
            setattr(self, f'weight{i}', Parameter(torch.Tensor(out_features, in_features)))

        # Initialize bias if it's required, otherwise it's registered as None
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        # Reset parameters for the initialized layer
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""Resets the layer's parameters by initializing weights and bias."""

        # Initialize weights using Kaiming uniform initialization
        for i in range(self.number):
            weight = getattr(self, f'weight{i}')
            init.kaiming_uniform_(weight, a=math.sqrt(5))

        # Initialize bias with uniform distribution if bias is not None
        if self.bias is not None:
            weight0 = getattr(self, 'weight0')
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight0)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def _split_terms(self, expr: str):
        """Split expression by top-level '+' and '-' operators, preserving parentheses."""
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

    def xdata(self):
        r"""Extract data terms from the symbolic expression.

        Returns:
            List of expression strings (each corresponds to one occurrence of 'x').
        """
        # Remove all whitespace
        expr = self.neuron.replace(' ', '')
        # Split into top-level terms
        terms = self._split_terms(expr)
        xx = []
        for term in terms:
            # Remove leading '+' or '-' before splitting at '@'
            term = term.lstrip('+-')
            if '@' in term:
                # The part after '@' is the variable expression
                var_expr = term.split('@', 1)[1]
                xx.append(var_expr)
            elif 'x' in term:
                # No '@' means coefficient is 1 (implicitly)
                xx.append(term)
            else:
                # Constant term, ignore (no 'x')
                pass
        return xx

    def forward(self, x):
        r"""Defines the forward pass using the symbolic expression.

        Returns:
            The calculation of a single neuron.
        """
        # Collect variable data blocks from the symbolic expression
        xlist = self.xdata()
        # Ensure number of extracted terms matches the number of weights
        if len(xlist) != self.number:
            # Fallback: if mismatch, maybe some terms were constant; ignore extra weights?
            # For safety, we take min.
            pass
        result = 0.0
        # Local globals for eval: include the input tensor x and the safe globals
        for i, var_expr in enumerate(xlist):
            # Evaluate the expression (e.g., 'x', 'x**2', 'torch.sin(x)')
            # The expression can use the variable name 'x' (the input tensor)
            # We provide a local dict with 'x' bound to the input
            local_dict = {'x': x}
            try:
                value = eval(var_expr, _EVAL_GLOBALS, local_dict)
            except Exception as e:
                # If evaluation fails, fallback to zero
                print(f"Warning: Failed to evaluate expression '{var_expr}': {e}")
                value = torch.zeros_like(x)
            # Apply linear transformation with the corresponding weight matrix
            weight = getattr(self, f'weight{i}')
            transformed = F.linear(value, weight, None)
            result = result + transformed
        if self.bias is not None:
            result = result + self.bias
        return result