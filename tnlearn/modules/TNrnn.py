"""
Recurrent neural network layers with custom neuron aggregation.

This module provides RNN, LSTM, GRU and their cell variants where the input
is first transformed by a user‑defined symbolic expression (e.g., 'x + torch.sin(x)')
before being fed into the recurrent computation. The transformation is applied
element‑wise to the input features and the results are concatenated, effectively
augmenting the input dimension with learnable non‑linearities.

Based on PyTorch's torch.nn.modules.rnn.
"""

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, init
from torch.nn.utils.rnn import PackedSequence

from .TNlinear import TNLinear

__all__ = [
    'TNRNNBase', 'TNRNN', 'TNLSTM', 'TNGRU',
    'TNRNNCellBase', 'TNRNNCell', 'TNLSTMCell', 'TNGRUCell'
]

# ---------- Global evaluation environment for expression parsing ----------
_EVAL_GLOBALS = {
    'torch': torch,
    'np': __import__('numpy'),
    'math': math,
    'F': F,
}


def _parse_expression(expr: str) -> list:
    """
    Parses a symbolic expression and returns a list of callable basis functions.

    The expression is split at the top‑level '+' and '-' operators (respecting
    parentheses). Only sub‑expressions that contain 'x' are kept; each such
    sub‑expression is compiled into a lambda function that takes a tensor `x`
    and returns the transformed tensor. If a term contains '@', the part after
    '@' is used as the variable expression (the coefficient is ignored).

    Args:
        expr (str): The symbolic expression, e.g., 'x + torch.sin(x)'.

    Returns:
        list: A list of callable functions, each accepting a tensor and
        returning a tensor. If no valid 'x' term is found, returns [lambda x: x].
    """
    expr = expr.replace(' ', '')
    # Split at top‑level '+' and '-', ignoring parentheses
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

    funcs = []
    for t in terms:
        t = t.lstrip('+-')
        if 'x' in t:
            # If '@' is present, use the part after it as the variable expression
            if '@' in t:
                var_expr = t.split('@', 1)[1]
            else:
                var_expr = t
            try:
                fn = eval('lambda x: ' + var_expr, _EVAL_GLOBALS)
            except Exception as e:
                print(f"Warning: eval failed for '{var_expr}', using identity. Error: {e}")
                fn = lambda x: x
            funcs.append(fn)
    if not funcs:
        funcs = [lambda x: x]   # fallback to linear
    return funcs


def _augment_input(x: Tensor, funcs: list) -> Tensor:
    """
    Applies all basis functions to the input and concatenates the results.

    Args:
        x (Tensor): Input tensor of shape (..., feature_dim).
        funcs (list): List of callable functions.

    Returns:
        Tensor: Augmented tensor with shape (..., feature_dim * len(funcs)).
    """
    augmented = [func(x) for func in funcs]
    return torch.cat(augmented, dim=-1)


# ---------- Multi‑layer RNN base (feature augmentation + native RNN) ----------

class TNRNNBase(nn.Module):
    """
    Base class for multi‑layer RNNs with custom neuron aggregation.

    This class augments the input by applying the basis functions from the
    symbolic expression and concatenating the results, then delegates the
    actual recurrent computation to PyTorch's native RNN, LSTM, or GRU.
    This approach leverages the efficient `_VF` implementations.

    Args:
        mode (str): One of 'RNN', 'LSTM', 'GRU'.
        input_size (int): The number of expected features in the input.
        hidden_size (int): The number of features in the hidden state.
        num_layers (int): Number of recurrent layers. Default: 1.
        bias (bool): If False, the layer does not use bias weights. Default: True.
        batch_first (bool): If True, input and output tensors are provided as
            (batch, seq, feature). Default: False.
        dropout (float): If non‑zero, introduces a Dropout layer on the outputs
            of each RNN layer except the last. Default: 0.0.
        bidirectional (bool): If True, becomes a bidirectional RNN. Default: False.
        symbolic_expression (str): Symbolic expression defining the basis functions.
            Default: 'x'.
        device (torch.device, optional): Device for the parameters.
        dtype (torch.dtype, optional): Data type for the parameters.
    """
    __constants__ = ['input_size', 'hidden_size', 'num_layers', 'bias',
                     'batch_first', 'dropout', 'bidirectional', 'symbolic_expression']

    def __init__(self,
                 mode: str,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 bias: bool = True,
                 batch_first: bool = False,
                 dropout: float = 0.0,
                 bidirectional: bool = False,
                 symbolic_expression: str = 'x',
                 device=None,
                 dtype=None):
        super().__init__()
        self.mode = mode
        self.original_input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.symbolic_expression = symbolic_expression

        # Parse expression into basis functions
        self.funcs = _parse_expression(symbolic_expression)
        self.num_funcs = len(self.funcs)
        self.augmented_input_size = input_size * self.num_funcs

        # Create the native RNN module
        rnn_cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[mode]
        self.rnn = rnn_cls(
            input_size=self.augmented_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            device=device,
            dtype=dtype
        )

    def _augment(self, x: Tensor) -> Tensor:
        """Augments the input by applying all basis functions and concatenating."""
        augmented = [func(x) for func in self.funcs]
        return torch.cat(augmented, dim=-1)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None):
        """
        Forward pass of the RNN.

        Args:
            input (Tensor): Input tensor. Shape depends on batch_first.
            hx (Tensor, optional): Initial hidden state. If not provided, defaults to zeros.

        Returns:
            output, hidden_state: Same as the native RNN.
        """
        aug_input = self._augment(input)
        return self.rnn(aug_input, hx)

    def extra_repr(self) -> str:
        s = f'input_size={self.original_input_size}, hidden_size={self.hidden_size}'
        if self.num_layers != 1:
            s += f', num_layers={self.num_layers}'
        if self.bias is not True:
            s += f', bias={self.bias}'
        if self.batch_first is not False:
            s += f', batch_first={self.batch_first}'
        if self.dropout != 0:
            s += f', dropout={self.dropout}'
        if self.bidirectional is not False:
            s += f', bidirectional={self.bidirectional}'
        if self.symbolic_expression != 'x':
            s += f', symbolic_expression={self.symbolic_expression}'
        return s

    def __getstate__(self):
        # Remove compiled lambdas for pickling
        state = self.__dict__.copy()
        state.pop('funcs', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Re‑build funcs from the expression
        self.funcs = _parse_expression(self.symbolic_expression)


# ---------- Concrete RNN classes ----------

class TNRNN(TNRNNBase):
    """
    A multi‑layer RNN with custom neuron aggregation.

    This is the equivalent of :class:`torch.nn.RNN` where the input is first
    augmented by basis functions defined in `symbolic_expression`.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden features.
        num_layers (int): Number of recurrent layers. Default: 1.
        nonlinearity (str): The non‑linearity to use: 'tanh' or 'relu'. Default: 'tanh'.
        bias (bool): If False, no bias weights are used. Default: True.
        batch_first (bool): If True, input/output shape is (batch, seq, feature). Default: False.
        dropout (float): Dropout probability. Default: 0.0.
        bidirectional (bool): If True, becomes bidirectional. Default: False.
        symbolic_expression (str): Basis function expression. Default: 'x'.
        device (torch.device, optional): Device for parameters.
        dtype (torch.dtype, optional): Data type for parameters.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 nonlinearity: str = 'tanh', bias: bool = True, batch_first: bool = False,
                 dropout: float = 0.0, bidirectional: bool = False,
                 symbolic_expression: str = 'x', device=None, dtype=None):
        super().__init__('RNN', input_size, hidden_size, num_layers, bias,
                         batch_first, dropout, bidirectional, symbolic_expression,
                         device, dtype)
        self.rnn.nonlinearity = nonlinearity


class TNLSTM(TNRNNBase):
    """
    A multi‑layer LSTM with custom neuron aggregation.

    This is the equivalent of :class:`torch.nn.LSTM` where the input is first
    augmented by basis functions defined in `symbolic_expression`.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden features.
        num_layers (int): Number of recurrent layers. Default: 1.
        bias (bool): If False, no bias weights are used. Default: True.
        batch_first (bool): If True, input/output shape is (batch, seq, feature). Default: False.
        dropout (float): Dropout probability. Default: 0.0.
        bidirectional (bool): If True, becomes bidirectional. Default: False.
        symbolic_expression (str): Basis function expression. Default: 'x'.
        device (torch.device, optional): Device for parameters.
        dtype (torch.dtype, optional): Data type for parameters.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 bias: bool = True, batch_first: bool = False,
                 dropout: float = 0.0, bidirectional: bool = False,
                 symbolic_expression: str = 'x', device=None, dtype=None):
        super().__init__('LSTM', input_size, hidden_size, num_layers, bias,
                         batch_first, dropout, bidirectional, symbolic_expression,
                         device, dtype)


class TNGRU(TNRNNBase):
    """
    A multi‑layer GRU with custom neuron aggregation.

    This is the equivalent of :class:`torch.nn.GRU` where the input is first
    augmented by basis functions defined in `symbolic_expression`.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden features.
        num_layers (int): Number of recurrent layers. Default: 1.
        bias (bool): If False, no bias weights are used. Default: True.
        batch_first (bool): If True, input/output shape is (batch, seq, feature). Default: False.
        dropout (float): Dropout probability. Default: 0.0.
        bidirectional (bool): If True, becomes bidirectional. Default: False.
        symbolic_expression (str): Basis function expression. Default: 'x'.
        device (torch.device, optional): Device for parameters.
        dtype (torch.dtype, optional): Data type for parameters.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 bias: bool = True, batch_first: bool = False,
                 dropout: float = 0.0, bidirectional: bool = False,
                 symbolic_expression: str = 'x', device=None, dtype=None):
        super().__init__('GRU', input_size, hidden_size, num_layers, bias,
                         batch_first, dropout, bidirectional, symbolic_expression,
                         device, dtype)


# ---------- Cell versions (using TNLinear for per‑time‑step computation) ----------

class TNRNNCellBase(nn.Module):
    """
    Base class for RNN cells with custom neuron aggregation.

    This class uses :class:`TNLinear` for both input‑to‑hidden and
    hidden‑to‑hidden projections, applying the basis functions from the
    symbolic expression inside each linear layer. It is intended to be
    subclassed by specific cell types.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden features.
        bias (bool): If True, adds a bias to both linear layers.
        symbolic_expression (str): Basis function expression. Default: 'x'.
        num_chunks (int): Number of chunks to split the output into (e.g., 4 for LSTM).
        device (torch.device, optional): Device for parameters.
        dtype (torch.dtype, optional): Data type for parameters.
    """
    __constants__ = ['input_size', 'hidden_size', 'bias', 'symbolic_expression']

    def __init__(self, input_size: int, hidden_size: int, bias: bool,
                 symbolic_expression: str = 'x', num_chunks: int = 1,
                 device=None, dtype=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.symbolic_expression = symbolic_expression
        self.num_chunks = num_chunks

        # Input‑to‑hidden linear layer
        self.ih = TNLinear(
            in_features=input_size,
            out_features=num_chunks * hidden_size,
            symbolic_expression=symbolic_expression,
            bias=bias,
            device=device,
            dtype=dtype
        )
        # Hidden‑to‑hidden linear layer
        self.hh = TNLinear(
            in_features=hidden_size,
            out_features=num_chunks * hidden_size,
            symbolic_expression=symbolic_expression,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Parameter initialisation (handled internally by TNLinear)."""
        pass

    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.symbolic_expression != 'x':
            s += ', symbolic_expression={symbolic_expression}'
        return s.format(**self.__dict__)


class TNRNNCell(TNRNNCellBase):
    """
    An RNN cell with custom neuron aggregation.

    This is the equivalent of :class:`torch.nn.RNNCell` where both linear
    transformations use :class:`TNLinear` with the given symbolic expression.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden features.
        bias (bool): If True, adds a bias. Default: True.
        nonlinearity (str): 'tanh' or 'relu'. Default: 'tanh'.
        symbolic_expression (str): Basis function expression. Default: 'x'.
        device (torch.device, optional): Device for parameters.
        dtype (torch.dtype, optional): Data type for parameters.
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 nonlinearity: str = 'tanh', symbolic_expression: str = 'x',
                 device=None, dtype=None):
        super().__init__(input_size, hidden_size, bias, symbolic_expression,
                         num_chunks=1, device=device, dtype=dtype)
        self.nonlinearity = nonlinearity

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the RNN cell.

        Args:
            input (Tensor): Input tensor of shape (batch, input_size) or (input_size,).
            hx (Tensor, optional): Initial hidden state. If not provided, zeros.

        Returns:
            Tensor: Next hidden state.
        """
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size,
                             dtype=input.dtype, device=input.device)
        elif not is_batched:
            hx = hx.unsqueeze(0)

        ih_out = self.ih(input)
        hh_out = self.hh(hx)
        out = ih_out + hh_out
        if self.nonlinearity == 'tanh':
            out = torch.tanh(out)
        elif self.nonlinearity == 'relu':
            out = F.relu(out)
        else:
            raise ValueError(f"Unknown nonlinearity: {self.nonlinearity}")

        if not is_batched:
            out = out.squeeze(0)
        return out


class TNLSTMCell(TNRNNCellBase):
    """
    An LSTM cell with custom neuron aggregation.

    This is the equivalent of :class:`torch.nn.LSTMCell` where both linear
    transformations use :class:`TNLinear` with the given symbolic expression.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden features.
        bias (bool): If True, adds a bias. Default: True.
        symbolic_expression (str): Basis function expression. Default: 'x'.
        device (torch.device, optional): Device for parameters.
        dtype (torch.dtype, optional): Data type for parameters.
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 symbolic_expression: str = 'x', device=None, dtype=None):
        super().__init__(input_size, hidden_size, bias, symbolic_expression,
                         num_chunks=4, device=device, dtype=dtype)

    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the LSTM cell.

        Args:
            input (Tensor): Input tensor of shape (batch, input_size) or (input_size,).
            hx (Tuple[Tensor, Tensor], optional): Tuple of (hidden, cell) states.
                If not provided, both are zeros.

        Returns:
            Tuple[Tensor, Tensor]: (new_hidden, new_cell).
        """
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            h0 = torch.zeros(input.size(0), self.hidden_size,
                             dtype=input.dtype, device=input.device)
            c0 = torch.zeros(input.size(0), self.hidden_size,
                             dtype=input.dtype, device=input.device)
            hx = (h0, c0)
        else:
            h, c = hx
            if not is_batched:
                h = h.unsqueeze(0)
                c = c.unsqueeze(0)
            hx = (h, c)

        h, c = hx
        gates = self.ih(input) + self.hh(h)
        i, f, g, o = torch.chunk(gates, 4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        if not is_batched:
            h_new = h_new.squeeze(0)
            c_new = c_new.squeeze(0)
        return h_new, c_new


class TNGRUCell(TNRNNCellBase):
    """
    A GRU cell with custom neuron aggregation.

    This is the equivalent of :class:`torch.nn.GRUCell` where both linear
    transformations use :class:`TNLinear` with the given symbolic expression.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden features.
        bias (bool): If True, adds a bias. Default: True.
        symbolic_expression (str): Basis function expression. Default: 'x'.
        device (torch.device, optional): Device for parameters.
        dtype (torch.dtype, optional): Data type for parameters.
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 symbolic_expression: str = 'x', device=None, dtype=None):
        super().__init__(input_size, hidden_size, bias, symbolic_expression,
                         num_chunks=3, device=device, dtype=dtype)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the GRU cell.

        Args:
            input (Tensor): Input tensor of shape (batch, input_size) or (input_size,).
            hx (Tensor, optional): Initial hidden state. If not provided, zeros.

        Returns:
            Tensor: Next hidden state.
        """
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size,
                             dtype=input.dtype, device=input.device)
        elif not is_batched:
            hx = hx.unsqueeze(0)

        gates = self.ih(input) + self.hh(hx)
        r, z, n = torch.chunk(gates, 3, dim=-1)
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        n = torch.tanh(n)

        h_new = (1 - z) * n + z * hx

        if not is_batched:
            h_new = h_new.squeeze(0)
        return h_new