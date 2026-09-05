"""
Recurrent neural network layers with custom neuron aggregation supporting inner‑product cross terms.

Two modes are available:
- base (default): Uses custom cell implementation with TNLinear and manual loop.
- legacy: Uses input augmentation + native PyTorch RNN (RNN, LSTM, GRU) modules.

Copyright (c) 2026 Tieyun LI. All Rights Reserved.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, init
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from .TNlinear import TNLinear

__all__ = [
    'TNRNNBase', 'TNRNN', 'TNLSTM', 'TNGRU',
    'TNRNNCell', 'TNLSTMCell', 'TNGRUCell'
]

# ---------- Helper functions (shared) ----------

def _stack_states(states: List[Tensor], batch_first: bool) -> Tensor:
    if batch_first:
        return torch.stack(states, dim=1)
    else:
        return torch.stack(states, dim=0)


def _unbind_states(output: Tensor, batch_first: bool) -> List[Tensor]:
    if batch_first:
        return [output[:, t, :] for t in range(output.size(1))]
    else:
        return [output[t, :, :] for t in range(output.size(0))]


def _reverse_sequence(inputs: Tensor, seq_lengths: Optional[Tensor] = None,
                      batch_first: bool = False) -> Tensor:
    if seq_lengths is None:
        if batch_first:
            return inputs.flip(dims=(1,))
        else:
            return inputs.flip(dims=(0,))
    else:
        return inputs.flip(dims=(1,) if batch_first else (0,))


# ---------- Legacy mode helpers (expression parsing) ----------
_EVAL_GLOBALS = {
    'torch': torch,
    'np': __import__('numpy'),
    'math': __import__('math'),
    'F': F,
}


def _parse_expression(expr: str) -> list:
    """Parse symbolic expression and return list of callable basis functions."""
    expr = expr.replace(' ', '')
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
            if '@' in t:
                var_expr = t.split('@', 1)[1]
            else:
                var_expr = t
            try:
                fn = eval('lambda x: ' + var_expr, _EVAL_GLOBALS)
            except Exception as e:
                print(f"Legacy: eval failed for '{var_expr}', using identity. Error: {e}")
                fn = lambda x: x
            funcs.append(fn)
    if not funcs:
        funcs = [lambda x: x]
    return funcs


def _augment_input(x: Tensor, funcs: list) -> Tensor:
    """Apply all basis functions and concatenate along last dimension."""
    augmented = [func(x) for func in funcs]
    return torch.cat(augmented, dim=-1)


# ========== Multi‑layer RNN base ==========

class _TNRNNBase(nn.Module):
    __constants__ = ['input_size', 'hidden_size', 'num_layers', 'bias',
                     'batch_first', 'dropout', 'bidirectional', 'symbolic_expression',
                     'mode', 'rnn_type']

    def __init__(self, cell_factory, mode: str, rnn_type: str,
                 input_size: int, hidden_size: int, num_layers: int = 1,
                 bias: bool = True, batch_first: bool = False,
                 dropout: float = 0.0, bidirectional: bool = False,
                 symbolic_expression: str = 'x',
                 device=None, dtype=None,
                 already_parametrized: bool = False):  # NEW
        super().__init__()
        self.mode = mode.lower()
        if self.mode not in ('base', 'legacy'):
            raise ValueError("mode must be 'base' or 'legacy'")
        self.rnn_type = rnn_type  # 'RNN', 'LSTM', or 'GRU'

        self.original_input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.symbolic_expression = symbolic_expression
        self.device = device
        self.dtype = dtype
        self.already_parametrized = already_parametrized  # NEW

        self.num_directions = 2 if bidirectional else 1
        self.num_cells = num_layers * self.num_directions

        if self.mode == 'legacy':
            # ---------- Legacy mode: input augmentation + native RNN ----------
            self.funcs = _parse_expression(symbolic_expression)
            self.num_funcs = len(self.funcs)
            self.augmented_input_size = input_size * self.num_funcs

            rnn_cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[rnn_type]
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
            if rnn_type == 'RNN' and hasattr(self, 'nonlinearity'):
                self.rnn.nonlinearity = self.nonlinearity

        else:
            # ---------- Base mode: custom cells and manual loop ----------
            self.cells = nn.ModuleList()
            for layer in range(num_layers):
                layer_input_size = input_size if layer == 0 else hidden_size * (2 if bidirectional else 1)
                for direction in range(2 if bidirectional else 1):
                    cell = cell_factory(
                        layer_input_size, hidden_size, bias,
                        symbolic_expression, device, dtype,
                        already_parametrized=already_parametrized  # NEW
                    )
                    self.cells.append(cell)

            if dropout > 0.0:
                self.dropout_layer = nn.Dropout(dropout)
            else:
                self.dropout_layer = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    def _default_initial_state(self, batch_size: int, device: torch.device):
        num = self.num_layers * self.num_directions
        if self.mode == 'legacy':
            return None
        else:
            if self.rnn_type == 'LSTM':
                h = torch.zeros(num, batch_size, self.hidden_size, device=device, dtype=self.dtype)
                c = torch.zeros(num, batch_size, self.hidden_size, device=device, dtype=self.dtype)
                return (h, c)
            else:
                return torch.zeros(num, batch_size, self.hidden_size, device=device, dtype=self.dtype)

    def _prepare_state(self, state, batch_size: int, device: torch.device):
        if self.mode == 'legacy':
            return state
        else:
            if self.rnn_type == 'LSTM':
                h, c = state
                if h.dim() == 2:
                    h = h.unsqueeze(0)
                    c = c.unsqueeze(0)
                    h = h.expand(self.num_layers * self.num_directions, -1, -1).contiguous()
                    c = c.expand(self.num_layers * self.num_directions, -1, -1).contiguous()
                elif h.dim() == 3 and h.size(0) == 1:
                    h = h.expand(self.num_layers * self.num_directions, -1, -1).contiguous()
                    c = c.expand(self.num_layers * self.num_directions, -1, -1).contiguous()
                return (h, c)
            else:
                if state.dim() == 2:
                    state = state.unsqueeze(0)
                    state = state.expand(self.num_layers * self.num_directions, -1, -1).contiguous()
                return state

    def _forward_impl_base(self, input: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, Tensor]]],
                           seq_lengths: Optional[Tensor] = None) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor, Tensor]]]:
        """Base mode: manual loop with cells."""
        if self.batch_first:
            input = input.transpose(0, 1)
        seq_len, batch_size, _ = input.size()

        if state is None:
            state = self._default_initial_state(batch_size, input.device)
        else:
            state = self._prepare_state(state, batch_size, input.device)

        if self.rnn_type == 'LSTM':
            h0, c0 = state
            h_list = torch.unbind(h0, dim=0)
            c_list = torch.unbind(c0, dim=0)
            h_states = list(h_list)
            c_states = list(c_list)
        else:
            h_list = torch.unbind(state, dim=0)
            h_states = list(h_list)
            c_states = None

        layer_input = input

        for layer in range(self.num_layers):
            dir_outputs = []
            for direction in range(self.num_directions):
                cell_idx = layer * self.num_directions + direction
                cell = self.cells[cell_idx]

                if self.rnn_type == 'LSTM':
                    h = h_states[cell_idx]
                    c = c_states[cell_idx]
                    state_t = (h, c)
                else:
                    h = h_states[cell_idx]
                    state_t = h

                time_outputs = []
                if direction == 0:  # forward
                    for t in range(seq_len):
                        x_t = layer_input[t]
                        if self.rnn_type == 'LSTM':
                            h, c = cell(x_t, state_t)
                            state_t = (h, c)
                        else:
                            state_t = cell(x_t, state_t)
                            h = state_t
                        time_outputs.append(h)
                    if self.rnn_type == 'LSTM':
                        h_states[cell_idx] = h
                        c_states[cell_idx] = c
                    else:
                        h_states[cell_idx] = h
                    dir_outputs.append(time_outputs)
                else:  # backward
                    rev_outputs = []
                    for t in range(seq_len - 1, -1, -1):
                        x_t = layer_input[t]
                        if self.rnn_type == 'LSTM':
                            h, c = cell(x_t, state_t)
                            state_t = (h, c)
                        else:
                            state_t = cell(x_t, state_t)
                            h = state_t
                        rev_outputs.append(h)
                    time_outputs = rev_outputs[::-1]
                    if self.rnn_type == 'LSTM':
                        h_states[cell_idx] = h
                        c_states[cell_idx] = c
                    else:
                        h_states[cell_idx] = h
                    dir_outputs.append(time_outputs)

            if self.bidirectional:
                combined = [torch.cat([f, b], dim=-1) for f, b in zip(dir_outputs[0], dir_outputs[1])]
            else:
                combined = dir_outputs[0]

            if self.dropout_layer is not None and layer < self.num_layers - 1:
                combined = [self.dropout_layer(x) for x in combined]

            layer_input = torch.stack(combined, dim=0)
            if layer == self.num_layers - 1:
                final_output = layer_input

        if self.batch_first:
            output = final_output.transpose(0, 1)
        else:
            output = final_output

        if self.rnn_type == 'LSTM':
            h_final = torch.stack(h_states, dim=0).view(self.num_layers * self.num_directions,
                                                         batch_size, self.hidden_size)
            c_final = torch.stack(c_states, dim=0).view(self.num_layers * self.num_directions,
                                                         batch_size, self.hidden_size)
            final_state = (h_final, c_final)
        else:
            h_final = torch.stack(h_states, dim=0).view(self.num_layers * self.num_directions,
                                                         batch_size, self.hidden_size)
            final_state = h_final

        return output, final_state

    def _forward_impl_legacy(self, input: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None):
        """Legacy mode: augment input, delegate to native RNN."""
        aug_input = _augment_input(input, self.funcs)
        return self.rnn(aug_input, state)

    def forward(self, input: Union[Tensor, PackedSequence],
                state: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None):
        if self.mode == 'legacy':
            if isinstance(input, PackedSequence):
                input_unpacked, lengths = pad_packed_sequence(input, batch_first=self.batch_first)
                output, final_state = self._forward_impl_legacy(input_unpacked, state)
                output_packed = pack_padded_sequence(output, lengths.cpu(),
                                                      batch_first=self.batch_first,
                                                      enforce_sorted=False)
                return output_packed, final_state
            else:
                return self._forward_impl_legacy(input, state)
        else:
            if isinstance(input, PackedSequence):
                input_unpacked, lengths = pad_packed_sequence(input, batch_first=self.batch_first)
                output, final_state = self._forward_impl_base(input_unpacked, state)
                output_packed = pack_padded_sequence(output, lengths.cpu(),
                                                      batch_first=self.batch_first,
                                                      enforce_sorted=False)
                return output_packed, final_state
            else:
                return self._forward_impl_base(input, state)

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
            s += f', symbolic_expression="{self.symbolic_expression}"'
        if self.mode != 'base':
            s += f', mode="{self.mode}"'
        if self.already_parametrized != False:  # NEW
            s += f', already_parametrized={self.already_parametrized}'
        return s

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.mode == 'legacy':
            state.pop('funcs', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.mode == 'legacy':
            self.funcs = _parse_expression(self.symbolic_expression)


# ========== Concrete RNN classes ==========

class TNRNN(_TNRNNBase):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 nonlinearity: str = 'tanh', bias: bool = True, batch_first: bool = False,
                 dropout: float = 0.0, bidirectional: bool = False,
                 symbolic_expression: str = 'x', device=None, dtype=None,
                 mode: str = 'base',
                 already_parametrized: bool = False):  # NEW
        self.nonlinearity = nonlinearity
        def cell_factory(in_size, h_size, b, sym_expr, dev, dtyp, ap):  # added ap
            return TNRNNCell(in_size, h_size, bias=b, nonlinearity=nonlinearity,
                             symbolic_expression=sym_expr, device=dev, dtype=dtyp, mode=mode,
                             already_parametrized=ap)  # pass ap
        super().__init__(cell_factory, mode, 'RNN', input_size, hidden_size, num_layers,
                         bias, batch_first, dropout, bidirectional,
                         symbolic_expression, device, dtype,
                         already_parametrized=already_parametrized)  # NEW


class TNLSTM(_TNRNNBase):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 bias: bool = True, batch_first: bool = False,
                 dropout: float = 0.0, bidirectional: bool = False,
                 symbolic_expression: str = 'x', device=None, dtype=None,
                 mode: str = 'base',
                 already_parametrized: bool = False):  # NEW
        def cell_factory(in_size, h_size, b, sym_expr, dev, dtyp, ap):  # added ap
            return TNLSTMCell(in_size, h_size, bias=b,
                              symbolic_expression=sym_expr, device=dev, dtype=dtyp, mode=mode,
                              already_parametrized=ap)  # pass ap
        super().__init__(cell_factory, mode, 'LSTM', input_size, hidden_size, num_layers,
                         bias, batch_first, dropout, bidirectional,
                         symbolic_expression, device, dtype,
                         already_parametrized=already_parametrized)  # NEW


class TNGRU(_TNRNNBase):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 bias: bool = True, batch_first: bool = False,
                 dropout: float = 0.0, bidirectional: bool = False,
                 symbolic_expression: str = 'x', device=None, dtype=None,
                 mode: str = 'base',
                 already_parametrized: bool = False):  # NEW
        def cell_factory(in_size, h_size, b, sym_expr, dev, dtyp, ap):  # added ap
            return TNGRUCell(in_size, h_size, bias=b,
                             symbolic_expression=sym_expr, device=dev, dtype=dtyp, mode=mode,
                             already_parametrized=ap)  # pass ap
        super().__init__(cell_factory, mode, 'GRU', input_size, hidden_size, num_layers,
                         bias, batch_first, dropout, bidirectional,
                         symbolic_expression, device, dtype,
                         already_parametrized=already_parametrized)  # NEW


# ---------- Public alias for backward compatibility ----------
class TNRNNBase(_TNRNNBase):
    pass


# ========== Cell classes ==========

class TNRNNCellBase(nn.Module):
    __constants__ = ['input_size', 'hidden_size', 'bias', 'symbolic_expression', 'mode']

    def __init__(self, input_size: int, hidden_size: int, bias: bool,
                 symbolic_expression: str = 'x', num_chunks: int = 1,
                 device=None, dtype=None, mode: str = 'base',
                 already_parametrized: bool = False):  # NEW
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.symbolic_expression = symbolic_expression
        self.num_chunks = num_chunks
        self.mode = mode.lower()
        self.already_parametrized = already_parametrized  # NEW

        self.ih = TNLinear(
            in_features=input_size,
            out_features=num_chunks * hidden_size,
            symbolic_expression=symbolic_expression,
            bias=bias,
            device=device,
            dtype=dtype,
            mode=mode,
            already_parametrized=already_parametrized  # NEW
        )
        self.hh = TNLinear(
            in_features=hidden_size,
            out_features=num_chunks * hidden_size,
            symbolic_expression=symbolic_expression,
            bias=bias,
            device=device,
            dtype=dtype,
            mode=mode,
            already_parametrized=already_parametrized  # NEW
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    def extra_repr(self) -> str:
        s = f'{self.input_size}, {self.hidden_size}'
        if self.bias is not True:
            s += f', bias={self.bias}'
        if self.symbolic_expression != 'x':
            s += f', symbolic_expression={self.symbolic_expression}'
        if self.mode != 'base':
            s += f', mode={self.mode}'
        if self.already_parametrized != False:  # NEW
            s += f', already_parametrized={self.already_parametrized}'
        return s


class TNRNNCell(TNRNNCellBase):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 nonlinearity: str = 'tanh', symbolic_expression: str = 'x',
                 device=None, dtype=None, mode: str = 'base',
                 already_parametrized: bool = False):  # NEW
        super().__init__(input_size, hidden_size, bias, symbolic_expression,
                         num_chunks=1, device=device, dtype=dtype, mode=mode,
                         already_parametrized=already_parametrized)  # NEW
        self.nonlinearity = nonlinearity

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
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
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 symbolic_expression: str = 'x', device=None, dtype=None,
                 mode: str = 'base',
                 already_parametrized: bool = False):  # NEW
        super().__init__(input_size, hidden_size, bias, symbolic_expression,
                         num_chunks=4, device=device, dtype=dtype, mode=mode,
                         already_parametrized=already_parametrized)  # NEW

    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
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
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 symbolic_expression: str = 'x', device=None, dtype=None,
                 mode: str = 'base',
                 already_parametrized: bool = False):  # NEW
        super().__init__(input_size, hidden_size, bias, symbolic_expression,
                         num_chunks=3, device=device, dtype=dtype, mode=mode,
                         already_parametrized=already_parametrized)  # NEW

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
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