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

# ---------- 全局 eval 环境（用于表达式解析） ----------
_EVAL_GLOBALS = {
    'torch': torch,
    'np': __import__('numpy'),
    'math': math,
    'F': F,
}


def _parse_expression(expr: str):
    """解析符号表达式，返回基函数列表（可调用函数）"""
    expr = expr.replace(' ', '')
    # 简单拆分：按顶层 + 和 - 分割（忽略括号内）
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
            # 若含有 '@'，只取后面的部分
            if '@' in t:
                var_expr = t.split('@', 1)[1]
            else:
                var_expr = t
            # 编译为可调用函数
            try:
                fn = eval('lambda x: ' + var_expr, _EVAL_GLOBALS)
            except Exception as e:
                print(f"Warning: eval failed for '{var_expr}', using identity. Error: {e}")
                fn = lambda x: x
            funcs.append(fn)
    if not funcs:
        funcs = [lambda x: x]   # 默认线性
    return funcs


def _augment_input(x: Tensor, funcs: list):
    """对输入张量 x 应用所有基函数，并沿最后一个维度拼接"""
    augmented = [func(x) for func in funcs]
    return torch.cat(augmented, dim=-1)


# ---------- 多层 RNN 基类（基于特征增广 + 原生 RNN） ----------
class TNRNNBase(nn.Module):
    """基类：使用特征增广方法复用 `_VF` 加速"""
    __constants__ = ['input_size', 'hidden_size', 'num_layers', 'bias',
                     'batch_first', 'dropout', 'bidirectional', 'neuron_expression']

    def __init__(self,
                 mode: str,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 bias: bool = True,
                 batch_first: bool = False,
                 dropout: float = 0.0,
                 bidirectional: bool = False,
                 neuron_expression: str = 'x',
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
        self.neuron_expression = neuron_expression

        # 解析表达式，得到基函数列表
        self.funcs = _parse_expression(neuron_expression)
        self.num_funcs = len(self.funcs)
        # 增广后的输入维度
        self.augmented_input_size = input_size * self.num_funcs

        # 创建内部原生 RNN 模块
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
        """对输入进行特征增广"""
        # 如果 batch_first=True，x 形状为 (batch, seq, feature)，否则 (seq, batch, feature)
        # 我们始终在最后一维操作
        augmented = [func(x) for func in self.funcs]
        return torch.cat(augmented, dim=-1)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None):
        # 增广输入
        aug_input = self._augment(input)
        # 调用原生 RNN
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
        if self.neuron_expression != 'x':
            s += f', neuron_expression={self.neuron_expression}'
        return s

    def __getstate__(self):
        # 处理 lambda 序列化：移除 funcs，因为它是动态生成的
        state = self.__dict__.copy()
        state.pop('funcs', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # 重新构建 funcs
        self.funcs = _parse_expression(self.neuron_expression)


# ---------- 具体 RNN 类 ----------
class TNRNN(TNRNNBase):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 nonlinearity: str = 'tanh', bias: bool = True, batch_first: bool = False,
                 dropout: float = 0.0, bidirectional: bool = False,
                 neuron_expression: str = 'x', device=None, dtype=None):
        # 注意：RNN 有 nonlinearity 参数，但我们使用增广方式，原生 nn.RNN 已包含该参数
        super().__init__('RNN', input_size, hidden_size, num_layers, bias,
                         batch_first, dropout, bidirectional, neuron_expression,
                         device, dtype)
        # 将内部 rnn 的 nonlinearity 设置为指定值
        self.rnn.nonlinearity = nonlinearity


class TNLSTM(TNRNNBase):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 bias: bool = True, batch_first: bool = False,
                 dropout: float = 0.0, bidirectional: bool = False,
                 neuron_expression: str = 'x', device=None, dtype=None):
        super().__init__('LSTM', input_size, hidden_size, num_layers, bias,
                         batch_first, dropout, bidirectional, neuron_expression,
                         device, dtype)


class TNGRU(TNRNNBase):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 bias: bool = True, batch_first: bool = False,
                 dropout: float = 0.0, bidirectional: bool = False,
                 neuron_expression: str = 'x', device=None, dtype=None):
        super().__init__('GRU', input_size, hidden_size, num_layers, bias,
                         batch_first, dropout, bidirectional, neuron_expression,
                         device, dtype)


# ---------- Cell 版本（基于 TNLinear，用于逐时间步计算） ----------
class TNRNNCellBase(nn.Module):
    """基类：所有自定义 RNN Cell 的父类，使用 TNLinear 作为权重"""
    __constants__ = ['input_size', 'hidden_size', 'bias', 'neuron_expression']

    def __init__(self, input_size: int, hidden_size: int, bias: bool,
                 neuron_expression: str = 'x', num_chunks: int = 1,
                 device=None, dtype=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.neuron_expression = neuron_expression
        self.num_chunks = num_chunks

        # 输入→隐藏 线性层
        self.ih = TNLinear(
            in_features=input_size,
            out_features=num_chunks * hidden_size,
            symbolic_expression=neuron_expression,
            bias=bias,
            device=device,
            dtype=dtype
        )
        # 隐藏→隐藏 线性层
        self.hh = TNLinear(
            in_features=hidden_size,
            out_features=num_chunks * hidden_size,
            symbolic_expression=neuron_expression,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """参数初始化（由 TNLinear 内部完成，无需额外操作）"""
        pass

    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.neuron_expression != 'x':
            s += ', neuron_expression={neuron_expression}'
        return s.format(**self.__dict__)


class TNRNNCell(TNRNNCellBase):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 nonlinearity: str = 'tanh', neuron_expression: str = 'x',
                 device=None, dtype=None):
        super().__init__(input_size, hidden_size, bias, neuron_expression,
                         num_chunks=1, device=device, dtype=dtype)
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
                 neuron_expression: str = 'x', device=None, dtype=None):
        super().__init__(input_size, hidden_size, bias, neuron_expression,
                         num_chunks=4, device=device, dtype=dtype)

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
                 neuron_expression: str = 'x', device=None, dtype=None):
        super().__init__(input_size, hidden_size, bias, neuron_expression,
                         num_chunks=3, device=device, dtype=dtype)

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