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


class CustomNeuronLayer(nn.Module):
    r"""Build a neural network model of a custom architecture."""
    def __init__(self, in_features: int, out_features: int, symbolic_expression: str, bias: bool = True):
        r"""Initialize the custom layer with given input and output features
        and the symbolic expression that defines neuron functionality.

        Args:
            in_features:
            out_features:
            symbolic_expression:
            bias:
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
            exec('self.weight{} = Parameter(torch.Tensor(out_features, in_features))'.format(i))

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
            exec('init.kaiming_uniform_(self.weight{}, a=math.sqrt(5))'.format(i))

        # Initialize bias with uniform distribution if bias is not None
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight0)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def xdata(self):
        r"""Extract data terms from the symbolic expression."""
        temp = self.neuron.replace(' ', '')
        temp = re.split('\+|-', temp)
        xx = []
        for s in temp:
            if '@' in s:
                out = s[s.find('@') + 1:]
                xx.append(out)
            elif 'x' in s:
                out = s
                xx.append(out)
            else:
                pass
        return xx

    def forward(self, x):
        r"""Defines the forward pass using the symbolic expression.

        Returns:
            The calculation of a single neuron.
        """
        # Collect variable data blocks from the symbolic expression
        xlist = self.xdata()
        su = 0
        loc = locals()
        for i in range(self.number):
            # Execute a linear operation using the weights and input data
            exec('su += F.linear(eval(xlist[i]), self.weight{}, None)'.format(i))
            # Keep accumulating the output
            su = loc['su']

        # Add bias to the accumulated result if bias exists and return the output
        out = su + self.bias if self.bias is not None else su
        return out
