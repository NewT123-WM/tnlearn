"""
Program name: Custom Neuron Layer Implementation
Purpose description: This module implements the `CustomNeuronLayer` class in the 'tnlearn'
                     package. It encapsulates the creation of a custom neural network layer
                     based on a symbolic mathematical expression, allowing the instantiation
                     of neural networks with symbolic neurons for both regression and
                     classification tasks. It inherits from PyTorch's nn.Module, leveraging
                     deep learning frameworks for gradient-based optimization.
Last revision date: February 27, 2024
Known Issues: None identified at the time of the last revision.
Note: This overview presupposes that the user has basic understanding of deep learning
      concepts and is facile with PyTorch's core principles and operations.
"""

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
import re
from torch.nn import functional as F


class CustomNeuronLayer(nn.Module):
    # Initialize the custom layer with given input and output features
    # and the symbolic expression that defines neuron functionality
    def __init__(self, in_features: int, out_features: int, symbolic_expression: str, bias: bool = True):
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

    # Resets the layer's parameters by initializing weights and bias
    def reset_parameters(self) -> None:

        # Initialize weights using Kaiming uniform initialization
        for i in range(self.number):
            exec('init.kaiming_uniform_(self.weight{}, a=math.sqrt(5))'.format(i))

        # Initialize bias with uniform distribution if bias is not None
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight0)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    # Extract data terms from the symbolic expression
    def xdata(self):
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

    # Defines the forward pass using the symbolic expression
    def forward(self, x):
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
