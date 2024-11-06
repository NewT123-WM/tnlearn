import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from sympy import sympify, symbols, Poly


class CustomNeuronLayer_2(nn.Module):
    def __init__(self, in_features: int, out_features: int, symbolic_expression: str, bias: bool = True):
        super(CustomNeuronLayer_2, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.neuron = symbolic_expression

        self.terms = self._parse_expression(symbolic_expression)

        self.weights = nn.ParameterList([nn.Parameter(torch.Tensor(out_features, in_features)) for _ in self.terms])

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.weights:
            init.kaiming_uniform_(weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def _parse_expression(self, expression):
        x = symbols('x')
        expression = expression.replace('@', '*')

        expr = sympify(expression)
        poly_expr = Poly(expr, x)
        coeffs = poly_expr.all_coeffs()
        degrees = [monom[0] for monom in poly_expr.monoms()]

        terms = []
        for coeff, exponent in zip(coeffs, degrees):
            coefficient = float(coeff)
            terms.append({'coefficient': coefficient, 'exponent': exponent})

        return terms

    def forward(self, x):
        su = 0
        for i, term in enumerate(self.terms):
            exponent = term['exponent']
            if exponent == 0:
                term_output =  torch.ones((x.size(0), self.out_features), device=x.device)
            else:
                input_tensor = x ** exponent
                term_output = F.linear(input_tensor, self.weights[i], None)

            su += term_output

        if self.bias is not None:
            su += self.bias
        return su

class ModifiedResNet_2(nn.Module):
    def __init__(self, original_resnet, out_features, symbolic_expression):
        super(ModifiedResNet_2, self).__init__()

        self.stage1 = nn.Sequential(*list(original_resnet.children())[:5])
        self.stage2 = nn.Sequential(*list(original_resnet.children())[5:6])
        self.stage3 = nn.Sequential(*list(original_resnet.children())[6:7])
        self.stage4 = nn.Sequential(*list(original_resnet.children())[7][:2])

        with torch.no_grad():
            sample_input = torch.randn(1, 3, 224, 224)
            sample_output = self.stage4(self.stage3(self.stage2(self.stage1(sample_input))))
            in_features = sample_output.view(1, -1).size(1)

        self.poly_layers = nn.Sequential(
            CustomNeuronLayer_2(in_features, 512, symbolic_expression),
            CustomNeuronLayer_2(512, 512, symbolic_expression),
            CustomNeuronLayer_2(512, out_features, symbolic_expression)
        )

    def forward(self, x):
        x = self.stage1(x)
        #print("After stage1:", x.abs().mean().item())
        x = self.stage2(x)
        #print("After stage2:", x.abs().mean().item())
        x = self.stage3(x)
        x = self.stage4(x)
        #print("After stage3:", x.abs().mean().item())
        x = x.view(x.size(0), -1)
        x = self.poly_layers(x)
        #print("After poly_layers:", x.abs().mean().item())
        return x