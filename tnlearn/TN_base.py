import torch
import torch.nn as nn
import math
from sympy import symbols, sympify, Poly, sin, series

class TN_layer(nn.Module):
    def __init__(self, in_features: int, out_features: int, symbolic_expression: str, bias: bool = True):
        super(TN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.neuron = symbolic_expression
        self.poly_terms, self.sine_terms = self._parse_expression(symbolic_expression)
        self.poly_weights = nn.ParameterList([nn.Parameter(torch.Tensor(out_features, in_features)) 
                                           for _ in self.poly_terms])
        self.sine_weights = nn.ParameterList([nn.Parameter(torch.Tensor(out_features, in_features)) 
                                          for _ in self.sine_terms])
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.poly_weights:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        for weight in self.sine_weights:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _parse_expression(self, expression):
        """Parse a symbolic expression to extract polynomial and sine terms."""
        x = symbols('x')
        expression = expression.replace('@', '*')       
        expr = sympify(expression)
        sine_terms = []
        poly_expr = expr
        
        if 'sin' in expression:
            for term in expr.as_ordered_terms():
                if sin(x) in term.atoms(sin):
                    coeff = term / sin(x)
                    sine_terms.append({'coefficient': float(coeff), 'function': 'sin'})
                    poly_expr = poly_expr - term

        try:
            poly = Poly(poly_expr, x)
            coeffs = poly.all_coeffs()
            degrees = list(reversed(range(poly.degree() + 1))) 
            
            poly_terms = []
            for coeff, exponent in zip(coeffs, degrees):
                coefficient = float(coeff)
                poly_terms.append({'coefficient': coefficient, 'exponent': exponent})
        except:
            if poly_expr.is_constant():
                poly_terms = [{'coefficient': float(poly_expr), 'exponent': 0}]
            else:
                poly_terms = []
        
        return poly_terms, sine_terms

    def forward(self, x):
        result = 0
        batch_size = x.size(0)

        if x.dim() == 2:  # (batch_size, in_features)
            for i, term in enumerate(self.poly_terms):
                exponent = term['exponent']
                if exponent == 0:
                    term_output = torch.ones((batch_size, self.out_features), device=x.device)
                else:
                    term_output = nn.functional.linear(x, self.poly_weights[i], None)
                    if exponent > 1:
                        term_output = term_output ** exponent
                
                # Apply coefficient
                term_output = term_output * term['coefficient']
                result += term_output
            
            for i, term in enumerate(self.sine_terms):
                inner_value = nn.functional.linear(x, self.sine_weights[i], None)
                if term['function'] == 'sin':
                    term_output = torch.sin(inner_value)
                term_output = term_output * term['coefficient']
                result += term_output
                
        elif x.dim() == 3:  # (batch_size, num_patches, hidden_dim)
            num_patches = x.size(1)
            
            for i, term in enumerate(self.poly_terms):
                exponent = term['exponent']
                if exponent == 0:
                    term_output = torch.ones((batch_size, num_patches, self.out_features), device=x.device)
                else:
                    input_reshape = x.view(batch_size * num_patches, -1)
                    term_output = nn.functional.linear(input_reshape, self.poly_weights[i], None)
                    term_output = term_output.view(batch_size, num_patches, self.out_features)
                    if exponent > 1:
                        term_output = term_output ** exponent
                

                term_output = term_output * term['coefficient']
                result += term_output
            
            for i, term in enumerate(self.sine_terms):
                input_reshape = x.view(batch_size * num_patches, -1)
                inner_value = nn.functional.linear(input_reshape, self.sine_weights[i], None)
                inner_value = inner_value.view(batch_size, num_patches, self.out_features)

                if term['function'] == 'sin':
                    term_output = torch.sin(inner_value)
                
                term_output = term_output * term['coefficient']
                result += term_output

        if self.bias is not None:
            result += self.bias

        return result


"""
class TNConvLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 bias: bool = True):
        super(TNConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Define the learnable weights for the polynomial convolution
        self.weight_x2 = Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))  # Adjusted for 2D
        self.weight_x = Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))   # Adjusted for 2D
        self.weight_x3 = Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias_x = Parameter(torch.empty(out_channels))
            self.bias_x2 = Parameter(0.001*torch.ones(out_channels))

        else:
            self.register_parameter('bias_x', None)
            self.register_parameter('bias_x2', None)

        
        # Initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.weight_x, mean=0,
                        std=np.sqrt(0.25 / (self.weight_x.shape[1] * np.prod(self.weight_x.shape[2:]))) * 8)
        # init.normal_(self.weight_x2, mean=0,
        #              std=np.sqrt(0.25 / (self.weight_x.shape[1] * np.prod(self.weight_x.shape[2:]))) * 8)
        # init.kaiming_uniform_(self.weight_x3, a=math.sqrt(5))
        init.constant_(self.weight_x2, 0.001)
        # init.kaiming_uniform_(self.weight_x2, a=math.sqrt(5))
        # init.kaiming_uniform_(self.weight_x, a=math.sqrt(5))
        # self.weight_x =  0.3106 * self.weight_x
        if self.bias_x is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_x)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_x, -bound, bound)


    def forward(self, x):
        # Apply polynomial convolution (now using conv2d)
        # conv_x3 = F.conv2d(torch.pow(x, 3), self.weight_x2, None, self.stride, self.padding)
        conv_x2 = F.conv2d(torch.sin(x), self.weight_x2, self.bias_x2, self.stride, self.padding)
        conv_x = F.conv2d(x, self.weight_x, self.bias_x, self.stride, self.padding)
        
        output =  conv_x + conv_x2
        # if self.bias is not None:
        #     output += self.bias.view(1, -1, 1, 1)  # Adjusted for 4D input (batch_size, channels, height, width)
        
        return output
    

"""