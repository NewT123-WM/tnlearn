"""
MLPRegressor with customizable neuron layers.

This module provides two versions of CustomNeuronLayer:
- BaseCustomNeuronLayer (default): uses SymPy parameterization with InnerProduct.
- LegacyCustomNeuronLayer: imported from tnlearn.neurons (original implementation).

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

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
import sympy as sp
from sympy import symbols, Add, Mul, Pow, sin, cos, exp, simplify, expand, sympify, Symbol, Number
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary
from sklearn.metrics import r2_score

from tnlearn.seeds import random_seed
from tnlearn.activation_function import get_activation_function
from tnlearn.loss_function import get_loss_function
from tnlearn.optimizer import get_optimizer
from tnlearn.base1 import BaseModel1

# Import legacy CustomNeuronLayer from original package
from tnlearn.neurons import CustomNeuronLayer as LegacyCustomNeuronLayer

# Import InnerProduct utilities for base mode
from tnlearn.operator.inner_product import (
    InnerProduct,
    convert_pretty_to_innerproduct,
    parameterize_expression,
    evaluate_expression,
)


# ---------- Base CustomNeuronLayer (new version) ----------
class BaseCustomNeuronLayer(nn.Module):
    """
    Custom neuron layer using SymPy parameterization with InnerProduct.
    """
    def __init__(self, in_features: int, out_features: int, symbolic_expression: str, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.raw_expr = symbolic_expression

        # 1. Convert to InnerProduct format
        expr_str = convert_pretty_to_innerproduct(symbolic_expression)
        local_dict = {'InnerProduct': InnerProduct}
        try:
            sym_expr = sympify(expr_str, locals=local_dict)
        except Exception as e:
            raise ValueError(f"Failed to parse expression: {expr_str}\nError: {e}")

        # Expand and simplify
        sym_expr = expand(sym_expr)
        sym_expr = simplify(sym_expr)

        # 2. Parameterize
        self.param_expr = parameterize_expression(sym_expr, include_bias=True)
        self.param_expr_str = str(self.param_expr)

        # 3. Extract all symbols
        all_symbols = self.param_expr.free_symbols
        x_sym = symbols('x')
        weight_symbols = [sym for sym in all_symbols if sym != x_sym]
        self.w_syms = [str(sym) for sym in weight_symbols if str(sym).startswith('w')]
        self.c_syms = [str(sym) for sym in weight_symbols if str(sym).startswith('c')]
        self.b_syms = [str(sym) for sym in weight_symbols if str(sym).startswith('b')]

        # 4. Create parameters
        self.w_weights = nn.ParameterList()
        self.c_weights = nn.ParameterList()
        self.b_weights = nn.ParameterList()

        for _ in self.w_syms:
            self.w_weights.append(Parameter(torch.Tensor(out_features, in_features)))
        for _ in self.c_syms:
            self.c_weights.append(Parameter(torch.Tensor(out_features, 1)))
        for _ in self.b_syms:
            self.b_weights.append(Parameter(torch.Tensor(out_features, 1)))

        self.w_names = self.w_syms
        self.c_names = self.c_syms
        self.b_names = self.b_syms

        # 5. Bias (kept for compatibility, but b_i are already used)
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for w in self.w_weights:
            init.kaiming_uniform_(w, a=math.sqrt(5))
        for c in self.c_weights:
            init.normal_(c, mean=0.0, std=0.1)
        for b in self.b_weights:
            init.zeros_(b)
        if self.bias is not None:
            if len(self.w_weights) > 0:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.w_weights[0])
            else:
                fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        param_dict = {}
        for name, w in zip(self.w_names, self.w_weights):
            param_dict[name] = w
        for name, c in zip(self.c_names, self.c_weights):
            param_dict[name] = c
        for name, b in zip(self.b_names, self.b_weights):
            param_dict[name] = b

        result = evaluate_expression(self.param_expr, x, param_dict, self.out_features)

        # Numerical stability
        result = torch.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
        result = torch.clamp(result, -1e6, 1e6)

        if self.bias is not None:
            result = result + self.bias
        return result


# ---------- MLPRegressor ----------
class MLPRegressor(BaseModel1):
    def __init__(self,
                 neurons='x',
                 layers_list=None,
                 activation_funcs=None,
                 loss_function=None,
                 optimizer_name='adam',
                 random_state=1,
                 max_iter=300,
                 batch_size=128,
                 lr=0.001,
                 visual=False,
                 visual_interval=100,
                 save=False,
                 fig_path=None,
                 gpu=None,
                 interval=None,
                 scheduler=None,
                 l1_reg=False,
                 l2_reg=False,
                 mode='base',
                 ):
        r"""Construct MLPRegressor with task-based neurons.

        Args:
            neurons (str): Neuronal expression
            layers_list (list): List of neuron counts for each hidden layer
            activation_funcs (str): Activation functions
            loss_function (str): Loss function for the training process
            optimizer_name (str): Name of the optimizer algorithm
            random_state (int): Seed for random number generators for reproducibility
            max_iter (int): Maximum number of training iterations
            batch_size (int): Number of samples per batch during training
            lr (float): Learning rate for the optimizer
            visual (boolean): Boolean indicating if training visualization is to be shown
            save (boolean): Indicates if the training figure should be saved
            fig_path (str or None): Path to save the training figure
            visual_interval (int): Interval at which training visualization is updated
            gpu (int or None): Specifies GPU configuration for training
            interval (int): Interval of screen output during training
            scheduler (dict): Learning rate scheduler
            l1_reg (boolean): L1 regularization term
            l2_reg (boolean): L2 regularization term
            mode (str): Which neuron layer implementation to use:
                       'base'  -> BaseCustomNeuronLayer (SymPy + InnerProduct)
                       'legacy' -> LegacyCustomNeuronLayer (from tnlearn.neurons)
        """
        super(MLPRegressor, self).__init__()

        self.random_state = random_state
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.lr = lr
        self.neurons = neurons
        self.optimizer_name = optimizer_name
        self.save_fig = save
        self.visual = visual
        self.visual_interval = visual_interval
        self.interval = interval
        self.scheduler = scheduler
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.mode = mode
        if self.mode.lower() not in ('base', 'legacy'):
            raise ValueError("mode must be 'base' or 'legacy'")

        if fig_path is None:
            self.fig_path = './'
        else:
            self.fig_path = fig_path

        self.gpu = gpu
        self.device = self.select_device(gpu)

        assert isinstance(self.visual_interval,
                          int) and self.visual_interval, "visual_interval must be a non-zero integer"

        if layers_list is None:
            self.layers_list = [50, 30, 10]
        else:
            self.layers_list = layers_list

        # ----- scikit-learn compatible parameter storage -----
        # Store activation function as string for get_params/set_params
        if activation_funcs is None:
            self.activation_funcs_str = 'relu'
        else:
            if not isinstance(activation_funcs, str):
                raise ValueError("activation_funcs must be a string, got {}".format(type(activation_funcs)))
            self.activation_funcs_str = activation_funcs
        self.activation_funcs = get_activation_function(self.activation_funcs_str)

        # Store loss function as string
        if loss_function is None:
            self.loss_function_str = 'mse'
        else:
            if not isinstance(loss_function, str):
                raise ValueError("loss_function must be a string, got {}".format(type(loss_function)))
            self.loss_function_str = loss_function
        self.loss_function = get_loss_function(self.loss_function_str)

        random_seed(self.random_state)

    def get_params(self, deep=True):
        """Get parameters for this estimator (scikit-learn compatibility)."""
        return {
            'neurons': self.neurons,
            'layers_list': self.layers_list,
            'activation_funcs': self.activation_funcs_str,
            'loss_function': self.loss_function_str,
            'optimizer_name': self.optimizer_name,
            'random_state': self.random_state,
            'max_iter': self.max_iter,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'visual': self.visual,
            'visual_interval': self.visual_interval,
            'save': self.save_fig,
            'fig_path': self.fig_path,
            'gpu': self.gpu,
            'interval': self.interval,
            'scheduler': self.scheduler,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg,
            'mode': self.mode,
        }

    def set_params(self, **params):
        """Set the parameters of this estimator (scikit-learn compatibility)."""
        for key, value in params.items():
            setattr(self, key, value)
        # Re-instantiate activation function if updated
        if 'activation_funcs' in params:
            self.activation_funcs_str = params['activation_funcs']
            self.activation_funcs = get_activation_function(self.activation_funcs_str)
        # Re-instantiate loss function if updated
        if 'loss_function' in params:
            self.loss_function_str = params['loss_function']
            self.loss_function = get_loss_function(self.loss_function_str)
        return self

    def select_device(self, gpu):
        r"""Selects the training device based on the 'gpu' parameter.

        Args:
            gpu: GPU ID.
        """
        if gpu is None:
            return torch.device("cpu")
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Training will default to CPU.")
        if isinstance(gpu, int):
            cuda_device = f'cuda:{gpu}'
            return torch.device(cuda_device)
        elif isinstance(gpu, (list, tuple)):
            for g in gpu:
                if not isinstance(g, int) or g >= torch.cuda.device_count():
                    raise ValueError(f"Invalid GPU index {g}")
            return gpu
        else:
            raise ValueError("Invalid 'gpu' parameter. It should be None, an integer, or a list/tuple of integers.")

    def build_model(self, input_dim, output_dim):
        r"""Constructs the neural network model based on the specified architecture.

        Args:
            input_dim: The input dimension of the network.
            output_dim: The output dimension of the network.

        Returns:
            A fully connected network architecture.
        """
        # Choose the appropriate CustomNeuronLayer class based on mode
        if self.mode.lower() == 'legacy':
            layer_class = LegacyCustomNeuronLayer
        else:
            layer_class = BaseCustomNeuronLayer

        layers = []
        last_dim = input_dim
        for neuron_count in self.layers_list:
            layers.append(layer_class(last_dim, neuron_count, self.neurons))
            last_dim = neuron_count
            layers.append(self.activation_funcs)

        layers.append(nn.Linear(last_dim, output_dim))
        return nn.Sequential(*layers)

    def prepare_data(self, X, y):
        r"""Prepares the input data and splits it into training and validation sets.

        Args:
            X (numpy ndarray): Training data.
            y (numpy ndarray): Label data.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be a NumPy array.")
        if len(X.shape) != 2:
            raise ValueError("X should be a 2D NumPy array.")
        self.input_dim = X.shape[1]

        if not isinstance(y, np.ndarray):
            raise ValueError("y should be a NumPy array.")
        self.output_dim = y.shape[1] if len(y.shape) > 1 else 1

        if not isinstance(X, torch.Tensor):
            self.X = torch.tensor(X, dtype=torch.float32)

        if not isinstance(y, torch.Tensor):
            self.y = torch.tensor(y, dtype=torch.float32)

        trainset = TensorDataset(self.X, self.y)
        self.trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)

    def fit(self, X, y):
        r"""Train the network with training data.

        Args:
            X (numpy ndarray): Training data.
            y (numpy ndarray): Label data.
        """
        self.prepare_data(X, y)
        self.losses = []

        if self.device == torch.device("cpu"):
            self.net = self.build_model(self.input_dim, self.output_dim).to(self.device)
        else:
            self.net = self.build_model(self.input_dim, self.output_dim)
            if isinstance(self.device, list):
                self.net = nn.DataParallel(self.net, device_ids=self.device)
                self.net.to(f'cuda:{self.device[0]}')
                self.device = f'cuda:{self.device[0]}'
            else:
                self.net = self.net.to(self.device)
                self.device = self.device

        self.cost = self.loss_function.to(self.device)
        self.optimizer = get_optimizer(name=self.optimizer_name, parameters=list(self.net.parameters()),
                                       lr=self.lr)

        if self.scheduler is not None:
            scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.scheduler["step_size"],
                                            gamma=self.scheduler["gamma"])

        self.net.train()
        for epoch in range(self.max_iter):
            self.current_epoch = epoch + 1
            running_loss = 0.0
            for inputs, targets in self.trainloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device).reshape(-1, 1)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.cost(outputs, targets)

                if self.l1_reg:
                    l1_loss = sum(p.abs().sum() for p in self.net.parameters())
                    loss += self.l1_reg * l1_loss

                if self.l2_reg:
                    l2_loss = sum(p.pow(2.0).sum() for p in self.net.parameters())
                    loss += self.l2_reg * l2_loss

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            if self.scheduler is not None:
                scheduler.step()

            if self.interval is not None and (epoch + 1) % self.interval == 0:
                print(f'Epoch [{epoch + 1}/{self.max_iter}], Loss: {running_loss:.4f}')

            self.losses.append(running_loss)

            if self.visual and epoch % self.visual_interval == 0:
                self.plot_progress_regression(loss=self.losses)

        if self.save_fig:
            self.regression_savefigure(loss=self.losses, path=self.fig_path)

    def predict(self, X):
        r"""Use a trained model to make predictions.

        Args:
            X (torch.Tensor): Data that needs to be predicted.

        Returns:
            Predicted value
        """
        if not isinstance(X, torch.Tensor):
            X = torch.Tensor(X)

        dataset = TensorDataset(X)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.net.eval()
        predictions = []
        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs[0].to(self.device)
                outputs = self.net(inputs)
                predictions.extend(outputs.cpu().numpy())
        return np.array(predictions).flatten()

    def score(self, X, y):
        r"""Evaluate the coefficient of determination.

        Args:
            X (numpy ndarray): Test data.
            y (numpy ndarray): Label data.

        Returns:
            Coefficient of determination.
        """
        pred = self.predict(X)
        return r2_score(y, pred)

    def count_param(self):
        r"""Print the network structure and output the number of network parameters."""
        summary(self.net, input_size=(self.batch_size, self.input_dim))