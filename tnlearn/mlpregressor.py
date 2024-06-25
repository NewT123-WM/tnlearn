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

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from tnlearn.neurons import CustomNeuronLayer
from tnlearn.seeds import random_seed
from tnlearn.activation_function import get_activation_function
from tnlearn.loss_function import get_loss_function
from tnlearn.optimizer import get_optimizer
from tnlearn.base1 import BaseModel1
from torchinfo import summary
from sklearn.metrics import r2_score


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
                 ):
        r""" Construct MLPRegressor with task-based neurons.

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
        """

        super(MLPRegressor, self).__init__()

        # Initialize the member variables with the values provided to the constructor
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

        if fig_path is None:
            self.fig_path = './'
        else:
            self.fig_path = fig_path

        # Select the device for training based on GPU configuration
        self.gpu = gpu
        self.device = self.select_device(gpu)

        # Validation to ensure that the visual_interval is an integer and not zero
        assert isinstance(self.visual_interval,
                          int) and self.visual_interval, "visual_interval must be a non-zero integer"

        # Use default layers if none are provided
        if layers_list is None:
            self.layers_list = [50, 30, 10]
        else:
            self.layers_list = layers_list

        # Set up the activation functions for layers if provided, otherwise use ReLU
        self.activation_funcs = get_activation_function(activation_funcs) if activation_funcs else nn.ReLU()

        # Get the loss function; default to MSE (Mean Squared Error) if not provided
        self.loss_function = get_loss_function(loss_function) if loss_function else nn.MSELoss()

        # Set the random seed for reproducibility
        random_seed(self.random_state)

    def select_device(self, gpu):
        r"""Selects the training device based on the 'gpu' parameter.

        Args:
            gpu: GPU ID.
        """
        # If GPU is not specified, use CPU for training
        if gpu is None:
            return torch.device("cpu")
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Training will default to CPU.")
        # Select specific GPU if an integer index is provided
        if isinstance(gpu, int):
            cuda_device = f'cuda:{gpu}'
            return torch.device(cuda_device)
        # If a list or tuple is provided, ensure all indexes are valid and return the GPU list
        elif isinstance(gpu, (list, tuple)):
            for g in gpu:
                if not isinstance(g, int) or g >= torch.cuda.device_count():
                    raise ValueError(f"Invalid GPU index {g}")
            return gpu  # Return the list of GPU indexes for DataParallel use
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

        layers = []
        last_dim = input_dim
        # Iterate over the list to create each layer in the neural network model
        for neuron_count in self.layers_list:
            # Append a custom layer with specified number of neurons
            layers.append(CustomNeuronLayer(last_dim, neuron_count, self.neurons))
            last_dim = neuron_count
            layers.append(self.activation_funcs)

        # Add the final linear layer that outputs to the desired output dimension
        layers.append(nn.Linear(last_dim, output_dim))
        # Compile all the layers into a single sequential model
        return nn.Sequential(*layers)

    def prepare_data(self, X, y):
        r"""Prepares the input data and splits it into training and validation sets.

        Args:
            X (numpy ndarray): Training data.
            y (numpy ndarray): Label data.
        """
        # Check and set the dimensions of the input data
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be a NumPy array.")
        if len(X.shape) != 2:
            raise ValueError("X should be a 2D NumPy array.")
        self.input_dim = X.shape[1]

        if not isinstance(y, np.ndarray):
            raise ValueError("y should be a NumPy array.")
        # Automatically determine the output dimension from the target data
        self.output_dim = y.shape[1] if len(y.shape) > 1 else 1

        # Convert the input and target data to torch tensors if they are not already
        if not isinstance(X, torch.Tensor):
            self.X = torch.tensor(X, dtype=torch.float32)

        if not isinstance(y, torch.Tensor):
            self.y = torch.tensor(y, dtype=torch.float32)

        # Create dataset objects to be used with DataLoaders
        trainset = TensorDataset(self.X, self.y)

        # DataLoader for the training set allows iterating over the data in batches and shuffling
        self.trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)

    def fit(self, X, y):
        r"""Train the network with training data.

        Args:
            X (numpy ndarray): Training data.
            y (numpy ndarray): Label data.
        """
        # Prepare the data by performing any necessary preprocessing
        self.prepare_data(X, y)

        # Initialize the lists for tracking loss
        self.losses = []

        # Build the model and move it to the appropriate device (CPU or GPU).
        if self.device == torch.device("cpu"):
            self.net = self.build_model(self.input_dim, self.output_dim).to(self.device)
        else:
            self.net = self.build_model(self.input_dim, self.output_dim)
            # Check if multiple GPUs are used.
            if isinstance(self.device, list):
                self.net = nn.DataParallel(self.net, device_ids=self.device)
                self.net.to(f'cuda:{self.device[0]}')
                self.device = f'cuda:{self.device[0]}'

            else:  # Single GPU case.
                self.net = self.net.to(self.device)
                self.device = self.device

        # Move the loss function to the device.
        self.cost = self.loss_function.to(self.device)
        self.optimizer = get_optimizer(name=self.optimizer_name, parameters=list(self.net.parameters()),
                                       lr=self.lr)

        # Initialize the scheduler if provided for learning rate adjustment.
        if self.scheduler is not None:
            scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.scheduler["step_size"],
                                            gamma=self.scheduler["gamma"])

        # Set the network to training mode.
        self.net.train()
        for epoch in range(self.max_iter):
            self.current_epoch = epoch + 1  # Update the current epoch counter.

            running_loss = 0.0
            # Iterate over the training data.
            for inputs, targets in self.trainloader:
                # Move the inputs and targets to the same device as the model.
                inputs, targets = inputs.to(self.device), targets.to(self.device).reshape(-1, 1)

                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.cost(outputs, targets)

                # L1 regularization if enabled.
                if self.l1_reg:
                    l1_loss = sum(p.abs().sum() for p in self.net.parameters())
                    loss += self.l1_reg * l1_loss

                # L2 regularization if enabled.
                if self.l2_reg:
                    l2_loss = sum(p.pow(2.0).sum() for p in self.net.parameters())
                    loss += self.l2_reg * l2_loss

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # Step the scheduler to update the learning rate if applicable.
            if self.scheduler is not None:
                scheduler.step()

            # Log the progress at the specified interval.
            if self.interval is not None and (epoch + 1) % self.interval == 0:
                print(f'Epoch [{epoch + 1}/{self.max_iter}], Loss: {running_loss:.4f}')

            # Append the loss (and optionally accuracy) to the lists for tracking.
            self.losses.append(running_loss)

            # Visualization of the training progress if enabled.
            if self.visual:
                if epoch % self.visual_interval == 0:
                    self.plot_progress_regression(loss=self.losses)

        # Save the visualization if enabled
        if self.save_fig:
            self.regression_savefigure(loss=self.losses, path=self.fig_path)

    def predict(self, X):
        r"""Use a trained model to make predictions.

        Args:
            X (torch.Tensor): Data that needs to be predicted.

        Returns:
            Predicted value
        """
        # Convert the input to a PyTorch tensor if it is not one already
        if not isinstance(X, torch.Tensor):
            X = torch.Tensor(X)

        # Create a TensorDataset with the tensor
        dataset = TensorDataset(X)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.net.eval()

        predictions = []
        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs[0].to(self.device)
                outputs = self.net(inputs)
                # Move outputs to cpu and convert to numpy, then add to predictions list
                predictions.extend(outputs.cpu().numpy())

        # Flatten the predictions list to a 1D numpy array
        predictions = np.array(predictions).flatten()
        return predictions

    def score(self, X, y):
        r"""Evaluate the coefficient of determination.

        Args:
            X (numpy ndarray): Test data.
            y (numpy ndarray): Label data.

        Returns:
            Coefficient of determination.
        """

        # Convert the input to a PyTorch tensor if it is not one already
        if not isinstance(X, torch.Tensor):
            X = torch.Tensor(X)

        # Create a TensorDataset with the tensor
        dataset = TensorDataset(X)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.net.eval()

        predictions = []
        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs[0].to(self.device)
                outputs = self.net(inputs)
                # Move outputs to cpu and convert to numpy, then add to predictions list
                predictions.extend(outputs.cpu().numpy())

        # Flatten the predictions list to a 1D numpy array
        predictions = np.array(predictions).flatten()

        return r2_score(y, predictions)

    def count_param(self):
        r"""Print the network structure and output the number of network parameters."""

        summary(self.net, input_size=(self.batch_size, 1, 10, self.input_dim))
