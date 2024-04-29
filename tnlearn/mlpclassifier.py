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
from tnlearn.base import BaseModel
from torchinfo import summary


class MLPClassifier(BaseModel):
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
                 save=False,
                 fig_path=None,
                 visual_interval=100,
                 gpu=None,
                 interval=None,
                 scheduler=None,
                 l1_reg=None,
                 l2_reg=None,
                 ):
        r"""Construct MLPClassifier with task-based neurons.

    Args:
         neurons: Neuronal expression
         layers_list: List of neuron counts for each hidden layer
         activation_funcs: Activation functions
         loss_function: Loss function for the training process
         optimizer_name: Name of the optimizer algorithm
         random_state: Seed for random number generators for reproducibility
         max_iter: Maximum number of training iterations
         batch_size: Number of samples per batch during training
         lr: Learning rate for the optimizer
         visual: Boolean indicating if training visualization is to be shown
         visual_interval: Interval at which training visualization is updated
         save: Indicates if the training figure should be saved
         gpu: Specifies GPU configuration for training
         interval: Interval of screen output during training
         scheduler: Learning rate scheduler
         l1_reg: L1 regularization term
         l2_reg: L2 regularization term
        """

        super(MLPClassifier, self).__init__()

        # Initialize parameters
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

        # Set device for computation
        self.gpu = gpu
        self.device = self.select_device(gpu)

        # Check if visual_interval is an integer and non-zero
        assert isinstance(self.visual_interval,
                          int) and self.visual_interval, "visual_interval must be a non-zero integer"

        # Set default layers if layers_list is not provided
        if layers_list is None:
            self.layers_list = [50, 30, 10]
        else:
            self.layers_list = layers_list

        # Convert activation function string to PyTorch activation function
        self.activation_funcs = get_activation_function(activation_funcs) if activation_funcs else nn.ReLU()

        # Convert loss function string to PyTorch loss function
        self.loss_function = get_loss_function(loss_function) if loss_function else nn.CrossEntropyLoss()

        # Set random seed
        random_seed(self.random_state)

    def select_device(self, gpu):
        r"""Selects the training device based on the 'gpu' parameter.

        Args:
            gpu: GPU ID.
        """
        # If gpu is None, return CPU as device
        if gpu is None:
            return torch.device("cpu")
        # Check if CUDA is available, raise an error if it's not
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Training will default to CPU.")
        # If gpu is an integer, return the corresponding CUDA device
        if isinstance(gpu, int):
            cuda_device = f'cuda:{gpu}'
            return torch.device(cuda_device)
        # If gpu is a list or tuple, validate each element as a valid GPU index
        elif isinstance(gpu, (list, tuple)):
            for g in gpu:
                if not isinstance(g, int) or g >= torch.cuda.device_count():
                    raise ValueError(f"Invalid GPU index {g}")
            # Return the list of GPU indices, which will be used in DataParallel later
            return gpu
        # Raise an error for an invalid 'gpu' parameter
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
        # Iterate over the list to create each layer
        for neuron_count in self.layers_list:
            layers.append(CustomNeuronLayer(last_dim, neuron_count, self.neurons))
            last_dim = neuron_count
            layers.append(self.activation_funcs)

        layers.append(CustomNeuronLayer(last_dim, output_dim, self.neurons))  # Final output layer
        return nn.Sequential(*layers)

    def prepare_data(self, X, y):
        r"""Prepares the input data and splits it into training and validation sets.

        Args:
            X (torch.Tensor or numpy ndarray): Training data.
            y (torch.Tensor or numpy ndarray): Label data.
        """
        # Data dimension check and setting
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be a NumPy array.")
        if len(X.shape) != 2:
            raise ValueError("X should be a 2D NumPy array.")
        self.input_dim = X.shape[1]

        if not isinstance(y, np.ndarray):
            raise ValueError("y should be a NumPy array.")
        self.output_dim = len(np.unique(y))

        # Ensure X and y are PyTorch tensors
        if not isinstance(X, torch.Tensor):
            self.X = torch.tensor(X, dtype=torch.float32)

        if not isinstance(y, torch.Tensor):
            self.y = torch.tensor(y, dtype=torch.long)

        # Dataset transformation
        trainset = TensorDataset(self.X, self.y)
        self.trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)

    def fit(self, X, y):
        r"""Train the network with training data.

        Args:
            X (torch.Tensor or numpy ndarray): Training data.
            y (torch.Tensor or numpy ndarray): Label data.
        """
        # Prepare the data
        self.prepare_data(X, y)

        # Initialize lists to track losses and, optionally, accuracies
        self.losses = []

        # Build the neural network model based on input and output dimensions and move it to the specified device
        if self.device == torch.device("cpu"):
            self.net = self.build_model(self.input_dim, self.output_dim).to(self.device)
        else:
            self.net = self.build_model(self.input_dim, self.output_dim)
            # If using multiple GPUs
            if isinstance(self.device, list):
                self.net = nn.DataParallel(self.net, device_ids=self.device)
                self.net.to(f'cuda:{self.device[0]}')
                self.device = f'cuda:{self.device[0]}'
            else:
                # If using a single GPU
                self.net = self.net.to(self.device)
                self.device = self.device

        # Move the loss function to the specified device
        self.cost = self.loss_function.to(self.device)

        # Define the optimizer
        self.optimizer = get_optimizer(name=self.optimizer_name,
                                       parameters=list(self.net.parameters()),
                                       lr=self.lr)

        # Add learning rate adjustment strategy
        if self.scheduler is not None:  # If a learning rate adjustment strategy is provided
            scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.scheduler["step_size"],
                                            gamma=self.scheduler["gamma"])

        # Initialize lists to track training accuracies
        self.net_train_accuracy = []

        # Set the model to training mode
        self.net.train()

        # Iterate through each epoch
        for epoch in range(self.max_iter):
            self.current_epoch = epoch + 1  # Update the current epoch

            running_loss = 0.0
            for inputs, labels in self.trainloader:
                # Move inputs and labels to the specified device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Reset the gradients
                self.optimizer.zero_grad()
                # Forward pass
                outputs = self.net(inputs)
                # Calculate the loss
                loss = self.cost(outputs, labels)

                # Add L1 regularization if specified
                if self.l1_reg:
                    l1_regularization = torch.tensor(0., requires_grad=True).to(self.device)
                    for param in self.net.parameters():
                        l1_regularization = l1_regularization + torch.norm(param, 1)
                    loss += self.l1_reg * l1_regularization

                # Add L2 regularization if specified
                if self.l2_reg:
                    l2_regularization = torch.tensor(0., requires_grad=True).to(self.device)
                    for param in self.net.parameters():
                        l2_regularization = l2_regularization + torch.norm(param, 2)
                    loss += 0.5 * self.l2_reg * l2_regularization

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # Update the learning rate at the end of each epoch if a scheduler is provided
            if self.scheduler is not None:
                scheduler.step()

            train_accuracy = 0.0
            with torch.no_grad():
                for inputs, labels in self.trainloader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    predict = self.net(inputs)
                    _, predict_class = torch.max(predict, 1)
                    train_accuracy += (predict_class == labels).sum().item()
            train_accuracy = train_accuracy / self.X.shape[0]

            # Log the progress at the specified interval.
            if self.interval is not None and (epoch + 1) % self.interval == 0:
                print(f'Epoch [{epoch + 1}/{self.max_iter}], Loss: {running_loss:.4f}, Accuracy:{train_accuracy:.4f}')

            # Append the loss and accuracy to the history list
            self.losses.append(running_loss)

            self.net_train_accuracy.append(train_accuracy)

            # Visualize the progress if enabled and at specified intervals
            if self.visual:
                if epoch % self.visual_interval == 0:
                    self.plot_progress_classification(loss=self.losses, accuracy=self.net_train_accuracy)

        # Save the visualization if enabled
        if self.save_fig:
            self.classification_savefigure(loss=self.losses, accuracy=self.net_train_accuracy, path=self.fig_path)

    def predict(self, X):
        r"""Use a trained model to make predictions.

        Args:
            X (torch.Tensor): Data that needs to be predicted.

        Returns:
            Predicted value
        """
        # Convert data to torch.Tensor if it's not already
        if not isinstance(X, torch.Tensor):
            X = torch.Tensor(X)

        # Create a TensorDataset object
        dataset = TensorDataset(X)
        # Create a DataLoader
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # Set the network to evaluation mode
        self.net.eval()

        predictions = []
        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs[0].to(self.device)  # Here, inputs is a list containing one element
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                # Extend the list of predictions
                predictions.extend(predicted.cpu().numpy())

        # Return the predictions
        return np.array(predictions)

    def score(self, X_, y_):
        r"""Evaluate the score of the model.

        Args:
            X_ (torch.Tensor or numpy ndarray): Training data.
            y_ (torch.Tensor or numpy ndarray): Label data.

        Returns:
            accuracy
        """
        # Ensure X and y are PyTorch tensors
        if not isinstance(X_, torch.Tensor):
            X_ = torch.tensor(X_, dtype=torch.float32)
        if not isinstance(y_, torch.Tensor):
            y_ = torch.tensor(y_, dtype=torch.long)

        # Create a DataLoader
        dataset = TensorDataset(X_, y_)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # Set the network to evaluation mode
        self.net.eval()
        correct = 0.0

        # Disable gradient calculation
        with torch.no_grad():
            for inputs, labels in loader:
                # Move to the appropriate device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Forward pass
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                # Calculate the number of correct predictions
                correct += (predicted == labels).sum().item()

        # Set the network back to training mode
        self.net.train()

        # Calculate accuracy
        correct /= X_.shape[0]

        print(f'Accuracy: {correct:.4f}')
        return correct

    def count_param(self):
        summary(self.net, input_size=(self.batch_size, 1, 10, self.input_dim))
