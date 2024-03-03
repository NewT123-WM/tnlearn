"""
Program name: MLPRegressor Class Implementation
Purpose description: This script embodies the MLPRegressor class, designed as an extension
                     of BaseModel within a custom machine learning framework.
                     It facilitates the creation and training of a multilayer perceptron (MLP)
                     for regression tasks. Features of the class include the flexibility to
                     define custom neural network architectures through parameters such as
                     neuronal expression and layer structure, the utilization of various activation
                     functions, and the incorporation of modern optimization algorithms and
                     regularization techniques. Moreover, the class is equipped with optional
                     GPU support for enhanced computational efficiency, as well as functionalities
                     for training visualization, validation, and model performance evaluation.
                     This representation of an MLP is tailored to adapt to an array of regression
                     problems while ensuring ease of use and extensibility.
Last revision date: February 27, 2024
Known Issues: None have been identified at the time of the last revision.
Note: The application of the MLPRegressor class presupposes that the necessary neural
      network modules, loss functions, and optimizers are available and correctly configured
      in the encompassing framework. It is also assumed that the user has proper knowledge
      of the regression task requirements and has the appropriate data pre-processing
      steps applied to their data before model training.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import torch.optim.lr_scheduler as lr_scheduler
from .utils import MyData
from .neurons import CustomNeuronLayer
from .utils import random_seed
from .utils import get_activation_function
from .utils import get_loss_function
from .utils import get_optimizer
from .base import BaseModel


class MLPRegressor(BaseModel):
    def __init__(self,
                 neurons='x',
                 layers_list=None,
                 activation_funcs=None,
                 loss_function=None,
                 optimizer_name='adam',
                 random_state=1,
                 max_iter=300,
                 batch_size=128,
                 valid_size=0.2,
                 lr=0.01,
                 visual=False,
                 visual_interval=100,
                 save=False,
                 gpu=None,
                 interval=None,
                 scheduler=None,
                 l1_reg=False,
                 l2_reg=False,
                 ):
        """ Parameter interpretation

         neurons: Neuronal expression
         layers_list: List of neuron counts for each hidden layer
         activation_funcs: Activation functions
         loss_function: Loss function for the training process
         optimizer_name: Name of the optimizer algorithm
         random_state: Seed for random number generators for reproducibility
         max_iter: Maximum number of training iterations
         batch_size: Number of samples per batch during training
         valid_size: Fraction of training data used for validation
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

        super(MLPRegressor, self).__init__()

        # Initialize the member variables with the values provided to the constructor
        self.random_state = random_state
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.valid_size = valid_size
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

        # Select the device for training based on GPU configuration
        self.gpu = gpu
        self.device = self.select_device(gpu)

        # Validation to ensure that the visual_interval is an integer and not zero
        assert isinstance(self.visual_interval, int) and self.visual_interval, "visual_interval must be a non-zero integer"

        # Use default layers if none are provided
        if layers_list is None:
            self.layers_list = [50, 30, 10]
        else:
            self.layers_list = layers_list

        # Setup the activation functions for layers if provided, otherwise use ReLU
        self.activation_funcs = get_activation_function(activation_funcs) if activation_funcs else nn.ReLU()

        # Get the loss function; default to MSE (Mean Squared Error) if not provided
        self.loss_function = get_loss_function(loss_function) if loss_function else nn.MSELoss()

        # Set the random seed for reproducibility
        random_seed(self.random_state)

    def select_device(self, gpu):
        """Selects the training device based on the 'gpu' parameter."""
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

    # Constructs the neural network model based on the specified architecture
    def build_model(self, input_dim, output_dim):
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
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        # Split the dataset into training and validation sets
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=self.valid_size,
                                                              random_state=self.random_state)
        # Create dataset objects to be used with DataLoaders
        trainset = MyData(X_train, y_train)
        # DataLoader for the training set allows iterating over the data in batches and shuffling
        self.trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        validset = MyData(X_valid, y_valid)
        # DataLoader for the validation set, also in batches
        self.validloader = DataLoader(validset, batch_size=self.batch_size, shuffle=True)

    def fit(self, X, y):
        # Prepare the data by performing any necessary preprocessing.
        self.prepare_data(X, y)

        # Initialize the lists for tracking loss and, if required, accuracy.
        self.losses = []
        # self.accuracies = []

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
            running_loss /= len(self.trainloader)

            # Step the scheduler to update the learning rate if applicable.
            if self.scheduler is not None:
                scheduler.step()

            # Evaluate the model on the validation set after this epoch.
            valid_loss = self.evaluate(self.validloader)

            # Log the progress at the specified interval.
            if self.interval is not None and (epoch + 1) % self.interval == 0:
                print(f'Epoch [{epoch + 1}/{self.max_iter}], Loss: {running_loss:.4f},'
                      f' Validation Loss: {valid_loss:.4f}')

            # Append the loss (and optionally accuracy) to the lists for tracking.
            self.losses.append(running_loss)

            # Visualization of the training progress if enabled.
            if self.visual:
                if epoch % self.visual_interval == 0:
                    self.plot_progress(loss=self.losses)

        # Save the plot to file if save_fig is set.
        if self.save_fig:
            self.plot_progress(loss=self.losses, savefig=self.save_fig)

    # Evaluate the network using validation set
    def evaluate(self, dataloader):
        # Set the network to evaluation mode
        self.net.eval()
        total_loss = 0.0

        # Disable gradient calculations
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device).reshape(-1, 1)
                outputs = self.net(inputs)
                loss = self.cost(outputs, targets)
                total_loss += loss.item()

        # Return the average loss over the batches
        return total_loss / len(dataloader)

    # Use a trained model to make predictions.
    def predict(self, X):
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
        print(predictions)
        return predictions

    def score(self, X, y):
        # Convert X to a PyTorch float tensor if it is not one already
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        # Convert y to a PyTorch float tensor if it is not one already
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        # Create a DataLoader
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.net.eval()

        # Initialize total loss to zero
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = torch.nn.functional.mse_loss(outputs, targets.reshape_as(outputs))
                # Update total loss with the weighted loss of the batch
                total_loss += loss.item() * inputs.size(0)

        # Divide the total loss by the number of samples to get the average loss
        total_loss /= len(loader)

        print(f'Mean Squared Error: {total_loss:.4f}')
        # Return the mean squared error
        return total_loss
