"""
Program name: PyTorch Utilities for tnlearn
Purpose description: This utility script is designed to streamline various machine
                     learning tasks related to PyTorch training experiments. It provides
                     essential utilities such as seeding for reproducibility, a custom
                     dataset class for data pairs, activation and loss function
                     fetchers, an optimizer selector, and a visualization class for
                     plotting training progress in terms of loss and accuracy. By providing
                     these utilities, tnlearn enhances the consistency and ease of
                     managing training workflows and evaluating model performance.
Last revision date: February 23, 2024
Known Issues: None identified at the time of the last revision.
Note: Assumes that all the dependencies, especially PyTorch, NumPy, and Matplotlib,
      are installed and properly configured.
"""

import torch
import numpy as np
import random
from torch.utils.data import Dataset
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
from IPython.display import clear_output


# Set the random seed for reproducibility of experiments
def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Dataset class to handle data pairs
class MyData(Dataset):
    def __init__(self, pics, labels):
        self.pics = pics
        self.labels = labels

    def __getitem__(self, index):
        # Fetch a single item by index
        assert index < len(self.pics)
        return torch.Tensor(self.pics[index]), self.labels[index]

    def __len__(self):
        # Return the size of the dataset
        return len(self.pics)

    def get_tensors(self):
        # Return all images and labels as tensor batches
        return torch.Tensor([self.pics]), torch.Tensor(self.labels)


# Get the corresponding PyTorch activation function by name
def get_activation_function(name):
    """Return the corresponding PyTorch activation function from a string name."""
    activations = {
        'relu': nn.ReLU(),
        'leakyrelu': nn.LeakyReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'softmax': nn.Softmax(dim=-1),  # You could need to specify the dimension
        # Add more activation functions if needed
    }
    # Return the requested activation function or ReLU as default
    return activations.get(name.lower(), nn.ReLU())


# Get the corresponding PyTorch loss function by name
def get_loss_function(name):
    """Map of loss function names to their torch.nn equivalents."""
    activations = {
        'mse': nn.MSELoss(),
        'l1': nn.L1Loss(),
        'crossentropy': nn.CrossEntropyLoss(),
        'bce': nn.BCELoss(),

        # Add more activation functions if needed
    }
    # Return the requested loss function or MSELoss as default
    return activations.get(name.lower(), nn.MSELoss())


# Get the corresponding PyTorch optimizer by name
def get_optimizer(name, parameters, lr=0.001, **kwargs):
    """
    Return the corresponding PyTorch optimizer given its string name.

    :param name: The name of the optimizer (e.g., 'adam', 'sgd')
    :param parameters: The parameters of the model to optimize.
    :param lr: Learning rate
    :param kwargs: Other arguments specific to the optimizer
    :return: An instance of the requested optimizer.
    """

    optimizers = {
        'adam': optim.Adam(params=parameters, lr=lr, **kwargs),
        'sgd': optim.SGD(params=parameters, lr=lr, **kwargs),
        'rmsprop': optim.RMSprop(params=parameters, lr=lr, **kwargs),
        'adamw': optim.AdamW(params=parameters, lr=lr, **kwargs),
        # Add more optimizers if needed
    }

    # Return the requested optimizer or Adam as default
    return optimizers.get(name.lower(), optim.Adam(parameters, lr=lr))


# Class for plotting and visualizing training progress
class Visualization:
    def __init__(self, save_fig=False, save_path='train_plot.png'):
        self.save_fig = save_fig  # Determine whether to save the figure
        self.save_path = save_path  # Path where the figure will be saved
        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 4))
        plt.ioff()  # Turn interactive plotting off
        plt.ion()  # Turn interactive plotting on

    def update(self, epoch, loss, accuracy=None, savefig=False):
        # Update the plots for loss and accuracy across epochs

        clear_output(wait=True)  # Clear the output of the current cell showing the plot

        # Plot training loss
        ax_loss = self.ax[0]
        ax_loss.cla()
        ax_loss.set_title('Loss')
        ax_loss.plot(range(1, epoch + 1), loss, label='Training Loss')
        ax_loss.legend()

        # If accuracy is provided, plot it
        if accuracy is not None:
            ax_accuracy = self.ax[1]
            ax_accuracy.cla()
            ax_accuracy.set_title('Accuracy')
            ax_accuracy.plot(range(1, epoch + 1), accuracy, label='Training Accuracy')
            ax_accuracy.legend()

        # Save figure if requested
        if savefig or self.save_fig:
            self.save()

        plt.draw()  # Update the plot
        plt.pause(0.01)  # Pause the plot to update it
        plt.show()

    def save(self):
        # Save the figure to the path defined in __init__
        self.fig.savefig(self.save_path)

