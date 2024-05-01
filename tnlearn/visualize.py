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

import matplotlib.pyplot as plt
from IPython.display import clear_output


# class Visualization:
#     r"""Class for plotting and visualizing training progress."""
#
#     def __init__(self, save_fig=False, save_path='train_plot.png'):
#         self.save_fig = save_fig  # Determine whether to save the figure
#         self.save_path = save_path  # Path where the figure will be saved
#         self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 4))
#         # self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 4))
#         plt.ioff()  # Turn interactive plotting off
#         plt.ion()  # Turn interactive plotting on
#
#     def update(self, epoch, loss, accuracy=None, savefig=False):
#         r"""Update the plots for loss and accuracy across epochs.
#
#         Args:
#             epoch: Epochs during training.
#             loss: Loss value during training.
#             accuracy: Accuracy value during training.
#             savefig (Boolean, default: False): Save the visualization figure.
#         """
#
#         clear_output(wait=True)  # Clear the output of the current cell showing the plot
#
#         # Plot training loss
#         ax_loss = self.ax[0]
#         ax_loss.cla()
#         ax_loss.set_title('Loss')
#         ax_loss.plot(range(1, epoch + 1), loss, label='Training Loss')
#         ax_loss.legend()
#
#         # If accuracy is provided, plot it
#         if accuracy is not None:
#             ax_accuracy = self.ax[1]
#             ax_accuracy.cla()
#             ax_accuracy.set_title('Accuracy')
#             ax_accuracy.plot(range(1, epoch + 1), accuracy, label='Training Accuracy')
#             ax_accuracy.legend()
#
#         plt.draw()  # Update the plot
#         plt.pause(0.01)  # Pause the plot to update it
#         plt.show()
#
#         # Save figure if requested
#         if savefig or self.save_fig:
#             self.fig.savefig(self.save_path)
#             # self.save()
# ===========================================================================================

class Visualization_Classification:
    r"""Class for plotting and visualizing training progress."""

    def __init__(self, save_path='train_plot.png'):
        self.save_path = save_path  # Path where the figure will be saved
        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 4))
        plt.ioff()  # Turn interactive plotting off
        plt.ion()  # Turn interactive plotting on

    def update(self, epoch, loss, accuracy):
        r"""Update the plots for loss and accuracy across epochs.

        Args:
            epoch: Epochs during training.
            loss: Loss value during training.
            accuracy: Accuracy value during training.
        """

        clear_output(wait=True)  # Clear the output of the current cell showing the plot

        # Plot training loss
        ax_loss = self.ax[0]
        ax_loss.cla()
        ax_loss.set_title('Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.plot(range(1, epoch + 1), loss, label='Training Loss')
        # ax_loss.legend()

        # Plot training accuracy
        ax_accuracy = self.ax[1]
        ax_accuracy.cla()
        ax_accuracy.set_title('Accuracy')
        ax_accuracy.set_xlabel('Epoch')
        ax_accuracy.set_ylabel('Accuracy')
        ax_accuracy.plot(range(1, epoch + 1), accuracy, label='Training Accuracy')
        # ax_accuracy.legend()

        plt.draw()  # Update the plot
        plt.pause(0.01)  # Pause the plot to update it
        plt.show()

    def savefigure(self, loss, accuracy, path):
        r"""Update the plots for loss and accuracy across epochs.

        Args:
            loss: Loss value during training.
            accuracy: Accuracy value during training.
        """

        clear_output(wait=True)  # Clear the output of the current cell showing the plot

        # Plot training loss
        ax_loss = self.ax[0]
        ax_loss.cla()
        ax_loss.set_title('Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.plot(loss, label='Training Loss')
        # ax_loss.legend()

        # Plot training accuracy
        ax_accuracy = self.ax[1]
        ax_accuracy.cla()
        ax_accuracy.set_title('Accuracy')
        ax_accuracy.set_xlabel('Epoch')
        ax_accuracy.set_ylabel('Accuracy')
        ax_accuracy.plot(accuracy, label='Training Accuracy')
        # ax_accuracy.legend()

        self.fig.savefig(path + self.save_path)


class Visualization_Regression:
    r"""Class for plotting and visualizing training progress."""

    def __init__(self, save_path='train_plot.png'):
        self.save_path = save_path  # Path where the figure will be saved
        plt.figure(figsize=(6, 4))
        plt.ioff()  # Turn interactive plotting off
        plt.ion()  # Turn interactive plotting on

    def update(self, epoch, loss):
        r"""Update the plots for loss and accuracy across epochs.

        Args:
            epoch: Epochs during training.
            loss: Loss value during training.
            accuracy: Accuracy value during training.
        """

        clear_output(wait=True)  # Clear the output of the current cell showing the plot

        plt.clf()
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(loss, label='Training Loss', color='royalblue')

        plt.draw()  # Update the plot
        plt.pause(0.01)  # Pause the plot to update it
        plt.show()

    def savefigure(self, loss, path):
        r"""Update the plots for loss and accuracy across epochs.

        Args:
            loss: Loss value during training.
        """

        clear_output(wait=True)  # Clear the output of the current cell showing the plot

        # Plot training loss
        plt.clf()
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(loss, label='Training Loss', color='royalblue')

        plt.savefig(path + self.save_path)
