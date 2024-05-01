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

import os
import torch
from tnlearn.visualize import Visualization_Classification
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score


class BaseModel:
    r"""Basic module used to ensure the normal operation of MLPClassifier."""

    def __init__(self):
        r"""Initialization method of the BaseModel class that sets up a visualization tool"""
        self.visualization_classification = Visualization_Classification()

    def plot_progress_classification(self, loss, accuracy):
        r"""Method to update the progress plot during training.

        Args:
            loss: Training loss.
            accuracy: Training accuracy.
        """
        # Update visualization with the current epoch, loss, and accuracy
        self.visualization_classification.update(self.current_epoch, loss, accuracy)

    def classification_savefigure(self, loss, accuracy, path):
        r"""Method to save the training process figure.

        Args:
            loss: Training loss.
            accuracy: Training accuracy.
            path: Path to save the figure.
        """
        self.visualization_classification.savefigure(loss, accuracy, path)

    def save(self, path, filename):
        r"""Save the current model to the specified path with the given filename.

        Args:
            path: The path where the model weights are saved.
            filename: Name of the weight files.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        full_path = os.path.join(path, filename)
        # Save the model's weights to the constructed path
        torch.save(self.net.state_dict(), full_path)
        print(f"Model saved to {full_path}")

    def load(self, path, filename, input_dim, output_dim):
        r"""Load a model from the specified path with the given filename.

        Args:
            path: The location of the trained model.
            filename: File name of the trained model.
            input_dim: The input dimension of the network.
            output_dim: The output dimension of the network.
        """
        full_path = os.path.join(path, filename)
        # Check if the specified model file exists
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"No model found at {full_path}")

        # Reconstruct the model's architecture before loading the weights
        self.build_model(input_dim=input_dim, output_dim=output_dim)
        self.net.load_state_dict(torch.load(full_path))
        self.net.eval()  # Set the model to evaluation mode after loading weights
        print(f"Model loaded from {full_path}")

    def calculate_auc(self, y_true, y_pred):
        r"""Calculate the Area Under the Receiver Operating Characteristic Curve (ROC AUC)"""
        return roc_auc_score(y_true, y_pred)

    def calculate_f1_score(self, y_true, y_pred):
        r"""Calculate the F1 score, a weighted average of precision and recall"""
        return f1_score(y_true, y_pred)

    def calculate_recall(self, y_true, y_pred):
        r"""Calculate the recall, the ability of the classifier to find all the positive samples"""
        return recall_score(y_true, y_pred)

    def calculate_precision(self, y_true, y_pred):
        r"""Calculate the precision, the ability of the classifier not to label a sample
         as positive if it is negative"""
        return precision_score(y_true, y_pred)
