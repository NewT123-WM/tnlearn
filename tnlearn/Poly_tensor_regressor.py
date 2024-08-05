import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#import torch.nn.utils as nn_utils
#from sklearn.preprocessing import StandardScaler
#import pandas as pd

import matplotlib.pyplot as plt


class PolynomialTensorRegression(nn.Module):
    r"""
         Initializes the Polynomial Tensor Regression model.

        Args:
            decomp_rank (int): The rank of the tensor decomposition.
            poly_order (int): The order of the polynomial.
            method: The decomposition method ('cp' or 'tucker').
            net_dims: The dimensions of the neural network layers.
            reg_lambda_w: Regularization parameter for W tensor.
            reg_lambda_c: Regularization parameter for C coefficients.
            num_epochs (int): Number of training iterations.
            learning_rate (int): Learning rate.
            batch_size (int): Number of samples per batch.
    """

    def __init__(self,
                 decomp_rank,
                 poly_order,
                 method='cp',
                 net_dims=(64, 32),
                 reg_lambda_w=0.01,
                 reg_lambda_c=0.01,
                 num_epochs=100,
                 learning_rate=0.001,
                 batch_size=64):

        super(PolynomialTensorRegression, self).__init__()
        self.decomp_rank = decomp_rank
        self.poly_order = poly_order
        self.method = method
        self.net_dims = net_dims
        self.reg_lambda_w = reg_lambda_w
        self.reg_lambda_c = reg_lambda_c
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.C = nn.ParameterList(
            [nn.Parameter(torch.randn(1)) for _ in range(poly_order)])
        self.beta = nn.Parameter(torch.randn(1))
        self.network = self.build_network(self.net_dims)
        self.neuron = None
        self.U = nn.ModuleList()
        if method == 'tucker':
            self.core_tensors = nn.ParameterList()
            for i in range(poly_order):
                core_shape = tuple([decomp_rank] * (i + 1))
                self.core_tensors.append(
                    nn.Parameter(torch.randn(core_shape, dtype=torch.float32)))

    def initialize_factor_u(self, input_dim):
        r"""
        Initializes the U factor matrices.

        Parameters:
        - input_dim: The dimensionality of the input data.
        """
        if self.method == 'cp':
            for i in range(1, self.poly_order + 1):
                U_i = nn.ParameterList()
                for n in range(i):
                    U_i.append(
                        nn.Parameter(torch.randn(input_dim, self.decomp_rank)))
                self.U.append(U_i)
        elif self.method == 'tucker':
            for i in range(1, self.poly_order + 1):
                U_i = nn.ParameterList()
                for n in range(i):
                    U_i.append(
                        nn.Parameter(
                            torch.randn(input_dim,
                                        self.core_tensors[i - 1].shape[n])))
                self.U.append(U_i)

    def build_network(self, net_dims):
        r"""
        Builds the neural network.

        Parameters:
        - net_dims: Number of neurons in each layer.

        Returns:
        - A neural network composed of fully connected layers and ReLU activations.
        """
        layers = []
        input_dim = self.decomp_rank
        for dim in net_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            input_dim = dim
        layers.append(nn.Linear(input_dim, 1))
        return nn.Sequential(*layers)

    def tucker_tensor_reconstruct(self, core_tensor, u_factors):
        r"""
        Reconstructs a tensor using Tucker decomposition.

        Parameters:
        - core_tensor: The core tensor.
        - u_factors: List of U factor matrices.

        Returns:
        - The reconstructed tensor.
        """
        W = core_tensor
        for i, U in enumerate(u_factors):
            W = torch.tensordot(W, U, dims=([W.ndimension() - 1 - i], [1]))
        return W

    def cp_tensor_reconstruct(self, u_factors):
        r"""
        Reconstructs a tensor using CP decomposition.

        Parameters:
        - u_factors: List of U factor matrices.

        Returns:
        - The reconstructed tensor.
        """
        rank = self.decomp_rank
        input_dim = u_factors[0].shape[0]
        rank_shape = (input_dim, ) * len(u_factors)
        W = torch.zeros(rank_shape)
        for i in range(rank):
            outer_product = u_factors[0][:, i]
            for factor in u_factors[1:]:
                outer_product = torch.ger(outer_product, factor[:,i]).reshape(-1)
            W += outer_product.reshape(rank_shape)
        return W

    def compute_polynomial_term(self, z, i):
        r"""
        Computes a specific term of the polynomial.

        Parameters:
        - z: Input tensor.
        - i: Index of the term.

        Returns:
        - The value of the i-th term.
        """
        u_factors = [U for U in self.U[i]]
        #print(f'U factor shape is{u_factors.shape}')
        if self.method == 'tucker':
            core_tensor = self.core_tensors[i]
            weight_tensor = self.tucker_tensor_reconstruct(
                core_tensor, u_factors)
        elif self.method == 'cp':
            weight_tensor = self.cp_tensor_reconstruct(u_factors)
        #print(f'W tensor of term {i} shape is : {weight_tensor.shape}')
        for _ in range(i + 1):

            weight_tensor = torch.tensordot(weight_tensor, z, dims=([0], [0]))
        term = self.C[i] * weight_tensor
        #print(f'shape of term {i} is : {term.shape}')
        return term, weight_tensor

    def forward(self, X):
        r"""
        Calculate polynomial value and loss of fitting  
        Using absolute value of C as regularization term

        """
        preds = []
        regularization_loss_w = 0
        for x in X:
            z = x.view(-1)
            result = self.beta.clone()
            for j in range(self.poly_order):
                term, weight_tensor = self.compute_polynomial_term(z, j)
                result += term
                #regularization_loss_w += torch.sum(weight_tensor**2)/weight_tensor.numel()
                regularization_loss_w += torch.sum(
                    torch.abs(weight_tensor)) / weight_tensor.numel()
            preds.append(result)
        regularization_loss_c = self.reg_lambda_c * torch.sum(
            torch.stack([torch.sum(torch.abs(c)) for c in self.C]))
        #regularization_loss_c = self.reg_lambda_c * torch.sum(torch.stack([torch.sum(c**2) for c in self.C]))
        regularization_loss = regularization_loss_c
        #+ regularization_loss_c
        #self.reg_lambda_w * regularization_loss_w
        return torch.stack(preds).view(-1), regularization_loss

    def train_model(self, X, y, view_training_process=False):
        r"""
        Trains the model.

        Parameters:
        - X: Input data.
        - y: Target data.
        - loss_picture: Boolean indicating whether to plot the training loss.
        """
        input_dim = np.prod(X.shape[1:])
        self.initialize_factor_u(input_dim)

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=True)

        losses = []

        for epoch in range(self.num_epochs):
            self.train()
            epoch_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs, reg_loss = self.forward(batch_x)
                loss = criterion(outputs, batch_y) + reg_loss
                loss.backward()
                epoch_loss += loss.item()
                optimizer.step()

            epoch_loss /= len(dataloader)
            losses.append(epoch_loss)
            if view_training_process:
                if (epoch + 1) % 10 == 0:
                    print(
                        f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}'
                    )

        if view_training_process:
            plt.figure(figsize=(10, 5))
            plt.plot(losses, label='Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training Loss Over Epochs')
            plt.show()

    def get_dynamic_threshold(self):
        """
        Calculates a dynamic threshold for identifying significant polynomial terms.

        Returns:
        - The dynamic threshold.
        """
        all_c_values = [c.item() for c in self.C]
        mean_val = np.mean(np.abs(all_c_values))
        std_val = np.std(np.abs(all_c_values))
        return mean_val - std_val

    def get_significant_polynomial(self):
        """
        Retrieves significant polynomial terms.

        Returns:
        - A string representing the significant terms.
        """
        threshold = self.get_dynamic_threshold()
        significant_terms = []
        if self.beta.item() >= 0:
            significant_terms.append(f'{self.beta.item():.4f}')
        else:
            significant_terms.append(f'- {abs(self.beta.item()):.4f}')
            
        for i, c in enumerate(self.C):
            c_value = c.item()
            if abs(c_value) > threshold:
                if c_value >= 0:
                    term = f'+ {c_value:.4f} @ x**{i + 1}'
                else:
                    term = f'- {abs(c_value):.4f} @ x**{i + 1}'
                significant_terms.append(term)
        
        polynomial = ' '.join(significant_terms)
        if not polynomial:
            polynomial = '0'
        #print('Significant Polynomial:', polynomial)
        return polynomial

    def fit(self, X, y, view_training_process=False):
        r"""
        Fits the model to the data.

        Parameters:
        - X: Input data.
        - y: Target data.
        - view_training_process: Boolean indicating whether to plot the training loss.

        Returns:
        - The significant polynomial as a string.
        """
        self.train_model(X, y, view_training_process=view_training_process)
        self.neuron = self.get_significant_polynomial()
