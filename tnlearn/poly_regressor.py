import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Optional
from itertools import product
from torch.optim.lr_scheduler import CosineAnnealingLR


class PolyTensorRegression(nn.Module):
    def __init__(self,
                 rank,
                 poly_order,
                 method='cp',
                 reg_lambda_w=0.1,
                 reg_lambda_c=0.05,
                 num_epochs=100,
                 learning_rate=0.001,
                 batch_size=64,
                 task_type='regression',
                 num_classes: Optional[int] = None,
                 device: Optional[torch.device] = None,
                 track_callback = None):
        super(PolyTensorRegression, self).__init__()
        self.rank = rank
        self.poly_order = poly_order
        self.method = method      
        self.reg_lambda_w = reg_lambda_w
        self.reg_lambda_c = reg_lambda_c
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.task_type = task_type
        self.ablation_scores = None
        self.num_classes = num_classes
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.C = nn.ParameterList([nn.Parameter(torch.randn(1)).to(self.device) for _ in range(poly_order)])
        self.beta = nn.Parameter(torch.randn(1).to(self.device))
        self.neuron = None
        self.weight_scalars = {}
        self.network = None     
        self.track_callback = track_callback 
        self.U = nn.ParameterList()
        if method == 'tucker':
            self.core_tensors = nn.ParameterList()
            for i in range(poly_order):
                core_shape = tuple([rank] * (i + 1))
                self.core_tensors.append(nn.Parameter(torch.randn(core_shape)).to(self.device))
        
    
    def initialize_factor_u(self, input_dim):
        """Initialize factor matrices for CP/Tucker decomposition"""
        self.U = nn.ParameterList()
        if self.method == 'cp':
            for order in range(1, self.poly_order+1):
                U_i = nn.ParameterList([nn.Parameter(torch.randn(input_dim, self.rank, device=self.device)) 
                                      for _ in range(order)])
                self.U.append(U_i)
        elif self.method == 'tucker':
            for idx, core in enumerate(self.core_tensors):
                order = idx + 1 
                U_i = nn.ParameterList([nn.Parameter(torch.randn(input_dim, core.shape[dim], device=self.device))
                                      for dim in range(order)])
                self.U.append(U_i)

    def build_network(self, input_dim):
        """Build final classification layer if needed"""
        return nn.Linear(1, self.num_classes).to(self.device) if self.task_type == 'classification' else None # type: ignore

    def compute_term_cp(self, x, factors, order):
        batch_size, input_dim = x.size()
        rank = factors[0].size(1)
        result = torch.zeros(batch_size, device=x.device)
        
        for r in range(rank):
            term = 1.0
            for i in range(order):
                vec = factors[i][:, r]  # (input_dim,)
                term *= torch.matmul(x, vec)  # (batch_size,)
            result += term
        
        return result

    def compute_term_tucker(self,x, core, factors, order):
        dots_list = []
        batch_size = x.size(0)
        device = x.device

        for i in range(order):
            vecs = factors[i]  # (input_dim, R_i)
            dots = torch.matmul(x, vecs)  # (batch_size, R_i)
            dots_list.append(dots)
        
        result = torch.zeros(batch_size, device=device)
        
        for indices in product(*[range(dim) for dim in core.shape]):
            core_value = core[indices].item()
            terms = [dots_list[i][:, idx] for i, idx in enumerate(indices)]
            product_term = torch.prod(torch.stack(terms, dim=1), dim=1)  # (batch_size,)
            result += core_value * product_term
        
        return result

    def forward(self, X):
        X = X.to(self.device).view(X.size(0), -1)
        batch_size, input_dim = X.size()
        result = self.beta.expand(batch_size).clone()
        reg_loss_w = 0.0

        for j in range(self.poly_order):
            order = j + 1
            order_factor = order**2
            if self.method == 'cp':
                factors = self.U[j]
                term = self.compute_term_cp(X, factors, order)
                # Regularization: mean L1 of all factors
                factor_params = sum([u.numel() for u in factors])
                reg_term = sum([torch.abs(u).sum() for u in factors]) / factor_params
                reg_loss_w += self.reg_lambda_w * reg_term #* order_factor
            elif self.method == 'tucker':
                core = self.core_tensors[j]
                factors = self.U[j]
                term = self.compute_term_tucker(X, core, factors, order)
                core_params = core.numel()
                factor_params = sum([u.numel() for u in factors])
                total_params = core_params + factor_params
                reg_term = (torch.abs(core).sum() + sum([torch.abs(u).sum() for u in factors])) / total_params
                reg_loss_w += self.reg_lambda_w * reg_term #* order_factor

            term = self.C[j] * term
            result = result + term

        reg_loss_c = self.reg_lambda_c * torch.stack([c.abs().sum() for c in self.C]).sum()
        total_reg = reg_loss_w + reg_loss_c

        if self.task_type == 'classification':
            logits = self.network(result.unsqueeze(1)) # type: ignore
            return logits, total_reg
        return result, total_reg

    def train_model(self, X, y, view_training_process=False):
        """Training loop with regularization"""
        for param in self.parameters():
            if param.requires_grad:
                param.data = torch.randn_like(param.data)
        input_dim = np.prod(X.shape[1:])
        self.initialize_factor_u(input_dim)
        X_tensor = X.to(self.device)
        if self.task_type == 'regression':
            y_tensor = y.float().view(-1).to(self.device)
            self.loss_fn = nn.MSELoss()
        elif self.task_type == 'classification':
            y_tensor = y.long().view(-1).to(self.device)
            self.loss_fn = nn.CrossEntropyLoss()

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=self.num_epochs 
        )
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor) # type: ignore
        dataloader = torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=True) # type: ignore
        losses = []

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                preds, reg_loss = self.forward(batch_x)
                loss = self.loss_fn(preds, batch_y) + reg_loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(dataloader)
            losses.append(epoch_loss)
            scheduler.step()

            if self.track_callback:
                solution = self.get_significant_polynomial()
                self.track_callback(solution)
            

            print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.4f}')

        if view_training_process:
            plt.plot(losses)
            plt.show()

    def get_significant_polynomial(self, return_full=False):
        w_norms = []
        for j in range(self.poly_order):
            if self.method == 'cp':
                factors = self.U[j]
                factor_norms = 1.0
                for factor in factors:
                    norms = torch.norm(factor, p=2, dim=0)  
                    factor_norms *= norms.mean() 
                w_norms.append(factor_norms.item()) # type: ignore
            elif self.method == 'tucker':
                factors = self.U[j]
                core = self.core_tensors[j]
                factor_norms = 1.0
                for factor in factors:
                    norms = torch.norm(factor, p=2, dim=0)
                    factor_norms *= norms.mean()
                core_norm = torch.norm(core)
                w_norms.append((factor_norms * core_norm).item())
        
        adjusted_c_values = [c.item() * w_norms[i] for i, c in enumerate(self.C)]
        def format_term(value, power=0):
            if power == 0:  # 常数项
                if value >= 0:
                    return f'{value:.4f}'
                else:
                    return f'- {abs(value):.4f}'
            else:  
                if value >= 0:
                    return f'+ {value:.4f} @ x**{power}'
                else:
                    return f'- {abs(value):.4f} @ x**{power}'

        beta_value = self.beta.item()
        full_terms = [format_term(beta_value)]
        for i, adjusted_c in enumerate(adjusted_c_values):
            full_terms.append(format_term(adjusted_c, i+1))
        
        full_polynomial = ' '.join(full_terms)
        
        if return_full:
            return full_polynomial
 
        threshold = self.get_dynamic_threshold_for_adjusted_c([abs(c) for c in adjusted_c_values])
        significant_terms = []
        
        if abs(beta_value) > threshold:
            significant_terms.append(format_term(beta_value))
        
        for i, adjusted_c in enumerate(adjusted_c_values):
            if abs(adjusted_c) > threshold:
                significant_terms.append(format_term(adjusted_c, i+1))
        
        significant_polynomial = ' '.join(significant_terms)
        if not significant_terms:
            significant_polynomial = '0'
        
        return {
            'full': full_polynomial,
            'significant': significant_polynomial
        }
    
    def get_dynamic_threshold_for_adjusted_c(self, adjusted_c_values):
        if not adjusted_c_values:
            return 0
        max_val = max(adjusted_c_values)
        return max_val * 0.4  # adjust this parameter to control the threshold, higher value means less terms will be included

    def fit(self, X, y, view_training_process=False):
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X_tensor = X.to(self.device)

        if not isinstance(y, torch.Tensor):
            y_tensor = torch.tensor(y, device=self.device)
        else:
            y_tensor = y.to(self.device)

        input_dim = np.prod(X_tensor.shape[1:])
        if self.task_type == 'classification' or self.num_classes is None:
            self.num_classes = len(torch.unique(y_tensor))
            self.network = self.build_network(1)

        self.train_model(X_tensor, y_tensor, view_training_process=view_training_process)
        self.neuron = self.get_significant_polynomial()

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            outputs, _ = self.forward(X)
            if self.task_type == 'classification':
                probabilities = torch.softmax(outputs, dim=1)
                predicted_labels = torch.argmax(probabilities, dim=1)
                return predicted_labels
            else:
                return outputs.view(-1)