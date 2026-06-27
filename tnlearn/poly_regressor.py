import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from itertools import product
from torch.optim.lr_scheduler import CosineAnnealingLR
import random

def random_seed(seed):
    r"""Set the random seed for reproducibility of experiments.

    Args:
        seed: Random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class LegacyPolyTensorRegressor(nn.Module):
    def __init__(self,
                 rank=3,
                 poly_order=3,
                 method='cp',
                 reg_lambda_w=0.01,
                 reg_lambda_c=0.05,
                 num_epochs=100,
                 learning_rate=0.001,
                 batch_size=64,
                 task_type='regression',
                 num_classes: Optional[int] = None,
                 device: Optional[torch.device] = None,
                 track_callback = None,
                 random_state: Optional[int] = None):
        super().__init__()
        if random_state is not None:
            random_seed(random_state)
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
        # Add a parameter for the sine term
        self.C_sin = nn.Parameter(torch.randn(1).to(self.device))
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
                order = idx + 1  # Core tensor idx corresponds to polynomial order
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
            #order_factor = order**2
            if self.method == 'cp':
                factors = self.U[j]
                term = self.compute_term_cp(X, factors, order)
                factor_params = sum([u.numel() for u in factors])
                reg_term = sum([torch.abs(u).sum() for u in factors]) / factor_params
                reg_loss_w += self.reg_lambda_w * reg_term 
            elif self.method == 'tucker':
                core = self.core_tensors[j]
                factors = self.U[j]
                term = self.compute_term_tucker(X, core, factors, order)
                core_params = core.numel()
                factor_params = sum([u.numel() for u in factors])
                total_params = core_params + factor_params
                reg_term = (torch.abs(core).sum() + sum([torch.abs(u).sum() for u in factors])) / total_params
                reg_loss_w += self.reg_lambda_w * reg_term 

            term = self.C[j] * term
            result = result + term

        sine_term = self.C_sin * torch.sin(X).sum(dim=1)
        result = result + sine_term

        reg_loss_c = self.reg_lambda_c * (torch.stack([c.abs().sum() for c in self.C]).sum() + self.C_sin.abs().sum())
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
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=True)
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
            import matplotlib.pyplot as plt

            plt.plot(losses)
            plt.show()

    def get_dynamic_threshold(self):
        all_c_values = [c.item() for c in self.C] + [self.C_sin.item()]
        mean_val = np.mean(np.abs(all_c_values))
        std_val = np.std(np.abs(all_c_values))
        return mean_val   #std_val

    def get_significant_polynomial(self):
        threshold = self.get_dynamic_threshold()
        threshold = max(threshold, 0)
        significant_terms = []

        beta_value = self.beta.item()
        if abs(beta_value) > threshold:
            if beta_value >= 0:
                significant_terms.append(f'{beta_value:.4f}')
            else:
                significant_terms.append(f'- {abs(beta_value):.4f}')

        for i, c in enumerate(self.C):
            c_value = c.item()
            if abs(c_value) > threshold:
                if c_value >= 0:
                    term = f'+ {c_value:.4f} @ x**{i + 1}'
                else:
                    term = f'- {abs(c_value):.4f} @ x**{i + 1}'
                significant_terms.append(term)
        
        c_sin_value = self.C_sin.item()
        if abs(c_sin_value) > threshold:
            if c_sin_value >= 0:
                term = f'+ {c_sin_value:.4f} @ torch.sin(x)'
            else:
                term = f'- {abs(c_sin_value):.4f} @ torch.sin(x)'
            significant_terms.append(term)

        polynomial = ' '.join(significant_terms)
        if not polynomial:
            polynomial = '0'
        return polynomial

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
        if self.task_type == 'classification':
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


class DualStreamInteractionLayer(nn.Module):
    """Dual-stream polynomial interaction core from NeuronSeek-TD."""

    def __init__(self, input_dim: int, num_classes: int, rank: int, poly_order: int):
        super().__init__()
        self.rank = rank
        self.num_classes = num_classes
        self.poly_order = poly_order

        self.factors = nn.ModuleList()
        for order in range(1, poly_order + 1):
            order_params = nn.ParameterList([
                nn.Parameter(torch.empty(input_dim, rank, num_classes))
                for _ in range(order)
            ])
            self.factors.append(order_params)

        # Proxy projection weights for stage-1 gate optimization only.
        # Exported tnlearn neuron terms always use unit coefficients.
        self.coeffs_pure = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim, num_classes))
            for _ in range(poly_order)
        ])
        self.beta = nn.Parameter(torch.zeros(num_classes))
        self.reset_parameters()

    def reset_parameters(self):
        for order_params in self.factors:
            for param in order_params:
                nn.init.normal_(param, std=0.05)
        for param in self.coeffs_pure:
            nn.init.normal_(param, std=0.05)

    def get_pure_term(self, x: torch.Tensor, order_idx: int) -> torch.Tensor:
        order = order_idx + 1
        term = x if order == 1 else x.pow(order)
        return term @ self.coeffs_pure[order_idx]

    def get_interaction_term(self, x: torch.Tensor, order_idx: int) -> torch.Tensor:
        factors = self.factors[order_idx]
        projections = [torch.einsum('bd, drc -> brc', x, factor) for factor in factors]
        combined = projections[0]
        for projection in projections[1:]:
            combined = combined * projection
        return torch.sum(combined, dim=1)


class L0Gate(nn.Module):
    """Hard-concrete L0 gate used to prune polynomial orders."""

    def __init__(self, temperature=0.66, limit_l=-0.1, limit_r=1.1, init_prob=0.9):
        super().__init__()
        self.temp = temperature
        self.limit_l = limit_l
        self.limit_r = limit_r
        init_val = np.log(init_prob / (1 - init_prob))
        self.log_alpha = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))

    def forward(self, x, training=True):
        if training:
            u = torch.rand_like(self.log_alpha)
            s = torch.sigmoid((torch.log(u + 1e-8) - torch.log(1 - u + 1e-8) + self.log_alpha) / self.temp)
            s = s * (self.limit_r - self.limit_l) + self.limit_l
        else:
            s = torch.sigmoid(self.log_alpha) * (self.limit_r - self.limit_l) + self.limit_l
        z = torch.clamp(s, min=0.0, max=1.0)
        return x * z

    def regularization_term(self):
        return torch.sigmoid(self.log_alpha - self.temp * np.log(-self.limit_l / self.limit_r))

    def get_prob(self):
        return torch.sigmoid(self.log_alpha).item()


class SparseSearchAgent(nn.Module):
    """Differentiable NeuronSeek-TD structure search agent."""

    def __init__(self, input_dim=10, num_classes=1, rank=8, max_order=5):
        super().__init__()
        self.input_dim = input_dim
        self.max_order = max_order
        self.core = DualStreamInteractionLayer(input_dim, num_classes, rank, max_order)
        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.gates_pure = nn.ModuleList([L0Gate() for _ in range(max_order)])
        self.gates_int = nn.ModuleList([L0Gate() for _ in range(max_order)])
        self.bn_pure = nn.ModuleList(nn.BatchNorm1d(num_classes, affine=True) for _ in range(max_order))
        self.bn_int = nn.ModuleList(nn.BatchNorm1d(num_classes, affine=True) for _ in range(max_order))

    def forward(self, x, training=True):
        output = self.bias.unsqueeze(0).expand(x.size(0), -1).clone()

        for i, gate in enumerate(self.gates_pure):
            term = self.core.get_pure_term(x, i)
            output = output + gate(self.bn_pure[i](term), training=training)

        for i, gate in enumerate(self.gates_int):
            term = self.core.get_interaction_term(x, i)
            output = output + gate(self.bn_int[i](term), training=training)

        return output

    def get_structure(self, threshold=0.5):
        pure_active = []
        interact_active = []
        with torch.no_grad():
            for i, gate in enumerate(self.gates_pure):
                if gate.regularization_term() > threshold:
                    pure_active.append(i + 1)
            for i, gate in enumerate(self.gates_int):
                if gate.regularization_term() > threshold:
                    interact_active.append(i + 1)
        return pure_active, interact_active

    def calculate_regularization(self):
        reg_loss = 0.0
        for gate in self.gates_pure:
            reg_loss = reg_loss + gate.regularization_term()
        for gate in self.gates_int:
            reg_loss = reg_loss + gate.regularization_term()
        return reg_loss


class PolyTensorRegressor(nn.Module):
    """
    NeuronSeek-TD stage-1 polynomial term searcher for tnlearn.

    The search process follows the revised NeuronSeek dual-stream design to
    select active polynomial orders with differentiable L0 gates. tnlearn's
    second stage learns its own layer weights, so the exported ``neuron``
    contains selected terms with unit coefficients only.
    """

    def __init__(self,
                 rank=8,
                 poly_order=5,
                 method='cp',
                 reg_lambda_w=0.01,
                 reg_lambda_c=0.05,
                 num_epochs=100,
                 learning_rate=0.01,
                 batch_size=64,
                 task_type='regression',
                 num_classes: Optional[int] = None,
                 device: Optional[torch.device] = None,
                 track_callback=None,
                 random_state: Optional[int] = None,
                 structure_threshold=0.5):
        super().__init__()
        if random_state is not None:
            random_seed(random_state)
        self.rank = rank
        self.poly_order = poly_order
        self.method = method
        self.reg_lambda_w = reg_lambda_w
        self.reg_lambda_c = reg_lambda_c
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.task_type = task_type
        self.num_classes = num_classes
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.track_callback = track_callback
        self.random_state = random_state
        self.structure_threshold = structure_threshold
        self.agent = None
        self.neuron = None
        self.structure_ = None
        self.terms_ = []
        self.logs_ = {'loss': [], 'lambda_val': []}

    def _prepare_tensors(self, X, y):
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X_tensor = X.to(self.device, dtype=torch.float32)
        X_tensor = X_tensor.view(X_tensor.size(0), -1)

        if not isinstance(y, torch.Tensor):
            y_tensor = torch.tensor(y, device=self.device)
        else:
            y_tensor = y.to(self.device)

        if self.task_type == 'classification':
            y_tensor = y_tensor.long().view(-1)
            if self.num_classes is None:
                self.num_classes = len(torch.unique(y_tensor))
        else:
            y_tensor = y_tensor.float().view(-1, 1)
            self.num_classes = 1

        return X_tensor, y_tensor

    def fit(self, X, y, view_training_process=False):
        X_tensor, y_tensor = self._prepare_tensors(X, y)
        input_dim = X_tensor.shape[1]
        output_dim = int(self.num_classes or 1)

        self.agent = SparseSearchAgent(
            input_dim=input_dim,
            num_classes=output_dim,
            rank=self.rank,
            max_order=self.poly_order,
        ).to(self.device)

        loss_fn = nn.CrossEntropyLoss() if self.task_type == 'classification' else nn.MSELoss()
        optimizer = torch.optim.Adam([
            {'params': self.agent.core.coeffs_pure.parameters(), 'lr': self.learning_rate * 0.5, 'weight_decay': 1e-4},
            {'params': [self.agent.core.beta], 'lr': self.learning_rate * 0.5},
            {'params': self.agent.core.factors.parameters(), 'lr': self.learning_rate, 'weight_decay': 1e-5},
            {'params': list(self.agent.bn_pure.parameters()) + list(self.agent.bn_int.parameters()), 'lr': self.learning_rate},
            {'params': list(self.agent.gates_pure.parameters()) + list(self.agent.gates_int.parameters()), 'lr': self.learning_rate},
        ])
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-5)

        batch_size = min(self.batch_size, len(X_tensor))
        drop_last = len(X_tensor) > batch_size
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)

        warmup_end = int(self.num_epochs * 0.25)
        anneal_end = max(warmup_end + 1, int(self.num_epochs * 0.75))
        max_lambda = self.reg_lambda_c

        self.agent.train()
        losses = []
        for epoch in range(self.num_epochs):
            current_lambda = 0.0
            if epoch >= warmup_end:
                if epoch < anneal_end:
                    progress = (epoch - warmup_end) / (anneal_end - warmup_end)
                    current_lambda = max_lambda * progress
                else:
                    current_lambda = max_lambda

            is_frozen = epoch < warmup_end
            for param in self.agent.gates_pure.parameters():
                param.requires_grad = not is_frozen
            for param in self.agent.gates_int.parameters():
                param.requires_grad = not is_frozen

            total_loss = 0.0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                preds = self.agent(batch_x, training=True)
                task_loss = loss_fn(preds, batch_y)
                reg_loss = current_lambda * self.agent.calculate_regularization()
                loss = task_loss + reg_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            epoch_loss = total_loss / max(1, len(dataloader))
            losses.append(epoch_loss)
            self.logs_['loss'].append(epoch_loss)
            self.logs_['lambda_val'].append(current_lambda)

            if self.track_callback:
                self.track_callback(self._export_current_neuron())

            print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {epoch_loss:.4f}, Lambda: {current_lambda:.4f}')

        self.structure_ = self.get_structure_info()
        self.terms_ = self._structure_to_terms(self.structure_)
        self.neuron = self._terms_to_neuron(self.terms_)

        if view_training_process:
            import matplotlib.pyplot as plt

            plt.plot(losses)
            plt.show()

        return self

    def forward(self, X):
        if self.agent is None:
            raise RuntimeError("PolyTensorRegressor must be fitted before calling forward().")
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X_tensor = X.to(self.device, dtype=torch.float32)
        X_tensor = X_tensor.view(X_tensor.size(0), -1)
        output = self.agent(X_tensor, training=self.training)
        return output, self.agent.calculate_regularization()

    def predict(self, X):
        if self.agent is None:
            raise RuntimeError("PolyTensorRegressor must be fitted before calling predict().")
        self.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            else:
                X_tensor = X.to(self.device, dtype=torch.float32)
            X_tensor = X_tensor.view(X_tensor.size(0), -1)
            outputs = self.agent(X_tensor, training=False)
            if self.task_type == 'classification':
                return torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            return outputs.view(-1)

    def get_structure_info(self):
        if self.agent is None:
            return {
                'type': 'neuronseek',
                'pure_indices': [],
                'interact_indices': [],
                'rank': self.rank,
            }
        pure_indices, interact_indices = self.agent.get_structure(threshold=self.structure_threshold)
        return {
            'type': 'neuronseek',
            'pure_indices': pure_indices,
            'interact_indices': interact_indices,
            'rank': self.rank,
        }

    def get_significant_polynomial(self):
        if self.structure_ is None:
            self.structure_ = self.get_structure_info()
        self.terms_ = self._structure_to_terms(self.structure_)
        self.neuron = self._terms_to_neuron(self.terms_)
        return self.neuron

    def _export_current_neuron(self):
        structure = self.get_structure_info()
        terms = self._structure_to_terms(structure)
        return self._terms_to_neuron(terms)

    def _structure_to_terms(self, structure):
        pure_indices = structure.get('pure_indices', [])
        interact_indices = structure.get('interact_indices', [])
        orders = sorted(set(pure_indices) | set(interact_indices))
        if not orders:
            orders = [1]
        return [
            {
                'coefficient': 1,
                'order': order,
                'expression': 'x' if order == 1 else f'x**{order}',
                'source': self._term_source(order, pure_indices, interact_indices),
            }
            for order in orders
        ]

    @staticmethod
    def _term_source(order, pure_indices, interact_indices):
        in_pure = order in pure_indices
        in_interact = order in interact_indices
        if in_pure and in_interact:
            return 'pure+interact'
        if in_interact:
            return 'interact'
        return 'pure'

    @staticmethod
    def _terms_to_neuron(terms):
        return ' + '.join(f"{term['coefficient']}@{term['expression']}" for term in terms)


PolyTensorRegression = PolyTensorRegressor
PolynomialTensorRegression = PolyTensorRegressor
