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
import torch.optim as optim


def get_optimizer(name, parameters, lr=0.001, **kwargs):
    r"""Get the corresponding PyTorch optimizer by name.

    Args:
        name: The name of the optimizer (e.g., 'adam', 'sgd').
        parameters: The parameters of the model to optimize.
        lr: Learning rate.
        kwargs: Other arguments specific to the optimizer.

    Returns:
        An instance of the requested optimizer.
    """

    optimizers = {
        'adam': optim.Adam(params=parameters, lr=lr, **kwargs),
        'sgd': optim.SGD(params=parameters, lr=lr, **kwargs),
        'rmsprop': optim.RMSprop(params=parameters, lr=lr, **kwargs),
        'adamw': optim.AdamW(params=parameters, lr=lr, **kwargs),
        'asgd': optim.ASGD(params=parameters, lr=lr, **kwargs),
        'adagrad': optim.Adagrad(params=parameters, lr=lr, **kwargs),
        'adamax': optim.Adamax(params=parameters, lr=lr, **kwargs),

        # Add more optimizers if needed
    }

    # Return the requested optimizer or Adam as default
    return optimizers.get(name.lower(), optim.Adam(parameters, lr=lr))