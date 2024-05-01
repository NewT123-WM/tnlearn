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

from torch import nn


def get_loss_function(name):
    r"""Get the corresponding PyTorch loss function by name.

    Args:
        name: A string name of the loss function.

    Returns:
        The corresponding PyTorch loss function
    """
    activations = {
        'mse': nn.MSELoss(),
        'l1': nn.L1Loss(),
        'crossentropy': nn.CrossEntropyLoss(),
        'bce': nn.BCELoss(),
        'nll': nn.NLLLoss(),
        'poissonnll': nn.PoissonNLLLoss(),
        'marginranking': nn.MarginRankingLoss(),
        'smoothl1': nn.SmoothL1Loss(),

        # Add more activation functions if needed
    }
    # Return the requested loss function or MSELoss as default
    return activations.get(name.lower(), nn.MSELoss())
