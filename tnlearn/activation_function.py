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


def get_activation_function(name):
    r"""Get the corresponding PyTorch activation function by name.

    Args:
        name: A string name of the activation function.

    Returns:
        The corresponding PyTorch activation function.
    """

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
