# Copyright 2026 Tieyun LI. All Rights Reserved.
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
"""Custom modules: including fully-connected, convolutional, recurrent, 
and transformer layers supporting symbolic expressions."""

from .TNlinear import TNLinear
from .TNconv import (
    TNConv1d,
    TNConv2d,
    TNConv3d,
    TNConvTranspose1d,
    TNConvTranspose2d,
    TNConvTranspose3d,
)
from .TNrnn import (
    TNRNNBase,
    TNRNN,
    TNLSTM,
    TNGRU,
    TNRNNCellBase,
    TNRNNCell,
    TNLSTMCell,
    TNGRUCell,
)
from .TNtransformer import (
    TNTransformer,
    TNTransformerEncoder,
    TNTransformerDecoder,
    TNTransformerEncoderLayer,
    TNTransformerDecoderLayer

)

__all__ = [
    # Fully-connected layers
    'TNLinear',
    # Convolutional layers
    'TNConv1d',
    'TNConv2d',
    'TNConv3d',
    'TNConvTranspose1d',
    'TNConvTranspose2d',
    'TNConvTranspose3d',
    # RNN related
    'TNRNNBase',
    'TNRNN',
    'TNLSTM',
    'TNGRU',
    'TNRNNCellBase',
    'TNRNNCell',
    'TNLSTMCell',
    'TNGRUCell',
    # Transformer related
    'TNTransformer',
    'TNTransformerEncoder',
    'TNTransformerDecoder',
    'TNTransformerEncoderLayer',
    'TNTransformerDecoderLayer'
]