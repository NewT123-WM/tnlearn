# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This file is based on the DRSR project (https://github.com/scientific-intelligent-modelling/drsr)
# and has been modified for vectorized symbolic regression.

# drsr/template.py
SINGLE_VAR_TEMPLATE = '''
import numpy as np
from scipy.optimize import minimize

MAX_NPARAMS = {max_params}
N_RESTARTS = {n_restarts}
BFGS_MAXITER = {bfgs_maxiter}

def equation(x, params):
    """
    Single-variable mathematical expression.
    x: scalar or 1D array (will be vectorized)
    params: 1D array of coefficients
    """
    {equation_body}

def evaluate_run(data):
    inputs = data['inputs']
    outputs = data['outputs']

    def multivariate_predict(X, params):
        pred = np.zeros(X.shape[0])
        for i in range(X.shape[1]):
            pred += equation(X[:, i], params)
        return pred

    def loss(params):
        y_pred = multivariate_predict(inputs, params)
        if outputs.ndim == 2 and outputs.shape[1] > 1:
            y_pred_expanded = np.tile(y_pred[:, None], (1, outputs.shape[1]))
        else:
            y_pred_expanded = y_pred
        return np.mean((y_pred_expanded - outputs) ** 2)

    best_loss = np.inf
    best_params = None
    for _ in range(N_RESTARTS):
        x0 = np.random.uniform(-1, 1, size=MAX_NPARAMS)
        try:
            res = minimize(loss, x0, method='BFGS',
                          options={{'maxiter': BFGS_MAXITER, 'disp': False}})
            if res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
        except:
            continue

    if best_params is None:
        return -np.inf, None, None

    y_pred = multivariate_predict(inputs, best_params)
    if outputs.ndim == 2 and outputs.shape[1] > 1:
        y_pred_expanded = np.tile(y_pred[:, None], (1, outputs.shape[1]))
        residuals = outputs - y_pred_expanded
    else:
        residuals = outputs - y_pred
    var_y = np.var(outputs)
    nmse = best_loss / var_y if var_y > 0 else np.inf

    if outputs.ndim == 1:
        outputs_2d = outputs[:, None]
        residuals_2d = residuals[:, None]
    else:
        outputs_2d = outputs
        residuals_2d = residuals
    result_data = np.column_stack((inputs, outputs_2d, residuals_2d))
    return -nmse, result_data, best_params
'''