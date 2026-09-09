"""Regression checks for NeuronSeek structure export and the base MLP API."""

import numpy as np
import pytest
import sympy as sp
import torch
from sklearn.datasets import make_regression

import tnlearn
from tnlearn import MLPRegressor, PolyTensorRegression, PolyTensorRegressor
from tnlearn.mlpregressor import BaseCustomNeuronLayer
from tnlearn.operator.inner_product import (
    InnerProduct,
    convert_pretty_to_innerproduct,
    neuronseek_config_to_string,
)
from tnlearn.poly_regressor import SparseSearchAgent


CONFIGS = [
    dict(type='neuronseek', pure_indices=[1, 2], interact_indices=[2, 3],
         interaction_form='cp_inner_product', rank=8),
    dict(pure_indices=[1, 3, 5], interact_indices=[]),
    dict(pure_indices=[], interact_indices=[2, 4]),
    dict(pure_indices=[1, 2], interact_indices=[2], periodic=True),
]


@pytest.fixture(autouse=True)
def small_cpu_tests():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_existing_package_exports_remain_available():
    assert PolyTensorRegression is not PolyTensorRegressor
    for name in ('GPSymRegressor', 'VecSymRegressor', 'LLMSymRegressor',
                 'RLRegressor', 'RLSymRegressor', 'TNLinear', 'TNTransformer',
                 'PolyTensorRegression', 'PolyTensorRegressor'):
        assert name in tnlearn.__all__
        assert hasattr(tnlearn, name)


@pytest.mark.parametrize('config', CONFIGS)
def test_colleague_configs_fit_and_predict(config):
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=1)
    neuron = neuronseek_config_to_string(config)
    # Two optimizer steps validate the public fit path without a benchmark run.
    model = MLPRegressor(neuron, layers_list=[4], max_iter=2)
    model.fit(X, y)
    assert model.predict(X).shape == (100,)
    assert np.isfinite(model.predict(X)).all()
    assert np.isfinite(model.losses).all()


def test_pure_and_interaction_terms_are_mathematically_distinct():
    pure = BaseCustomNeuronLayer(2, 1, neuronseek_config_to_string(
        dict(pure_indices=[2])), bias=False)
    interaction = BaseCustomNeuronLayer(2, 1, neuronseek_config_to_string(
        dict(interact_indices=[2])), bias=False)
    with torch.no_grad():
        for layer in (pure, interaction):
            for parameter in layer.parameters():
                parameter.fill_(1)
    x = torch.tensor([[2., 3.]])
    torch.testing.assert_close(pure(x), torch.tensor([[13.]]))
    torch.testing.assert_close(interaction(x), torch.tensor([[25.]]))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_cp_rank_preserves_independent_components_and_gradients(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    neuron = neuronseek_config_to_string(CONFIGS[0])
    expr = sp.sympify(convert_pretty_to_innerproduct(neuron),
                     locals={'InnerProduct': InnerProduct})
    # Two pure terms plus 8 independent components for each interaction order.
    assert len(sp.Add.make_args(expr)) == 2 + 8 + 8
    assert len(expr.free_symbols - {sp.Symbol('x')}) == 2 + 8 * (2 + 3)
    layer = BaseCustomNeuronLayer(2, 1, neuron, bias=False).to(device)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.fill_(1)
    x = torch.tensor([[2., 3.], [1., 2.]], device=device)
    # <1,x> + <1,x**2> + 8*<1,x>**2 + 8*<1,x>**3
    expected = x.sum(1) + x.square().sum(1) + 8*x.sum(1)**2 + 8*x.sum(1)**3
    torch.testing.assert_close(layer(x).flatten(), expected)
    layer(x).sum().backward()
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_rank_one_and_missing_fields_keep_existing_format():
    assert neuronseek_config_to_string(dict(pure_indices=[1], interact_indices=[2])) == (
        '<w1, x>+<w2, x>*<w3, x>')
    assert neuronseek_config_to_string({}) == ''
    assert neuronseek_config_to_string(dict(pure_indices=[True, 0, -1, 2])) == '<w1, x**2>'


@pytest.mark.parametrize('rank', [0, -1, 1.5, '8', True, None])
def test_invalid_cp_rank_is_rejected(rank):
    with pytest.raises(ValueError, match='rank'):
        neuronseek_config_to_string(dict(interact_indices=[2], rank=rank))


def test_export_paths_preserve_structure_without_training():
    reg = PolyTensorRegressor(rank=2, poly_order=3, device='cpu')
    reg.agent = SparseSearchAgent(input_dim=2, rank=2, max_order=3)
    with torch.no_grad():
        for gate in list(reg.agent.gates_pure) + list(reg.agent.gates_int):
            gate.log_alpha.fill_(-100)
        reg.agent.gates_pure[1].log_alpha.fill_(100)
        reg.agent.gates_int[1].log_alpha.fill_(100)
    expected = neuronseek_config_to_string(dict(
        pure_indices=[2], interact_indices=[2], rank=2))
    assert reg._export_current_neuron() == expected
    assert reg.get_significant_polynomial() == expected
    assert reg.structure_['pure_indices'] == reg.structure_['interact_indices'] == [2]
    assert '@' not in reg.neuron


def test_empty_search_exports_zero_and_builds_bias_only_mlp():
    reg = PolyTensorRegressor(rank=1, poly_order=1, device='cpu')
    reg.agent = SparseSearchAgent(input_dim=2, rank=1, max_order=1)
    with torch.no_grad():
        reg.agent.gates_pure[0].log_alpha.fill_(-100)
        reg.agent.gates_int[0].log_alpha.fill_(-100)
    assert reg.get_significant_polynomial() == reg._export_current_neuron() == '0'
    layer = BaseCustomNeuronLayer(2, 3, reg.neuron, bias=False)
    torch.testing.assert_close(layer(torch.randn(4, 2)), torch.zeros(4, 3))
    X = np.ones((4, 2), dtype=np.float32)
    model = MLPRegressor(reg.neuron, layers_list=[3], max_iter=2)
    model.fit(X, np.ones(4))
    assert model.predict(X).shape == (4,)


@pytest.mark.parametrize('task_type', ['regression', 'classification'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_search_fit_export_callback_predict_and_refit(task_type, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    X = np.random.default_rng(1).normal(size=(9, 3)).astype(np.float32)
    y = X[:, 0] + 3 if task_type == 'regression' else np.arange(9) % 2
    callbacks = []
    reg = PolyTensorRegressor(rank=2, poly_order=2, num_epochs=2, batch_size=4,
                              device=device, random_state=1, task_type=task_type,
                              track_callback=callbacks.append)
    assert reg.fit(X, y) is reg
    assert reg.neuron == neuronseek_config_to_string(reg.structure_)
    assert reg.neuron == reg.get_significant_polynomial() == callbacks[-1]
    assert reg.agent.bias.detach().abs().sum() > 0
    assert np.isfinite(reg.logs_['loss']).all()
    first = reg.predict(X)
    assert first.shape == (9,)
    assert torch.isfinite(first).all()
    model = MLPRegressor(reg.neuron, layers_list=[3], max_iter=2)
    model.fit(X, y)
    assert np.isfinite(model.predict(X)).all()
    reg.fit(X, y)
    assert len(reg.logs_['loss']) == 2
    assert len(callbacks) == 4
    torch.testing.assert_close(first, reg.predict(X))


@pytest.mark.parametrize('kwargs', [dict(method='tucker'), dict(batch_size=1),
                                    dict(rank=0), dict(structure_threshold=2)])
def test_unsupported_search_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        PolyTensorRegressor(**kwargs)


@pytest.mark.parametrize('reg_lambda_w', [0.0, 0.1])
def test_weight_regularization_is_controlled_by_reg_lambda_w(monkeypatch, reg_lambda_w):
    initial_weights = {}
    original_forward = SparseSearchAgent.forward

    def zero_task_output(agent, *args, **kwargs):
        initial_weights.update({name: p.detach().clone()
                                for name, p in agent.core.named_parameters()})
        # Keep the real forward/backward path, but eliminate task gradients.
        return original_forward(agent, *args, **kwargs) * 0

    monkeypatch.setattr(SparseSearchAgent, 'forward', zero_task_output)
    X = np.random.default_rng(1).normal(size=(4, 3)).astype(np.float32)
    reg = PolyTensorRegressor(rank=2, poly_order=2, num_epochs=1, batch_size=4,
                              reg_lambda_w=reg_lambda_w, reg_lambda_c=0,
                              learning_rate=1e-5, device='cpu', random_state=1)
    reg.fit(X, np.zeros(4, dtype=np.float32))
    assert initial_weights
    for name, parameter in reg.agent.core.named_parameters():
        before = initial_weights[name]
        if reg_lambda_w == 0:
            torch.testing.assert_close(parameter, before, rtol=0, atol=0)
        else:
            assert parameter.detach().abs().sum() < before.abs().sum()


def test_singleton_and_misaligned_training_data_are_rejected():
    reg = PolyTensorRegressor(device='cpu')
    with pytest.raises(ValueError, match='at least two'):
        reg.fit(np.ones((1, 2)), np.ones(1))
    with pytest.raises(ValueError, match='same sample count'):
        reg.fit(np.ones((3, 2)), np.ones(2))
