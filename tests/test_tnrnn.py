import torch
import tempfile
import os
from tnlearn import (
    TNRNN, TNLSTM, TNGRU,
    TNRNNCell, TNLSTMCell, TNGRUCell
)


def compare_tensors(out1, out2, atol=1e-6):
    """递归比较两个输出，支持嵌套元组/列表"""
    if isinstance(out1, torch.Tensor):
        assert torch.allclose(out1, out2, atol=atol), f"Tensor mismatch: {out1} vs {out2}"
    elif isinstance(out1, (tuple, list)):
        assert len(out1) == len(out2), f"Length mismatch: {len(out1)} vs {len(out2)}"
        for a, b in zip(out1, out2):
            compare_tensors(a, b, atol)
    else:
        raise TypeError(f"Unsupported type: {type(out1)}")


def test_save_load(model, input_args, hx_args=None, is_cell=False):
    """
    测试模型的保存和加载
    model: 待测试的模型
    input_args: 传递给 forward 的输入张量（或元组）
    hx_args: 可选的初始状态
    is_cell: 是否为 Cell（Cell 返回元组）
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 执行一次前向+反向更新
    optimizer.zero_grad()
    if hx_args is not None:
        if is_cell:
            # Cell 的输入可能是 (input, hx) 或 (input, (hx, cx))
            if isinstance(hx_args, tuple) and len(hx_args) == 2 and isinstance(hx_args[0], torch.Tensor):
                # LSTM Cell: (h0, c0)
                output = model(input_args, hx_args)
            else:
                output = model(input_args, hx_args)
        else:
            output = model(input_args, hx_args)
    else:
        output = model(input_args)

    # 使用输出的总和作为损失（简单反向）
    def sum_output(out):
        if isinstance(out, torch.Tensor):
            return out.sum()
        elif isinstance(out, (tuple, list)):
            return sum(sum_output(o) for o in out)
        else:
            return 0.0

    loss = sum_output(output)
    loss.backward()
    optimizer.step()

    # 记录更新后的输出（eval模式）
    model.eval()
    with torch.no_grad():
        if hx_args is not None:
            if is_cell:
                if isinstance(hx_args, tuple) and len(hx_args) == 2 and isinstance(hx_args[0], torch.Tensor):
                    out_after = model(input_args, hx_args)
                else:
                    out_after = model(input_args, hx_args)
            else:
                out_after = model(input_args, hx_args)
        else:
            out_after = model(input_args)

    # 保存完整模型
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        torch.save(model, f.name)
        path = f.name

    # 加载模型
    loaded_model = torch.load(path,weights_only=False)
    loaded_model.eval()

    # 比较输出（递归比较，支持嵌套元组）
    with torch.no_grad():
        if hx_args is not None:
            if is_cell:
                if isinstance(hx_args, tuple) and len(hx_args) == 2 and isinstance(hx_args[0], torch.Tensor):
                    out_loaded = loaded_model(input_args, hx_args)
                else:
                    out_loaded = loaded_model(input_args, hx_args)
            else:
                out_loaded = loaded_model(input_args, hx_args)
        else:
            out_loaded = loaded_model(input_args)

    compare_tensors(out_after, out_loaded, atol=1e-6)

    # 比较参数
    for (name1, p1), (name2, p2) in zip(model.state_dict().items(), loaded_model.state_dict().items()):
        assert torch.allclose(p1, p2, atol=1e-6), f"Parameter mismatch for {name1}"

    print(f"✅ Save/load test passed for {type(model).__name__}")
    os.unlink(path)


def test_rnn_models():
    # 1. TNRNN
    model = TNRNN(input_size=10, hidden_size=20, num_layers=2, batch_first=True,
                  neuron_expression='x + 0.5@torch.sin(x)')
    x = torch.randn(3, 5, 10)  # (batch, seq, feature)
    test_save_load(model, x)

    # 2. TNLSTM
    model = TNLSTM(input_size=10, hidden_size=20, num_layers=2, bidirectional=True,
                   neuron_expression='x**2 + torch.cos(x)')
    x = torch.randn(3, 5, 10)
    test_save_load(model, x)

    # 3. TNGRU
    model = TNGRU(input_size=10, hidden_size=20, num_layers=2, batch_first=False,
                  neuron_expression='x')
    x = torch.randn(5, 3, 10)  # (seq, batch, feature)
    test_save_load(model, x)

    # 4. TNRNNCell
    cell = TNRNNCell(input_size=10, hidden_size=20, nonlinearity='relu',
                     neuron_expression='x + torch.sin(x)')
    x_t = torch.randn(3, 10)
    h = torch.randn(3, 20)
    test_save_load(cell, x_t, hx_args=h, is_cell=True)

    # 5. TNLSTMCell
    cell = TNLSTMCell(input_size=10, hidden_size=20,
                      neuron_expression='x**3')
    x_t = torch.randn(3, 10)
    h = torch.randn(3, 20)
    c = torch.randn(3, 20)
    test_save_load(cell, x_t, hx_args=(h, c), is_cell=True)

    # 6. TNGRUCell
    cell = TNGRUCell(input_size=10, hidden_size=20,
                     neuron_expression='x + 0.1@torch.exp(x)*x**2')
    x_t = torch.randn(3, 10)
    h = torch.randn(3, 20)
    test_save_load(cell, x_t, hx_args=h, is_cell=True)

    print("All tests passed! ✅")


if __name__ == '__main__':
    test_rnn_models()