import torch
import tempfile
import os
from tnlearn import (
    TNRNN, TNLSTM, TNGRU,
    TNRNNCell, TNLSTMCell, TNGRUCell
)


def compare_tensors(out1, out2, atol=1e-6):
    """Recursively compare two outputs, supporting nested tuples/lists."""
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
    Test model saving and loading.

    Args:
        model: The model to test.
        input_args: Input tensor(s) to pass to forward.
        hx_args: Optional initial hidden/cell state.
        is_cell: Whether the model is a cell (returns a tuple).
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Perform one forward + backward update
    optimizer.zero_grad()
    if hx_args is not None:
        if is_cell:
            # Cell input may be (input, hx) or (input, (hx, cx))
            if isinstance(hx_args, tuple) and len(hx_args) == 2 and isinstance(hx_args[0], torch.Tensor):
                # LSTM Cell: (h0, c0)
                output = model(input_args, hx_args)
            else:
                output = model(input_args, hx_args)
        else:
            output = model(input_args, hx_args)
    else:
        output = model(input_args)

    # Use the sum of the output as loss (simple backward)
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

    # Record the output after update (eval mode)
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

    # Save the full model
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        torch.save(model, f.name)
        path = f.name

    # Load the model
    try:
        loaded_model = torch.load(path, weights_only=False)
    except TypeError:
        loaded_model = torch.load(path)
        
    loaded_model.eval()

    # Compare outputs (recursive, supports nested tuples)
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

    # Compare parameters
    for (name1, p1), (name2, p2) in zip(model.state_dict().items(), loaded_model.state_dict().items()):
        assert torch.allclose(p1, p2, atol=1e-6), f"Parameter mismatch for {name1}"

    print(f"✅ Save/load test passed for {type(model).__name__}")
    os.unlink(path)


def test_rnn_models():
    # 1. TNRNN
    model = TNRNN(input_size=10, hidden_size=20, num_layers=2, batch_first=True,
                  symbolic_expression='x + 0.5@torch.sin(x)')
    x = torch.randn(3, 5, 10)  # (batch, seq, feature)
    test_save_load(model, x)

    # 2. TNLSTM
    model = TNLSTM(input_size=10, hidden_size=20, num_layers=2, bidirectional=True,
                   symbolic_expression='x**2 + torch.cos(x)')
    x = torch.randn(3, 5, 10)
    test_save_load(model, x)

    # 3. TNGRU
    model = TNGRU(input_size=10, hidden_size=20, num_layers=2, batch_first=False,
                  symbolic_expression='x')
    x = torch.randn(5, 3, 10)  # (seq, batch, feature)
    test_save_load(model, x)

    # 4. TNRNNCell
    cell = TNRNNCell(input_size=10, hidden_size=20, nonlinearity='relu',
                     symbolic_expression='x + torch.sin(x)')
    x_t = torch.randn(3, 10)
    h = torch.randn(3, 20)
    test_save_load(cell, x_t, hx_args=h, is_cell=True)

    # 5. TNLSTMCell
    cell = TNLSTMCell(input_size=10, hidden_size=20,
                      symbolic_expression='x**3')
    x_t = torch.randn(3, 10)
    h = torch.randn(3, 20)
    c = torch.randn(3, 20)
    test_save_load(cell, x_t, hx_args=(h, c), is_cell=True)

    # 6. TNGRUCell
    cell = TNGRUCell(input_size=10, hidden_size=20,
                     symbolic_expression='x + 0.1@torch.exp(x)*x**2')
    x_t = torch.randn(3, 10)
    h = torch.randn(3, 20)
    test_save_load(cell, x_t, hx_args=h, is_cell=True)

    print("All tests passed! ✅")


if __name__ == '__main__':
    test_rnn_models()