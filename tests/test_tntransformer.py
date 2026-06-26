import torch
import tempfile
import os
from tnlearn import (
    TNTransformer,
    TNTransformerEncoder,
    TNTransformerDecoder,
    TNTransformerEncoderLayer,
    TNTransformerDecoderLayer
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


def test_model_save_load(model, args, kwargs=None):
    """
    Generic save-load test.

    Args:
        model: The model to test.
        args:  Positional arguments to pass to forward.
        kwargs: Keyword arguments to pass to forward (optional).
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Forward and backward update
    optimizer.zero_grad()
    if kwargs is None:
        output = model(*args)
    else:
        output = model(*args, **kwargs)

    # Sum outputs as loss (supports nested outputs)
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
        if kwargs is None:
            out_after = model(*args)
        else:
            out_after = model(*args, **kwargs)

    # Save the full model
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        torch.save(model, f.name)
        path = f.name

    # Load the model (compatible with older PyTorch)
    try:
        loaded_model = torch.load(path, weights_only=False)
    except TypeError:
        loaded_model = torch.load(path)

    loaded_model.eval()

    # Compare outputs before and after loading
    with torch.no_grad():
        if kwargs is None:
            out_loaded = loaded_model(*args)
        else:
            out_loaded = loaded_model(*args, **kwargs)

    compare_tensors(out_after, out_loaded, atol=1e-6)

    # Compare all parameters
    for (name1, p1), (name2, p2) in zip(model.state_dict().items(), loaded_model.state_dict().items()):
        assert torch.allclose(p1, p2, atol=1e-6), f"Parameter mismatch for {name1}"

    print(f"✅ Save/load test passed for {type(model).__name__}")
    os.unlink(path)


def test_transformer_models():
    # Fix random seed for reproducibility
    torch.manual_seed(42)

    # ---------- 1. TNTransformerEncoderLayer ----------
    # batch_first=False (default)
    encoder_layer = TNTransformerEncoderLayer(
        d_model=512, nhead=8, dim_feedforward=2048,
        dropout=0.1, activation='relu',
        neuron_expression='x + torch.sin(x)'
    )
    src = torch.randn(10, 32, 512)   # (seq, batch, feature)
    # forward signature: (src, src_mask=None, src_key_padding_mask=None, is_causal=False)
    test_model_save_load(encoder_layer, (src,))

    # ---------- 2. TNTransformerDecoderLayer ----------
    decoder_layer = TNTransformerDecoderLayer(
        d_model=512, nhead=8, dim_feedforward=2048,
        dropout=0.1, activation='gelu',
        neuron_expression='x**2 + torch.cos(x)'
    )
    tgt = torch.randn(20, 32, 512)
    memory = torch.randn(10, 32, 512)
    # forward signature: (tgt, memory, tgt_mask=None, memory_mask=None, ...)
    test_model_save_load(decoder_layer, (tgt, memory))

    # ---------- 3. TNTransformerEncoder ----------
    # Stack 2 layers with LayerNorm
    enc_layer = TNTransformerEncoderLayer(
        d_model=512, nhead=8, dim_feedforward=2048,
        dropout=0.1, activation='relu',
        neuron_expression='x + 0.5 * torch.sin(x)'
    )
    encoder = TNTransformerEncoder(enc_layer, num_layers=2)
    src = torch.randn(10, 32, 512)
    test_model_save_load(encoder, (src,))

    # ---------- 4. TNTransformerDecoder ----------
    dec_layer = TNTransformerDecoderLayer(
        d_model=512, nhead=8, dim_feedforward=2048,
        dropout=0.1, activation='gelu',
        neuron_expression='x * torch.sigmoid(x)'
    )
    decoder = TNTransformerDecoder(dec_layer, num_layers=2)
    tgt = torch.randn(20, 32, 512)
    memory = torch.randn(10, 32, 512)
    test_model_save_load(decoder, (tgt, memory))

    # ---------- 5. TNTransformer (full model) ----------
    transformer = TNTransformer(
        d_model=512, nhead=8,
        num_encoder_layers=2, num_decoder_layers=2,
        dim_feedforward=2048, dropout=0.1,
        activation='relu',
        neuron_expression='x + torch.tanh(x)',
        batch_first=False  # default
    )
    src = torch.randn(10, 32, 512)
    tgt = torch.randn(20, 32, 512)
    # forward has many optional args; pass only required src, tgt
    test_model_save_load(transformer, (src, tgt))

    print("All Transformer tests passed! ✅")


if __name__ == '__main__':
    test_transformer_models()