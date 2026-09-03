"""
Full test script for TNTransformer components with serialization support.

This script tests all Transformer modules using torch.save/load.
It assumes TNLinear has been patched with __getstate__/__setstate__.
"""

import torch
import tempfile
import os
from typing import Optional, Tuple, List, Union, Any
from tnlearn import (
    TNTransformer,
    TNTransformerEncoder,
    TNTransformerDecoder,
    TNTransformerEncoderLayer,
    TNTransformerDecoderLayer
)


# ---------- Helper: compare tensors ----------
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


# ---------- Save/load test using torch.save/load ----------
def test_model_save_load(model, args, kwargs=None):
    """
    Generic save-load test using torch.save/load (requires model picklable).
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Forward + backward
    optimizer.zero_grad()
    if kwargs is None:
        output = model(*args)
    else:
        output = model(*args, **kwargs)

    # Loss
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

    # Record output after update
    model.eval()
    with torch.no_grad():
        if kwargs is None:
            out_after = model(*args)
        else:
            out_after = model(*args, **kwargs)

    # Save full model
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        torch.save(model, f.name)
        path = f.name

    # Load model
    try:
        loaded_model = torch.load(path, weights_only=False)
    except TypeError:
        loaded_model = torch.load(path)
    loaded_model.eval()

    # Compare outputs
    with torch.no_grad():
        if kwargs is None:
            out_loaded = loaded_model(*args)
        else:
            out_loaded = loaded_model(*args, **kwargs)

    compare_tensors(out_after, out_loaded, atol=1e-6)

    # Compare parameters (optional, but good to verify)
    for (name1, p1), (name2, p2) in zip(model.state_dict().items(), loaded_model.state_dict().items()):
        assert torch.allclose(p1, p2, atol=1e-6), f"Parameter mismatch for {name1}"

    print(f"✅ Save/load test passed for {type(model).__name__}")
    os.unlink(path)


# ---------- Main test function ----------
def test_transformer_models():
    """Test all Transformer components with TNLinear."""
    torch.manual_seed(42)

    # ---------- 1. TNTransformerEncoderLayer ----------
    print("Testing TNTransformerEncoderLayer (batch_first=False)...")
    encoder_layer = TNTransformerEncoderLayer(
        d_model=512, nhead=8, dim_feedforward=2048,
        dropout=0.1, activation='relu',
        symbolic_expression='x + sin(x)'
    )
    src = torch.randn(10, 32, 512)   # (seq, batch, feature)
    test_model_save_load(encoder_layer, (src,))

    # batch_first=True
    print("Testing TNTransformerEncoderLayer (batch_first=True)...")
    encoder_layer_bf = TNTransformerEncoderLayer(
        d_model=512, nhead=8, dim_feedforward=2048,
        dropout=0.1, activation='gelu',
        batch_first=True,
        symbolic_expression='x**2 + cos(x)'
    )
    src_bf = torch.randn(32, 10, 512)  # (batch, seq, feature)
    test_model_save_load(encoder_layer_bf, (src_bf,))

    # ---------- 2. TNTransformerDecoderLayer ----------
    print("Testing TNTransformerDecoderLayer...")
    decoder_layer = TNTransformerDecoderLayer(
        d_model=512, nhead=8, dim_feedforward=2048,
        dropout=0.1, activation='gelu',
        symbolic_expression='x * sin(x)'
    )
    tgt = torch.randn(20, 32, 512)
    memory = torch.randn(10, 32, 512)
    test_model_save_load(decoder_layer, (tgt, memory))

    # ---------- 3. TNTransformerEncoder (stack) ----------
    print("Testing TNTransformerEncoder...")
    enc_layer = TNTransformerEncoderLayer(
        d_model=512, nhead=8, dim_feedforward=2048,
        dropout=0.1, activation='relu',
        symbolic_expression='x + 0.5 * sin(x)'
    )
    encoder = TNTransformerEncoder(enc_layer, num_layers=2)
    src = torch.randn(10, 32, 512)
    test_model_save_load(encoder, (src,))

    # ---------- 4. TNTransformerDecoder (stack) ----------
    print("Testing TNTransformerDecoder...")
    dec_layer = TNTransformerDecoderLayer(
        d_model=512, nhead=8, dim_feedforward=2048,
        dropout=0.1, activation='gelu',
        symbolic_expression='x * tanh(x)'
    )
    decoder = TNTransformerDecoder(dec_layer, num_layers=2)
    tgt = torch.randn(20, 32, 512)
    memory = torch.randn(10, 32, 512)
    test_model_save_load(decoder, (tgt, memory))

    # ---------- 5. Full TNTransformer ----------
    print("Testing TNTransformer (full model)...")
    transformer = TNTransformer(
        d_model=512, nhead=8,
        num_encoder_layers=2, num_decoder_layers=2,
        dim_feedforward=2048, dropout=0.1,
        activation='relu',
        symbolic_expression='x + tanh(x)',
        batch_first=False
    )
    src = torch.randn(10, 32, 512)
    tgt = torch.randn(20, 32, 512)
    test_model_save_load(transformer, (src, tgt))

    # ---------- 6. Test with masks and batch_first=True ----------
    print("Testing TNTransformer with masks and batch_first=True...")
    transformer_bf = TNTransformer(
        d_model=512, nhead=8,
        num_encoder_layers=2, num_decoder_layers=2,
        dim_feedforward=2048, dropout=0.1,
        activation='gelu',
        symbolic_expression='x + 0.1*sin(x)',
        batch_first=True
    )
    src_bf = torch.randn(32, 10, 512)
    tgt_bf = torch.randn(32, 20, 512)

    tgt_mask = transformer_bf.generate_square_subsequent_mask(20)
    src_key_padding_mask = torch.randint(0, 2, (32, 10)).bool()
    tgt_key_padding_mask = torch.randint(0, 2, (32, 20)).bool()

    test_model_save_load(
        transformer_bf,
        (src_bf, tgt_bf),
        kwargs={
            'tgt_mask': tgt_mask,
            'src_key_padding_mask': src_key_padding_mask,
            'tgt_key_padding_mask': tgt_key_padding_mask
        }
    )

    print("All Transformer tests passed! ✅")


if __name__ == '__main__':
    test_transformer_models()