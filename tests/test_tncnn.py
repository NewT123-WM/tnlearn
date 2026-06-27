import torch
import os
import tempfile
from tnlearn import TNLinear
from tnlearn import (
    TNConv1d, TNConv2d, TNConv3d,
    TNConvTranspose1d, TNConvTranspose2d, TNConvTranspose3d
)

# ---------- Test functions ----------
def test_save_load(model, input_tensor):
    """Perform one training step on the model, save and reload it, then verify outputs and parameters."""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    out = model(input_tensor)
    loss = out.sum()          # Simple loss to ensure backward pass
    loss.backward()
    optimizer.step()

    # Record the output after update
    model.eval()
    with torch.no_grad():
        out_after_train = model(input_tensor)

    # Save the full model (including architecture)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        torch.save(model, f.name)
        path = f.name

    # Load the model
    try:
        loaded_model = torch.load(path, weights_only=False)
    except TypeError:
        loaded_model = torch.load(path) 
    loaded_model.eval()
    with torch.no_grad():
        out_loaded = loaded_model(input_tensor)

    # Compare outputs
    assert torch.allclose(out_after_train, out_loaded, atol=1e-6), \
        f"Output mismatch for {type(model).__name__}"

    # Compare parameters
    for (name1, p1), (name2, p2) in zip(model.state_dict().items(),
                                        loaded_model.state_dict().items()):
        assert torch.allclose(p1, p2, atol=1e-6), \
            f"Parameter mismatch for {name1}"

    print(f"✅ Save/load test passed for {type(model).__name__}")
    os.unlink(path)


# ---------- Original test code (extended with lists) ----------
models = []
inputs = []

# 1. TNLinear
linear = TNLinear(10, 5, symbolic_expression='x**2+3@torch.sin(x)', bias=True)
x_lin = torch.randn(3, 10)
y_lin = linear(x_lin)
print(f"Linear output shape: {y_lin.shape}")
models.append(linear)
inputs.append(x_lin)

# 2. TNConv1d
conv1d = TNConv1d(
    in_channels=3,
    out_channels=64,
    kernel_size=5,
    stride=2,
    padding=1,
    symbolic_expression='x + torch.sin(x)',
    padding_mode='reflect'
)
x_c1 = torch.randn(2, 3, 100)
y_c1 = conv1d(x_c1)
print(f"Conv1d output shape: {y_c1.shape}")
models.append(conv1d)
inputs.append(x_c1)

# 3. TNConv2d
conv2d = TNConv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    symbolic_expression='x + 0.5@x**2',
    groups=1,
    dilation=2,
    padding_mode='zeros'
)
x_c2 = torch.randn(1, 3, 32, 32)
y_c2 = conv2d(x_c2)
print(f"Conv2d output shape: {y_c2.shape}")
models.append(conv2d)
inputs.append(x_c2)

# 4. TNConv3d
conv3d = TNConv3d(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    symbolic_expression='x'
)
x_c3 = torch.randn(1, 3, 16, 32, 32)
y_c3 = conv3d(x_c3)
print(f"Conv3d output shape: {y_c3.shape}")
models.append(conv3d)
inputs.append(x_c3)

# 5. TNConvTranspose1d
conv_transpose1d = TNConvTranspose1d(
    in_channels=16,
    out_channels=32,
    kernel_size=4,
    stride=2,
    padding=1,
    output_padding=0,
    symbolic_expression='x**5'
)
x_ct1 = torch.randn(2, 16, 100)
y_ct1 = conv_transpose1d(x_ct1)
print(f"ConvTranspose1d output shape: {y_ct1.shape}")
models.append(conv_transpose1d)
inputs.append(x_ct1)

# 6. TNConvTranspose2d
conv_transpose2d = TNConvTranspose2d(
    in_channels=3,
    out_channels=64,
    kernel_size=4,
    stride=2,
    padding=1,
    symbolic_expression='x + torch.sin(x)'
)
x_ct2 = torch.randn(1, 3, 32, 32)
y_ct2 = conv_transpose2d(x_ct2)
print(f"ConvTranspose2d output shape: {y_ct2.shape}")
models.append(conv_transpose2d)
inputs.append(x_ct2)

# 7. TNConvTranspose3d
conv_transpose3d = TNConvTranspose3d(
    in_channels=8,
    out_channels=16,
    kernel_size=3,
    stride=2,
    padding=1,
    symbolic_expression='x + 0.5@x**2'
)
x_ct3 = torch.randn(1, 8, 16, 32, 32)
y_ct3 = conv_transpose3d(x_ct3)
print(f"ConvTranspose3d output shape: {y_ct3.shape}")
models.append(conv_transpose3d)
inputs.append(x_ct3)

# ---------- Run save/load tests ----------
print("\n--- Running save/load tests ---")
for model, inp in zip(models, inputs):
    test_save_load(model, inp)
print("All tests passed! ✅")
