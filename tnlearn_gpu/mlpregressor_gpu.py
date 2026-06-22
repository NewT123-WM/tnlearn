import torch
from tnlearn.mlpregressor import MLPRegressor


class MLPRegressor_gpu(MLPRegressor):
    """GPU-first wrapper, isolated from original source tree.

    - Keeps original behavior but defaults to `gpu=0`.
    - Requires CUDA availability.
    """

    def __init__(self, *args, gpu=0, **kwargs):
        if gpu is None:
            gpu = 0
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. MLPRegressor_gpu requires CUDA."
            )
        super().__init__(*args, gpu=gpu, **kwargs)
