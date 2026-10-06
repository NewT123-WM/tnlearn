"""Initialization must not depend on Python's hash randomization."""

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


PROBE = """
import importlib
import json
import sys

import torch

layer_class = importlib.import_module(sys.argv[1]).BaseCustomNeuronLayer
expression = '<w1,x> + <w2,x**2> + <w10,x**3> + <x,x> + <x,x**2> + 1'
torch.set_num_threads(1)
x = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 12
snapshots = []
for seed in (42, 43):
    torch.manual_seed(seed)
    layer = layer_class(3, 2, expression, already_parametrized=False)
    parameters = {}
    names = {}
    for prefix in ('w', 'c', 'b'):
        names[prefix] = getattr(layer, prefix + '_names')
        weights = getattr(layer, prefix + '_weights')
        parameters.update({name: value.detach().tolist()
                           for name, value in zip(names[prefix], weights)})
    parameters['bias'] = layer.bias.detach().tolist()
    snapshots.append({'names': names, 'parameters': parameters,
                      'output': layer(x).detach().tolist()})
print(json.dumps(snapshots))
"""


class TestMLPInitialization(unittest.TestCase):
    def assert_hash_independent(self, module_name):
        snapshots = []
        for hash_seed in ("0", "5217"):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed, OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       NUMEXPR_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
            result = subprocess.run(
                [sys.executable, "-c", PROBE, module_name],
                cwd=Path(__file__).resolve().parents[1],
                env=env, check=True, capture_output=True, text=True, timeout=60,
            )
            snapshots.append(json.loads(result.stdout))

        # Both the symbol-to-value mapping and the initial function must agree.
        self.assertEqual(snapshots[0], snapshots[1])
        # A different model seed must still produce a different initialization.
        self.assertNotEqual(snapshots[0][0]["parameters"],
                            snapshots[0][1]["parameters"])

    def test_classifier(self):
        self.assert_hash_independent("tnlearn.mlpclassifier")

    def test_regressor(self):
        self.assert_hash_independent("tnlearn.mlpregressor")


if __name__ == "__main__":
    unittest.main()
