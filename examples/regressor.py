import h5py
from tnlearn import GPSymRegressor
import numpy as np

neuron = GPSymRegressor(random_state=100,
                         pop_size=50,
                         max_generations=3,
                         tournament_size=5,
                         coefficient_range=[-1, 1],
                         x_pct=0.7,
                         xover_pct=0.3,
                         save=False,
                         operations=None)

with h5py.File('data/data1.h5', 'r') as f:
    X = np.array(f['X'])
    y = np.array(f['y'])

neuron.fit(X, y)
print(neuron.neuron)
