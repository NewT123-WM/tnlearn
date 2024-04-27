import h5py
from tnlearn import VecSymRegressor
import numpy as np

neuron = VecSymRegressor(random_state=100,
                         pop_size=500,
                         max_generations=20,
                         tournament_size=10,
                         x_pct=0.7,
                         xover_pct=0.3,
                         operations=None)

with h5py.File('data/data1.h5', 'r') as f:
    X = np.array(f['X'])
    y = np.array(f['y'])

neuron.fit(X, y)
print('*' * 20)
print(neuron.neuron)
