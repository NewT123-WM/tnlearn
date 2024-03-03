import h5py
from tnlearn import Regressor
import numpy as np

neuron = Regressor(random_state=100,
                   pop_size=500,
                   max_generations=20,
                   tournament_size=10,
                   x_pct=0.7,
                   xover_pct=0.3,
                   save=True,
                   operations=None)

with h5py.File('epsilon_high.h5', 'r') as f:
    X_train = np.array(f['no1']['x_train'])
    y_label = np.array(f['no1']['y_train'])

neuron.fit(X_train, y_label)
print('*' * 20)
print(neuron.neuron)
