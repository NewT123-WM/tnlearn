import h5py
import numpy as np
from tnlearn import VecSymRegressor
from tnlearn import MLPRegressor
from sklearn.model_selection import train_test_split


with h5py.File('data/data1.h5', 'r') as f:
    X = np.array(f['X'])
    y = np.array(f['y'])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.1)


# neuron = VecSymRegressor(random_state=100,
#                          pop_size=500,
#                          max_generations=20,
#                          tournament_size=10,
#                          x_pct=0.7,
#                          xover_pct=0.3,
#                          operations=None)
#
# neuron.fit(X, y)


layers_list = [10, 10, 10]

clf = MLPRegressor(
    # neurons=neuron.neuron,
    neurons='x',
    layers_list=layers_list,
    activation_funcs='relu',
    loss_function='mse',
    random_state=1,
    optimizer_name='adam',
    max_iter=1000,
    batch_size=128,
    valid_size=0.2,
    lr=0.01,
    # visual=True,
    visual_interval=100,
    save=False,
    interval=100,
    gpu=None,
    # scheduler={'step_size': 30,
    #            'gamma': 0.2},
    l1_reg=False,
    l2_reg=False,
)

clf.fit(X_train, y_train)

# clf.score(X_train, y_train)
