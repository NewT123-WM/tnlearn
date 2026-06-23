import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tnlearn.seeds import random_seed
from tnlearn_gpu import MLPRegressor_gpu

random_seed(100)

with h5py.File('./data/epsilon_low.h5', 'r') as f:
    X_all = np.array(f['no1']['x_train'])
    y_all = np.array(f['no1']['y_train'])

scaler = MinMaxScaler()
X_all = scaler.fit_transform(X_all)
y_all = scaler.fit_transform(y_all)

X, X_test, y, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=10)

reg = MLPRegressor_gpu(
    neurons='6@x**2 + 2@x - 3',
    layers_list=[10, 10, 10],
    activation_funcs='sigmoid',
    loss_function='mse',
    random_state=50,
    optimizer_name='rmsprop',
    max_iter=80,
    batch_size=10,
    lr=0.01,
    visual=False,
    save=False,
    gpu=0,
    interval=20,
)

reg.fit(X, y)
print('GPU regression example finished.')
