import h5py
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from tnlearn.seeds import random_seed
from tnlearn import MLPRegressor

random_seed(100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

group = 'no1'

with h5py.File('./data/epsilon_low.h5', 'r') as f:
    X_all = np.array(f[group]['x_train'])
    y_all = np.array(f[group]['y_train'])

scaler = MinMaxScaler()
X_all = scaler.fit_transform(X_all)
y_all = scaler.fit_transform(y_all)

batch_size = 10


X, X_test, y, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=10)

layers_list = [10, 10, 10]

clf = MLPRegressor(
    # neurons=neuron.neuron,
    neurons='6@x**2 + 2@x - 3',
    layers_list=layers_list,
    activation_funcs='sigmoid',
    loss_function='mse',
    random_state=50,
    optimizer_name='rmsprop',
    max_iter=50,
    batch_size=10,
    lr=0.01,
    visual=False,
    fig_path=None,
    visual_interval=10,
    save=True,
    interval=10,
    gpu=None,
    # scheduler={'step_size': 30,
    #            'gamma': 0.2},
    l1_reg=False,
    l2_reg=False,
)

clf.fit(X, y)
clf.count_param()
print(clf.predict(X_test))

clf.score(X_test, y_test)

clf.save(path='my_model_dir', filename='mlp_regressor.pth')

clf.load(path='my_model_dir', filename='mlp_regressor.pth',
         input_dim=X.shape[1], output_dim=1)

clf.fit(X, y)
