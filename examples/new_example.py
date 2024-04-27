import h5py
import numpy as np
from tnlearn import VecSymRegressor
from tnlearn import MLPRegressor
from sklearn.model_selection import train_test_split
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tnlearn.utils import MyData
from sklearn.preprocessing import MinMaxScaler
import torch

with h5py.File('./data/data1.h5', 'r') as f:
    X = np.array(f['X'])
    y = np.array(f['y'])

scaler = MinMaxScaler()
group = 'no1'
with h5py.File('./data/epsilon_low.h5', 'r') as f:
    X_all = np.array(f[group]['x_train'])
    y_all = np.array(f[group]['y_train'])


X_train_, X_test, y_train_, y_test = train_test_split(X, y, random_state=1, test_size=0.1)

X_train, X_valid, y_train, y_valid = train_test_split(X_train_, y_train_, test_size=0.2,
                                                      random_state=100)

net = nn.Sequential(nn.Linear(X_train.shape[1], 10),
                    nn.ReLU(),

                    nn.Linear(10, 10),
                    nn.ReLU(),

                    nn.Linear(10, 10),
                    nn.ReLU(),

                    nn.Linear(10, 1)
                    )

learning_rate = 0.01
interval = 100

cost = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
batch_size = 128

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

trainset = MyData(X_train, y_train)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

for epoch in range(1000):
    running_loss = 0.0
    # Iterate over the training data.
    for inputs, targets in trainloader:
        # Move the inputs and targets to the same device as the model.
        inputs, targets = inputs, targets.reshape(-1, 1)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = cost(outputs, targets)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    running_loss /= X_train.shape[0]

    if (epoch + 1) % interval == 0:
        print(running_loss)
