import h5py
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from tnlearn.utils import MyData
from torch import nn
from tnlearn.seeds import random_seed
from torch.utils.data import DataLoader

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
max_epoch = 1000
sr = '6@x**2 + 2@x - 3'

X_all = torch.from_numpy(X_all).to(torch.float32)
y_all = torch.from_numpy(y_all).to(torch.float32)
X, X_test, y, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=10)
trainset = MyData(X, y)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

net = nn.Sequential(nn.Linear(10, 10),
                    nn.ReLU(),

                    nn.Linear(10, 10),
                    nn.ReLU(),

                    nn.Linear(10, 10),
                    nn.ReLU(),

                    nn.Linear(10, 1),
                    ).to(device)

cost = nn.MSELoss().to(device)
optimizer = torch.optim.RMSprop(net.parameters(), lr=0.01)

for k in range(max_epoch):
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        predict = net(inputs)
        loss = cost(predict, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if (k + 1) % 100 == 0:
        print(f'Epoch:{k + 1}, Loss:{running_loss:.4f}')
