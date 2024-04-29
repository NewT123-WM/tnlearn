import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
import numpy as np
from tnlearn.utils import MyData
from torch import nn
from tnlearn.seeds import random_seed
from torch.utils.data import DataLoader


random_seed(100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = MinMaxScaler(feature_range=(-1, 1))

max_epoch = 1000
test_size = 0.2
batch_size = 8

sr = '-0.074@x**4 + 0.068@x**3 + 0.07@x**2 + 0.001@x'

ori_data = datasets.fetch_openml('vehicle_reproduced', version=1)  # (846, 18, 4)
ori_data.target = pd.factorize(ori_data.target)[0].astype(float)
X_all = np.array(ori_data.data, dtype=float)
X_all = scaler.fit_transform(X_all)
X_all = torch.from_numpy(X_all).to(torch.float32)
y_all = torch.from_numpy(ori_data.target).long()

X, X_test, y, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=10)

trainset = MyData(X, y)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

net = nn.Sequential(nn.Linear(18, 10),
                    nn.ReLU(),

                    nn.Linear(10, 10),
                    nn.ReLU(),

                    nn.Linear(10, 4)
                    ).to(device)

cost = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

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

    train_accuracy_net1 = 0
    with torch.no_grad():
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            predict = net(inputs)
            _, predict_class = torch.max(predict, 1)
            train_accuracy_net1 += (predict_class == labels).sum().item()
    train_accuracy_net1 = train_accuracy_net1 / X.shape[0]

    if (k + 1) % 100 == 0:
        print(f'Epoch:{k + 1}, Loss:{running_loss:.4f}, Accuracy:{train_accuracy_net1:.4f}')
