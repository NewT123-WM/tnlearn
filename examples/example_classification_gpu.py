import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tnlearn.seeds import random_seed
from tnlearn_gpu import MLPClassifier_gpu

random_seed(100)
scaler = MinMaxScaler(feature_range=(-1, 1))

sr = '-0.074@x**4 + 0.068235@x**3 + 0.07168875@x**2 + 0.0015433@x'

ori_data = datasets.fetch_openml('vehicle_reproduced', version=1)
ori_data.target = pd.factorize(ori_data.target)[0].astype(float)
X_all = np.array(ori_data.data, dtype=float)
X_all = scaler.fit_transform(X_all)
y_all = ori_data.target

X, X_test, y, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=10)

clf = MLPClassifier_gpu(
    layers_list=[10, 10],
    neurons=sr,
    activation_funcs='sigmoid',
    loss_function='crossentropy',
    random_state=100,
    optimizer_name='adam',
    max_iter=200,
    batch_size=8,
    lr=0.01,
    visual=False,
    save=False,
    gpu=0,
    interval=20,
)

clf.fit(X, y)
print('GPU classification example finished.')
