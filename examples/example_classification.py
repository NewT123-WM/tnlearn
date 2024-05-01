import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
import numpy as np
from tnlearn import MLPClassifier
from tnlearn.seeds import random_seed

random_seed(100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = MinMaxScaler(feature_range=(-1, 1))

sr = '-0.074@x**4 + 0.068235@x**3 + 0.07168875@x**2 + 0.0015433@x'

ori_data = datasets.fetch_openml('vehicle_reproduced', version=1)  # (846, 18, 4)
ori_data.target = pd.factorize(ori_data.target)[0].astype(float)
X_all = np.array(ori_data.data, dtype=float)
X_all = scaler.fit_transform(X_all)
y_all = ori_data.target

X, X_test, y, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=10)

layers_list = [10, 10]
clf = MLPClassifier(
    layers_list=layers_list,
    neurons=sr,
    activation_funcs='sigmoid',
    loss_function='crossentropy',
    random_state=100,
    optimizer_name='adam',
    max_iter=500,
    batch_size=8,
    lr=0.01,
    visual=True,
    visual_interval=10,
    save=True,
    fig_path='./',
    gpu=None,
    interval=10,
    # scheduler={'step_size': 30,
    #            'gamma': 0.2},
    l1_reg=False,
    l2_reg=False,
)

clf.fit(X, y)

# print(clf.predict(X_test))
#
# clf.score(X_test, y_test)
#
# clf.count_param()
#
# clf.save(path='my_model_dir', filename='mlp_classifier.pth')

# clf.load(path='my_model_dir', filename='mlp_classifier.pth',
#          input_dim=X.shape[1], output_dim=1,
#          )
#
# clf.fit(X, y)
