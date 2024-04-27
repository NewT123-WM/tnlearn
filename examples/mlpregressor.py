from tnlearn import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=200, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

layers_list = [10, 10, 10]
clf = MLPRegressor(
    neurons='2@x**2',
    layers_list=None,
    activation_funcs='sigmoid',
    loss_function='mse',
    random_state=1,
    optimizer_name='adam',
    max_iter=200,
    batch_size=128,
    valid_size=0.2,
    lr=0.01,
    visual=True,
    visual_interval=100,
    save=False,
    interval=50,
    gpu=None,
    # scheduler={'step_size': 30,
    #            'gamma': 0.2},
    l1_reg=False,
    l2_reg=False,
)

clf.fit(X_train, y_train)
# a = clf.predict(X_test)
# clf.score(X_test, y_test)

# clf.save(path='my_model_dir', filename='mlp_regressor.pth')
# clf.load(path='my_model_dir', filename='mlp_regressor.pth', input_dim=20, output_dim=1)
# clf.fit(X_train, y_train)
