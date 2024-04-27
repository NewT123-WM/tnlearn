from tnlearn import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=200, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

layers_list = [50, 80, 10, 10]
clf = MLPClassifier(
    layers_list=layers_list,
    neurons='2@x**2',
    # neurons='x',
    activation_funcs='sigmoid',
    loss_function=None,
    random_state=1,
    optimizer_name='sgd',
    max_iter=2000,
    batch_size=128,
    valid_size=0.2,
    lr=0.01,
    visual=True,
    visual_interval=100,
    save=False,
    gpu=None,
    interval=None,
    # scheduler={'step_size': 30,
    #            'gamma': 0.2},
    l1_reg=None,
    l2_reg=None,
)

clf.fit(X_train, y_train)
# a = clf.predict(X_test)
clf.score(X_test, y_test)
# #
# clf.save(path='my_model_dir', filename='mlp_classifier.pth')
# clf.load(path='my_model_dir', filename='mlp_classifier.pth',
#          input_dim=X_train.shape[1], output_dim=1,
#          )
# clf.fit(X_train, y_train)
