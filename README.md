<div align="center">
  <img src="assets/logo.png" width="100%" />
</div>


 Tnlearn is an open source python library. It is based on the symbolic regression algorithm to generate task-based neurons, and then utilizes diverse neurons to build neural networks.

![Static Badge](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg) ![Static Badge](https://img.shields.io/badge/License-Apache--2.0-blue.svg) ![Static Badge](https://img.shields.io/badge/pypi-v0.1.0-orange?logo=PyPI) ![GitHub Repo stars](https://img.shields.io/github/stars/NewT123-WM/tnlearn?style=flat&logo=github)  



# Quick links

* [Motivation](#Motivation)

* [Features](#features)

* [Overview](#Overview)

* [Benchmarks](#Benchmarks)

* [Resource](#Resource)

* [Dependences](#Dependences)

* [Install](#install)

* [Quick start](#Quick-start)

* [API documentation](#API-documentation)

* [Citation](#citation)

* [The Team](#The-Team)

* [License](#License)

  

# Motivation

* **NuronAI inspired** In the past decade, successful networks have primarily used a single type of neurons within novel architectures, yet recent deep learning studies have been inspired by the diversity of human brain neurons, leading to the proposal of new artificial neuron designs.

* **Task-Based Neuron Design**  Given the human brain's reliance on task-based neurons, can artificial network design shift from focusing on task-based architecture to task-based neuron design?

* **Enhanced Representation** Since there are no universally applicable neurons, task-based neurons could enhance feature representation ability within the same structure, due to the intrinsic inductive bias for the task.



# Features

* Vectorized symbolic regression is employed to find optimal formulas that fit input data.

* We parameterize the obtained elementary formula to create learnable parameters, serving as the neuron's aggregation function.



# Overview



A nice picture describing the structure of tnlearn will be produced here.











# Benchmarks

We select several advanced machine learning methods for comparison.

|     Method      |                            Venues                            |                          Code link                           |
| :-------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|     XGBoost     | [ACM SIGKDD 2016](https://dl.acm.org/doi/abs/10.1145/2939672.2939785) |    [Adopt official code](https://github.com/dmlc/xgboost)    |
|    LightGBM     | [NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf) | Implemented by [widedeep](https://github.com/jrzaurin/pytorch-widedeep) |
|    CatBoost     | [Journal of big data](https://link.springer.com/article/10.1186/s40537-020-00369-8) | [Adopt official code](https://github.com/catboost/catboost)  |
|     TabNet      | [AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/16826) | Implemented by [widedeep](https://github.com/jrzaurin/pytorch-widedeep) |
| Tab Transformer |          [arxiv](https://arxiv.org/abs/2012.06678)           | [Adopt official code](https://github.com/lucidrains/tab-transformer-pytorch) |
| FT-Transformer  | [NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html) | Implemented by [widedeep](https://github.com/jrzaurin/pytorch-widedeep) |
|     DANETs      | [AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/20309) |  [Adopt official code](https://github.com/whatashot/danet)   |



We test multiple advanced machine learning methods on two sets of real-world data. The test results (MSE) are shown in the following table:

|       Method       | [Particle collision](https://www.kaggle.com/datasets/fedesoriano/cern-electron-collision-data) | [Asteroid prediction](https://www.kaggle.com/datasets/basu369victor/prediction-of-<br/>asteroid-diameter) |
| :----------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|      XGBoost       |                      $0.0094\pm0.0006$                       |                      $0.0646\pm0.1031$                       |
|      LightGBM      |                      $0.0056\pm0.0004$                       |                      $0.1391\pm0.1676$                       |
|      CatBoost      |                      $0.0028\pm0.0002$                       |                      $0.0817\pm0.0846$                       |
|       TabNet       |                      $0.0040\pm0.0006$                       |                      $0.0627\pm0.0939$                       |
|   TabTransformer   |                      $0.0038\pm0.0008$                       |                      $0.4219\pm0.2776$                       |
|   FT-Transformer   |                      $0.0050\pm0.0020$                       |                      $0.2136\pm0.2189$                       |
|       DANETs       |                      $0.0076\pm0.0009$                       |                      $0.1709\pm0.1859$                       |
| Task-based Network |                  $\mathbf{0.0016\pm0.0005}$                  |                  $\mathbf{0.0513\pm0.0551}$                  |



# Resource

Here is a resource summary for neuronal diversity in artificial networks.

|                           Resource                           |                             Type                             |                         Description                          |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|      [QuadraLib](https://github.com/zarekxu/QuadraLib)       | [Library](https://proceedings.mlsys.org/paper_files/paper/2022/hash/6270a15843a2e06a95d3e3ad8b489e4b-Abstract.html) | The QuadraLib is a library for the efficient optimization and design exploration of quadratic networks.The paper of QuadraLib won MLSys 2022’s best paper award. |
| [Dr. Fenglei Fan’s GitHub Page](https://github.com/FengleiFan) |                             Code                             | Dr. Fenglei Fan’s GitHub Page summarizes a series of papers and associated code on quadratic networks, including quadratic autoencoder and the training algorithm ReLinear. |
| [Polynomial Network](https://github.com/grigorisg9gr/polynomial_nets) |                             Code                             | This repertoire shows how to build a deep polynomial network and sparsify it with tensor decomposition. |
|     [Dendrite](http://www.dendrites.org/dendrites-book)      |                             Book                             | A comprehensive book covering all aspects of dendritic computation. |




# Dependences

You should ensure that the version of pytorch corresponds to the version of cuda so that gpu acceleration can be guaranteed. Here is a reference version

`Pytorch 2.1.0`

`cuda 12.1`

Other major dependencies are automatically installed when installing tnlearn.



# Install

Tnlearn and its dependencies can be easily installed with pip:

```shell
pip install tnlearn
```

Tnlearn and its dependencies can be easily installed with conda:

```shell
conda install -c tnlearn
```



# Quick start

This is a quick example to show you how to use tnlearn in regression tasks. Note that your data types should be tabular data.

```python
from tnlearn import VecSymRegressor
from tnlearn import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate data.
X, y = make_regression(n_samples=200, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# A vectorized symbolic regression algorithm is used to generate task-based neurons.
neuron = VecSymRegressor()
neuron.fit(X_train, y_train)

# Build neural network using task-based neurons and train it.
clf = MLPRegressor(neurons=neuron.neuron，
                   layers_list=[50,30,10]) #Specify the structure of the hidden layers in the MLP.
clf.fit(X_train, y_train)

# Predict
clf.predict(X_test)
```



There are many hyperparameters in tnlearn that can be debugged, making the neural network performance more superior. Please see the  [API documentation](#API documentation) for specific usage.



# API documentation

Here's our official API documentation, available on [Read the Docs](https://tnlearn-doc.readthedocs.io/en/latest/index.html).



# Citation

If you find Tnlearn useful, please cite it in your publications.

```bibtex
@article{


}
```





# The Team

Tnlearn is a work by [Meng Wang](https://github.com/NewT123-WM), [Juntong Fan](https://github.com/Juntongkuki), and [Fenglei Fan](https://github.com/FengleiFan).



# License

Tnlearn is released under Apache License 2.0.

