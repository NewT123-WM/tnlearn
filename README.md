
<div align="center">
  <img src="https://raw.githubusercontent.com/NewT123-WM/tnlearn/main/assets/logo.png" width="100%" />
</div>

Tnlearn is an open source python library. It is based on the symbolic regression algorithm to generate task-based neurons, and then utilizes diverse neurons to build neural networks.

![Static Badge](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg) ![Static Badge](https://img.shields.io/badge/License-Apache--2.0-blue.svg) [![PyPI](https://img.shields.io/pypi/v/tnlearn.svg?logo=pypi)](https://pypi.org/project/tnlearn/) ![GitHub Repo stars](https://img.shields.io/github/stars/NewT123-WM/tnlearn?style=flat&logo=github)  

# Quick links

- [Quick links](#quick-links)
- [Motivation](#motivation)
- [Framework](#framework)
- [Features](#features)
- [Dependencies](#dependencies)
- [Install](#install)
- [Quick start](#quick-start)
  - [GPSymRegressor](#gpsymregressor)
  - [PolyTensorRegressor](#polytensorregressor)
  - [RLSymRegressor](#rlsymregressor)
  - [LLMSymRegressor](#llmsymregressor)
  - [Supported LLM Providers](#supported-llm-providers)
- [API documentation](#api-documentation)
- [Benchmarks](#benchmarks)
- [Resource](#resource)
- [Citation](#citation)
- [The Team](#the-team)
- [License](#license)

# Motivation

* **NuronAI inspired** In the past decade, successful networks have primarily used a single type of neurons within novel architectures, yet recent deep learning studies have been inspired by the diversity of human brain neurons, leading to the proposal of new artificial neuron designs.

* **Task-Based Neuron Design**  Given the human brain's reliance on task-based neurons, can artificial network design shift from focusing on task-based architecture to task-based neuron design?

* **Enhanced Representation** Since there are no universally applicable neurons, task-based neurons could enhance feature representation ability within the same structure, due to the intrinsic inductive bias for the task.

# Framework

<div align="center">
  <img src="https://raw.githubusercontent.com/NewT123-WM/tnlearn/main/assets/framework.drawio.svg" alt="Tnlearn framework" width="100%" />
</div>

# Features

* Vectorized symbolic regression is employed to find optimal formulas that fit input data.

* We parameterize the obtained elementary formula to create learnable parameters, serving as the neuron's aggregation function.

# Dependencies

Tnlearn declares `torch>=1.12.0` and installs required Python dependencies
automatically. For GPU usage, install a PyTorch build that matches your hardware
from the [official PyTorch selector](https://pytorch.org/get-started/locally/)
before installing tnlearn.

# Install

From PyPI:

```shell
pip install tnlearn
```

From source:

```shell
git clone https://github.com/NewT123-WM/tnlearn.git
cd tnlearn
pip install -e .
```

If PyTorch is already installed with the correct CPU/GPU build,
`pip install -e .` will use it as long as it satisfies `torch>=1.12.0`.

# Quick start

Choose one symbolic regressor, search a task-based neuron expression, then pass
that expression to `MLPRegressor`. For local experiments, start with
`GPSymRegressor` or `PolyTensorRegressor`; use `LLMSymRegressor` when an LLM API
key is available.

## GPSymRegressor

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from tnlearn import GPSymRegressor, MLPRegressor

X, y = make_regression(n_samples=80, n_features=4, random_state=1)
X_train, X_test, y_train, _ = train_test_split(X, y, random_state=1)

search = GPSymRegressor(
    mode='legacy',
    pop_size=40,
    max_generations=2,
    tournament_size=3,
)
search.fit(X_train, y_train)

model = MLPRegressor(search.neuron, layers_list=[8], max_iter=20, mode='legacy')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

`GPSymRegressor(mode='legacy')` exports the legacy `@` expression format, so
the MLP also uses `mode='legacy'`.

`VecSymRegressor` is the historical class name for this legacy GP path. In
other symbolic regressors, `mode='legacy'` has the same compatibility meaning:
it selects the older simplified vectorized expression format without
inner-product interaction terms. The default/base modes below export
inner-product expressions for the current MLP API.

## PolyTensorRegressor

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from tnlearn import MLPRegressor, PolyTensorRegressor

X, y = make_regression(n_samples=80, n_features=6, random_state=1)
X_train, X_test, y_train, _ = train_test_split(X, y, random_state=1)

search = PolyTensorRegressor(rank=2, poly_order=2, num_epochs=10, random_state=1)
search.fit(X_train, y_train)

model = MLPRegressor(search.neuron, layers_list=[8], max_iter=20)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## RLSymRegressor

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from tnlearn import MLPRegressor, RLSymRegressor

X, y = make_regression(n_samples=80, n_features=4, random_state=1)
X_train, X_test, y_train, _ = train_test_split(X, y, random_state=1)

search = RLSymRegressor(
    max_episodes=10,
    max_terms_total=3,
    random_state=1,
    verbose=False,
)
search.fit(X_train, y_train)

model = MLPRegressor(search.neuron, layers_list=[8], max_iter=20)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## LLMSymRegressor

Set `DEEPSEEK_API_KEY` before running this example.

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from tnlearn import LLMSymRegressor, MLPRegressor

X, y = make_regression(n_samples=80, n_features=4, random_state=1)
X_train, X_test, y_train, _ = train_test_split(X, y, random_state=1)

search = LLMSymRegressor(
    llm_config={'model': 'deepseek/deepseek-chat'},
    max_iterations=1,
    samples_per_iteration=1,
    verbose=0,
    mode='base',
)
search.fit(X_train, y_train)

model = MLPRegressor(search.neuron, layers_list=[8], max_iter=20)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

`PolyTensorRegressor`, `RLSymRegressor`, and `LLMSymRegressor` export
inner-product expressions such as `<w1, x**2> + <w2, x>*<w3, x>`, which the
default MLP mode understands directly.

## Supported LLM Providers

| Provider | Environment Variable | Example `model` |
|----------|---------------------|-----------------|
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek/deepseek-chat` |
| SiliconFlow | `SILICONFLOW_API_KEY` | `siliconflow/Qwen/Qwen3-8B` |
| Ollama (local) | – | `ollama/llama3.1:8b` |
| BLT | `BLT_API_KEY` | `blt/gpt-4` |
| CSTCloud | `CSTCLOUD_API_KEY` | `cstcloud/gpt-oss-120b` |

# API documentation

For complete module references, class parameters, and advanced usage:

<a href="https://tnlearn-documentation.readthedocs.io/en/latest/index.html">
  <img
    src="https://img.shields.io/badge/Open%20API%20Documentation-Read%20the%20Docs-blue?style=for-the-badge&logo=readthedocs"
    alt="Open API Documentation"
  />
</a>

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

# Citation

If you find Tnlearn useful, please cite it in your publications.

```bibtex
@article{fan2026no,
  title={No one-size-fits-all neurons: Task-based neurons for artificial neural networks},
  author={Fan, Feng-Lei and Wang, Meng and Dong, Hang-Cheng and Ma, Jianwei and Zeng, Tieyong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2026},
  publisher={IEEE}
}
```

# The Team

Tnlearn is a work by [Meng Wang](https://github.com/NewT123-WM), [Juntong Fan](https://github.com/Juntongkuki), [Hanyu Pei](https://github.com/HanyuPei22), [Tieyun LI](https://github.com/MillenRosen), [Jingxiao Liao](https://github.com/asdvfghg), [Shuren Qi](https://github.com/ShurenQi), [Lizhao Xu](https://github.com/xlzion), [Zeyu LI](https://github.com/zyli-math), [Renfeng Peng](https://github.com/JimmyPeng1998), [Yudong Wang](https://github.com/Nanzhilin), [Can Dong](https://github.com/CanD3333), [Tansheng Zhu](https://github.com/tshzhu), [Liangchen Tan](https://github.com/Liangchen-0311), [Feifei Zhang](https://github.com/Faye2020), [Yihan Jin](https://github.com/15700342), [Yiqing Zhang](https://github.com/ZnGY9), [Kairan Zhang](https://github.com/zakarRoman) and [Fenglei Fan](https://github.com/FengleiFan).

# License

Tnlearn is released under Apache License 2.0.
