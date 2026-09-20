# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 0.2.0.dev0

### Added
- Added the `tnlearn.modules` subpackage for task-based PyTorch modules:
  - `TNLinear`
  - `TNConv1d`, `TNConv2d`, `TNConv3d`
  - `TNConvTranspose1d`, `TNConvTranspose2d`, `TNConvTranspose3d`
  - `TNRNN`, `TNLSTM`, `TNGRU` and cell variants
  - `TNTransformer`, encoder/decoder stacks, and transformer layers
- Added `InnerProduct` operator utilities, including parameterization,
  evaluation, pretty-print conversion, and expression conversion helpers.
- Added `base` and `legacy` modes across symbolic regressors and task-based
  neural modules, enabling newer inner-product expressions while preserving
  compatibility with legacy vectorized expressions.
- Added `GPSymRegressor` as the newer GP symbolic regressor interface;
  `VecSymRegressor` remains available for legacy compatibility.
- Added `parent_selection` support to `GPSymRegressor`.
- Added scikit-learn compatible `get_params` and `set_params` support for MLP
  estimators.
- Added `RLRegressor` and `RLSymRegressor` for reinforcement-learning-based
  symbolic regression.
- Added `LLMSymRegressor` and DrSR for LLM-assisted symbolic regression with
  BFGS optimization and multiple provider backends, including DeepSeek,
  SiliconFlow, Ollama, BLT, and CSTCloud.
- Added `already_parametrized` controls to `BaseCustomNeuronLayer`,
  `MLPRegressor`, `MLPClassifier`, `TNLinear`, convolution modules, recurrent
  modules, and transformer modules.
- Added NeuronSeek structure search with rank-preserving inner-product export.
- Added runnable NeuronSeek search examples and manual configuration examples.
- Added GPU support work for examples and core workflows.
- Added benchmark tooling for performance evaluation.
- Added `IMPROVEMENT.md` documenting extensions beyond the TPAMI task-based
  neuron paper for the JMLR MLOSS submission.

### Changed
- Replaced `setup.py` with modern `pyproject.toml` packaging.
- Updated exported package APIs and README examples to expose the new task-based
  modules.
- Updated DrSR components for mode switching and automatic conversion of
  non-inner-product expressions in base mode.
- Removed implicit weight decay from NeuronSeek so weight regularization can be
  controlled explicitly.
- Updated RLRegressor example variable naming.
- Removed obsolete GPU-specific requirements/examples after consolidating the
  packaging and examples.
- Updated changelog tooling and git-cliff configuration.
- Revised citation details for the task-based neurons article.

### Fixed
- Fixed random seed type validation.
- Fixed multi-output target handling in `LLMSymRegressor` and `RLRegressor`.
- Fixed vectorized formula generation in `LLMSymRegressor` and `RLRegressor` so
  generated formulas match expected output dimensions.
- Fixed dependency declarations and environment configuration.
- Preserved NeuronSeek CP rank during export.
- Handled constant inner-product expressions in NeuronSeek export paths.

### Tests and CI
- Added unit tests for `TNLinear`, convolution modules, recurrent modules, and
  transformer modules.
- Added NeuronSeek regression tests in CI.
- Added tests for NeuronSeek export and MLP integration.
- Added tests for explicit weight-regularization control.
- Added tests for `already_parametrized` behavior.

## [0.1.1] - 2025-06-05

### Added
- Core symbolic regression functionality
- Basic neural module infrastructure
- Vectorized symbolic regression (`VecSymRegressor`) for task‑based neuron generation
- MLP regressor with task‑based neuron support
- Poly tensor regressor
- Initial benchmarking suite

### Changed
- Refined code comments across the codebase
- Updated documentation and README
- Corrected dependencies

## [0.1.0] - 2025-06-05

### Added
- Initial pre‑release of tnlearn
- Foundational project structure and core dependencies
- Basic neural network building blocks
- Initial README and licensing
