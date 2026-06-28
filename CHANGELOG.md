# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — 0.2.0.dev0

### Added
- New `tnlearn.modules` subpackage with task‑based neural modules:
  - `TNLinear`, `TNConv1d`/`2d`/`3d`, `TNConvTranspose1d`/`2d`/`3d`
  - `TNRNN`, `TNLSTM`, `TNGRU` and their cell variants
  - `TNTransformer`, `TNTransformerEncoder`/`Decoder` and layers
- `RLRegressor` for reinforcement‑learning‑based symbolic regression (policy gradient)
- `LLMSymRegressor`: symbolic regression using LLM + BFGS optimization with support for multiple LLM providers (DeepSeek, SiliconFlow, Ollama, BLT, CSTCloud)
- Enhanced `LLMSymRegressor` and `RLRegressor` with improved `CustomNeuronLayer` compatibility
- GPU acceleration support 
- LLM-based Symbolic Regression (DrSR) module with automatic data insight extraction and experience replay
- Benchmark tool for performance evaluation

### Fixed
- Fixed random seed type validation
- Corrected environment dependencies

### Changed
- Replaced `setup.py` with modern `pyproject.toml` packaging
- Updated README documentation link

### Performance
- Added GPU acceleration support for core operations

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