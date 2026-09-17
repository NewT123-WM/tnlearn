# Improvements and Extensions of TNLearn Relative to the TPAMI Paper

This document is provided to satisfy the JMLR MLOSS prior-publication requirement. It describes the relationship between TNLearn and our earlier TPAMI paper, and lists the improvements and extensions of the software package.

## Prior Work
- **Paper**: "No One-Size-Fits-All Neurons: Task-based Neurons for Artificial Neural Networks"
- **Published in**: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2026
- **Link**: https://ieeexplore.ieee.org/abstract/document/11568692
- **Official reference implementation**: https://github.com/NewT123-WM/Task_based_neurons

## Relationship
TNLearn and Task_based_neurons are separate repositories that share the same core idea of task-based neurons. Task_based_neurons is the official code release accompanying the TPAMI paper and is intended mainly for reproducing the paper's experiments. TNLearn is an independently developed, user-oriented software package that builds on this idea and provides substantial improvements and extensions for practical use.

TNLearn is not a repackaging of the TPAMI reference implementation. It provides a new modular architecture, additional symbolic regression methods, broader network-layer support, improved documentation, packaging for public distribution, and active maintenance.

## Improvements and Extensions in TNLearn
- **Packaging and distribution**: pip-installable package on PyPI (`pip install tnlearn`), with versioned releases.
- **Modular architecture**: refactored codebase with clear separation between neuron discovery, neuron construction, network construction, and utility modules.
- **Four symbolic regression engines**:
  - `GPSymRegressor`: genetic programming for vectorized symbolic regression.
  - `LLMSymRegressor`: LLM-guided symbolic regression.
  - `RLSymRegressor`: reinforcement-learning-based symbolic term selection.
  - `PolyTensorRegressor`: differentiable tensor-decomposed polynomial regression.
- **Task-based network construction**: `MLPRegressor` and `MLPClassifier` for regression and classification, compatible with the scikit-learn estimator API.
- **Task-based layers**: modular layers for fully-connected, convolutional, recurrent, and Transformer architectures, e.g., `TNLinear`, `TNConv2d`.
- **Integration**: scikit-learn-compatible APIs and PyTorch integration.
- **Documentation and examples**: comprehensive documentation, tutorials, and representative examples.
- **Software engineering**: unit and integration tests, continuous integration, and active maintenance.
- **Community**: PyTorch ecosystem project, public issue tracker, and open contribution process.

## Summary
TNLearn extends the methodological ideas of the TPAMI paper into a mature, user-friendly, and actively maintained open-source software package. The TPAMI paper introduced the concept and methodology of task-based neurons; TNLearn focuses on usability, extensibility, reproducibility, and community-driven development.