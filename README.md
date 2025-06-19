![CI](https://github.com/RedactedCoder23/amt-hypervector/actions/workflows/ci.yml/badge.svg)
![PyPI](https://img.shields.io/pypi/v/amt-hypervector)
![License](https://img.shields.io/github/license/RedactedCoder23/amt-hypervector)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/RedactedCoder23/amt-hypervector/main)
# amt-hypervector

**Adaptive, Modular, Transparent Hypervector Reasoning (BHRE/AMT)**

- Deterministic 6-dimensional hypervector encoding (SHA256 → sinc)
- Dual-channel self-validation (ADF) for adaptive memory
- Self-play chess demo and simple Flask API

## Quick Start
```bash
git clone https://github.com/RedactedCoder23/amt-hypervector.git
cd amt-hypervector
pip install -e .[dev]
bhre --config config.yaml
```

## Examples
- [Encode Demo](docs/examples/encode_demo.ipynb)
- [Chess Self-Play Demo](docs/examples/chess_selfplay.ipynb)
- [Math+Binary HV Self-Play](examples/math_bin_hv/README.md)

## Topics
- hypervectors
- bayesian
- online-learning
