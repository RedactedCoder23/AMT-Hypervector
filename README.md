![CI](https://github.com/RedactedCoder23/AMT-Hypervector/actions/workflows/ci.yml/badge.svg)
![PyPI](https://img.shields.io/pypi/v/AMT-Hypervector)
![License](https://img.shields.io/github/license/RedactedCoder23/AMT-Hypervector)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/you/bhre-project/main)
# AMT-Hypervector

**Adaptive, Modular, Transparent Hypervector Reasoning (BHRE/AMT)**

- Deterministic 6-dimensional hypervector encoding (SHA256 → sinc)
- Dual-channel self-validation (ADF) for adaptive memory
- Sticky-pool replay buffer & GPU-accelerated info-gain search
- LoRA-style adapters for LLMs and a built-in chess self-play demo

## Quick Start
```bash
git clone https://github.com/RedactedCoder23/AMT-Hypervector.git
cd AMT-Hypervector
pip install -e .[dev]
bhre --config config.yaml
```

## Examples
- [Encode Demo](docs/examples/encode_demo.ipynb)
- [Chess Self-Play Demo](docs/examples/chess_selfplay.ipynb)
