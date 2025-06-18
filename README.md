![CI](https://github.com/RedactedCoder23/amt-hypervector/actions/workflows/ci.yml/badge.svg)
![PyPI](https://img.shields.io/pypi/v/amt-hypervector)
![License](https://img.shields.io/github/license/RedactedCoder23/amt-hypervector)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/RedactedCoder23/amt-hypervector/main)
# amt-hypervector

**Adaptive, Modular, Transparent Hypervector Reasoning (BHRE/AMT)**

- Deterministic 6-dimensional hypervector encoding (SHA256 → sinc)
- Dual-channel self-validation (ADF) for adaptive memory
- Sticky-pool replay buffer & GPU-accelerated info-gain search
- LoRA-style adapters for LLMs and a built-in chess self-play demo

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
- Also see `examples/sentiment_analysis/run_demo.py` for a text classification demo
  and `examples/tiny_gpt2_lora/run_demo.py` for a LoRA fine-tuning example.

## Topics
- hypervectors
- bayesian
- online-learning
