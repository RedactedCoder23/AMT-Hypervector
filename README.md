# AMT-Hypervector
![CI](https://github.com/yourusername/AMT-Hypervector/actions/workflows/tests.yml/badge.svg)

**Adaptive, Modular, Transparent Hypervector Reasoning (BHRE/AMT)**

- Deterministic 6-dimensional hypervector encoding (SHA256 → sinc)
- Dual-channel self-validation (ADF) for adaptive memory
- Sticky-pool replay buffer & GPU-accelerated info-gain search
- LoRA-style adapters for LLMs and a built-in chess self-play demo

## Quick Start
```bash
git clone https://github.com/yourusername/AMT-Hypervector.git
cd AMT-Hypervector
pip install -e .
# Run chess demo
python src/plugins/chess_toy/selfplay_chess.py
# Run sentiment demo
python examples/sentiment_analysis/run_demo.py
# Run GPT-2 demo
python examples/tiny_gpt2_lora/run_demo.py
```
