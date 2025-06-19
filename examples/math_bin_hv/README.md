# math_bin_hv – Toy Hypervector Self-Play (Math + Machine Code)

This prototype shows how to:
- deterministically encode **any string/token** as a 10 000-D real hypervector
- keep two running memories (`μ_pos`/`μ_neg`) via incremental mean
- generate random **math** statements or random **byte strings**
  – evaluate them with stub oracles, update memories
- surface “high-entropy” vectors (near the decision boundary)

> **Tip:** Replace stub oracles with Lean/Coq/Z3 calls or swap in your sinc-kernel encoder.

## Quick start

```bash
cd examples/math_bin_hv
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python math_bin_selfplay.py --cycles 10000
```

Check training_log.csv for per-step scores.
