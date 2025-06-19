import subprocess
import sys
from pathlib import Path


def test_math_bin_simple(tmp_path):
    work = tmp_path / "mbh"
    subprocess.run(["cp", "-r", "examples/math_bin_hv", str(work)], check=True)
    cmd = [sys.executable, "math_bin_selfplay.py", "--cycles", "1"]
    result = subprocess.run(cmd, cwd=work, capture_output=True)
    assert result.returncode == 0
    assert Path(work / "training_log.csv").exists()
