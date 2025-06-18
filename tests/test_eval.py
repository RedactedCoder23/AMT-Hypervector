import subprocess


def test_eval_help():
    res = subprocess.run(
        ["python", "examples/eval.py", "--help"],
        capture_output=True,
    )
    assert res.returncode == 0
