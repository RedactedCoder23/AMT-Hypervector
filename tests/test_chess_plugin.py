import subprocess


def test_chess_runs():
    res = subprocess.run(
        ["python", "src/plugins/chess_toy/selfplay_chess.py", "--help"],
        capture_output=True
    )
    assert res.returncode == 0
