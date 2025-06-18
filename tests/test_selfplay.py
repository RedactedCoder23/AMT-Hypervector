import subprocess


def test_selfplay_help():
    res = subprocess.run(["python", "-m", "selfplay_chess.train"], capture_output=True)
    assert res.returncode == 0
