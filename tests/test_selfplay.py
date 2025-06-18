import subprocess
import tempfile


def test_selfplay_runs():
    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
        res = subprocess.run(
            [
                "python",
                "-m",
                "selfplay_chess.train",
                "--pgn",
                "examples/selfplay/data/seed_data.pgn",
                "--engine",
                "/usr/bin/true",
                "--nodes",
                "1",
                "--output",
                tmp.name,
            ],
            capture_output=True,
        )
        assert res.returncode == 0
