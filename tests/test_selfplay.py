import subprocess
import sys
import pytest


@pytest.mark.parametrize(
    "script,args",
    [
        (
            "train",
            [
                "--pgn",
                "examples/selfplay/data/seed_data.pgn",
                "--engine",
                "true",
                "--nodes",
                "1",
            ],
        ),
        ("eval", ["--pgn", "examples/selfplay/data/seed_data.pgn"]),
    ],
)
def test_selfplay_scripts(script, args, tmp_path):
    out = tmp_path / "out.csv"
    cmd = [
        sys.executable,
        "-m",
        f"selfplay_chess.{script}",
        *args,
        "--output",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True)
    assert result.returncode == 0
    assert out.exists()
    with open(out) as f:
        assert f.readline().strip() != ""
