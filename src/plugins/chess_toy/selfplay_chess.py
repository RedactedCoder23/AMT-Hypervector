import argparse
from pathlib import Path
import chess
import chess.engine


def _find_engine() -> str:
    for path in ["/usr/games/stockfish", "stockfish"]:
        if Path(path).exists():
            return path
    raise FileNotFoundError("Stockfish engine not found")


def main():
    parser = argparse.ArgumentParser(description="Self-play chess demo")
    parser.add_argument("--moves", type=int, default=10, help="number of half moves")
    args = parser.parse_args()

    engine_path = _find_engine()
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    board = chess.Board()
    try:
        for _ in range(args.moves):
            if board.is_game_over():
                break
            result = engine.play(board, chess.engine.Limit(time=0.01))
            board.push(result.move)
    finally:
        engine.quit()
    print(board)


if __name__ == "__main__":
    main()
