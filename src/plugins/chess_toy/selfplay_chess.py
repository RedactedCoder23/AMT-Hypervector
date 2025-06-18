import chess
import chess.engine
from pathlib import Path
from hypervector_core import ADF, encode_token


def _find_engine() -> str:
    for path in ["/usr/games/stockfish", "stockfish"]:
        if Path(path).exists():
            return path
    raise FileNotFoundError("Stockfish engine not found")


def play_self_game(moves: int = 10) -> chess.Board:
    engine_path = _find_engine()
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    board = chess.Board()
    adf = ADF(dims=6)
    try:
        for _ in range(moves):
            if board.is_game_over():
                break
            hv = encode_token(board.fen())
            adf.update()
            result = engine.play(board, chess.engine.Limit(time=0.01))
            board.push(result.move)
        return board
    finally:
        engine.quit()


if __name__ == '__main__':
    print(play_self_game())
