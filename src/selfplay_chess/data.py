"""Data utilities for chess self-play."""

import chess.pgn
from typing import Iterator


def load_games(pgn_path: str) -> Iterator[chess.Board]:
    """Yield successive board states from games in a PGN file."""
    with open(pgn_path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                yield board
