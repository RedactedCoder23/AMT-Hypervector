import chess
from plugins.chess_toy import play_self_game


def test_play_self_game_stockfish():
    board = play_self_game(moves=2)
    assert board.is_valid()
    assert board.fullmove_number >= 1
