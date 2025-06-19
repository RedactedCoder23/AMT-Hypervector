"""ChessModel: wraps BHRE encoder + ADFMemory for move scoring."""

import chess
from typing import Sequence, Tuple
from amt.encoder import HypervectorEncoder
from amt.adf_update import ADFMemory


class ChessModel:
    def __init__(self, dim: int = 6, alpha: Sequence[float] = None):
        self.enc = HypervectorEncoder(dim=dim, alpha=alpha or [1.0] * dim)
        self.mem = ADFMemory(dim=dim)

    def predict(self, board: chess.Board, move: chess.Move) -> float:
        """Return pos–neg cosine score for the move’s hypervector."""
        hv = self.enc.encode(move.uci())
        pos, neg = self.mem.similarity_table([hv])[0]
        return pos - neg

    def update(self, move: chess.Move, positive: bool) -> None:
        """Update the ADF memory based on move outcome."""
        hv = self.enc.encode(move.uci())
        self.mem.update(hv, positive)

    def similarity(self, move: chess.Move) -> Tuple[float, float]:
        """Return (cosine_positive, cosine_negative) for the move."""
        hv = self.enc.encode(move.uci())
        return tuple(self.mem.similarity_table([hv])[0])
