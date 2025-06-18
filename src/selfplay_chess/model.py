"""ChessModel: wraps BHRE encoder + ADFMemory for move prediction."""

from amt.encoder import HypervectorEncoder
from amt.adf_update import ADFMemory
import chess
from typing import Sequence, Tuple


class ChessModel:
    def __init__(self, dim: int = 6, alpha: Sequence[float] | None = None):
        self.enc = HypervectorEncoder(dim=dim, alpha=alpha or [1.0] * dim)
        self.mem = ADFMemory(dim=dim)

    def predict(self, board: chess.Board, move: chess.Move) -> float:
        """Encode move and return difference of positive and negative cosines."""
        hv = self.enc.encode(move.uci())
        pos, neg = self.mem.similarity_table([hv])[0]
        return float(pos - neg)

    def update(self, move: chess.Move, positive: bool) -> None:
        """Update memory with hypervector for move."""
        hv = self.enc.encode(move.uci())
        self.mem.update(hv, positive)

    def similarity(self, board: chess.Board, move: chess.Move) -> Tuple[float, float]:
        """Return cosine similarity to positive and negative memories."""
        hv = self.enc.encode(move.uci())
        pos, neg = self.mem.similarity_table([hv])[0]
        return float(pos), float(neg)
