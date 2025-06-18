#!/usr/bin/env python3
import argparse
import csv
import chess
import chess.pgn
import chess.engine
from amt.encoder import HypervectorEncoder
from amt.adf_update import ADFMemory


def main():
    p = argparse.ArgumentParser("Self-play Chess with BHRE")
    p.add_argument(
        "-p",
        "--pgn",
        default="examples/selfplay/data/seed_data.pgn",
    )
    p.add_argument("-e", "--engine", default="stockfish")
    p.add_argument("-n", "--nodes", type=int, default=1000)
    p.add_argument("-d", "--dim", type=int, default=6)
    p.add_argument("-a", "--alpha", nargs="+", type=float)
    p.add_argument(
        "-o",
        "--output",
        default="examples/selfplay/training_log.csv",
    )
    args = p.parse_args()

    alpha = args.alpha or [1.0] * args.dim
    enc = HypervectorEncoder(dim=args.dim, alpha=alpha)
    mem = ADFMemory(dim=args.dim)
    try:
        engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    except Exception:
        engine = None

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["move", "score", "sim_pos", "sim_neg"])
        with open(args.pgn) as g:
            while True:
                game = chess.pgn.read_game(g)
                if game is None:
                    break
                board = game.board()
                for mv in game.mainline_moves():
                    board.push(mv)
                    score = 0
                    if engine:
                        limit = chess.engine.Limit(nodes=args.nodes)
                        info = engine.analyse(board, limit)
                        pov = info["score"].pov(board.turn)
                        score = pov.score(mate_score=10000) or 0
                    hv = enc.encode(mv.uci())
                    mem.update(hv, positive=(score > 0))
                    pos, neg = mem.similarity_table([hv])[0]
                    w.writerow([mv.uci(), score, pos, neg])
    if engine:
        engine.quit()
    print(f"Logs → {args.output}")


if __name__ == "__main__":
    main()
