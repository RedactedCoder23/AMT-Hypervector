#!/usr/bin/env python3
"""Evaluation harness for ChessModel using seed PGN games."""
import argparse
import csv
import chess.engine
from selfplay_chess.data import load_games
from selfplay_chess.model import ChessModel


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "-p",
        "--pgn",
        default="examples/selfplay/data/seed_data.pgn",
        help="Seed PGN file",
    )
    parser.add_argument(
        "-e",
        "--engine",
        default=None,
        help="Path to UCI engine (optional)",
    )
    parser.add_argument(
        "-n",
        "--nodes",
        type=int,
        default=1000,
        help="Engine node limit",
    )
    parser.add_argument("-d", "--dim", type=int, default=6)
    parser.add_argument("-a", "--alpha", nargs="+", type=float)
    parser.add_argument(
        "-o",
        "--output",
        default="examples/selfplay/data/eval_results.csv",
    )
    args = parser.parse_args()

    model = ChessModel(dim=args.dim, alpha=args.alpha)
    # fmt: off
    engine = (
        chess.engine.SimpleEngine.popen_uci(args.engine)
        if args.engine
        else None
    )
    # fmt: on

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        header = ["move_uci", "bhre_score"]
        if engine:
            header.append("engine_score")
        w.writerow(header)

        for board in load_games(args.pgn):
            # fmt: off
            move = (
                board.peek()
                if hasattr(board, "peek")
                else board.move_stack[-1]
            )
            # fmt: on
            bhre_score = model.predict(board, move)
            row = [move.uci(), bhre_score]
            if engine:
                info = engine.analyse(
                    board,
                    chess.engine.Limit(nodes=args.nodes),
                )
                engine_score = (
                    info["score"].pov(board.turn).score(mate_score=10000) or 0
                )
                row.append(engine_score)
            w.writerow(row)

    if engine:
        engine.quit()
    print(f"Eval complete. Results in {args.output}")


if __name__ == "__main__":
    main()
