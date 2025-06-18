#!/usr/bin/env python3
"""
selfplay_chess.py

Hypervector Chess Self-Play with BHRE adaptation & CUDA.
• Configurable MLP (default ~125M params)
• Buffer seeded from external PGN
• Batched self-play + Stockfish scoring for speed
• Persistent CSV logging
"""

import os
import hashlib
import random
import csv
import numpy as np
import argparse
import chess
import chess.engine
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
stockfish_path = "stockfish"  # or full path if needed
dims           = 6
alpha          = 0.5
buffer_size    = 100_000
batch_size     = 4096
lr             = 1e-3
games_per_iter = 16
eval_games     = 10
hidden_size    = 5000
num_layers     = 6
num_iters      = 20
start_eps      = 0.1
end_eps        = 0.01
seed_pgn       = "seed_data.pgn"
max_seed_games = 200
# ────────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.empty_cache()

# Fixed sink/quark axes
sink_axis  = torch.eye(dims, device=device)[0]
quark_axes = torch.eye(dims, device=device)[1:5]

# ADF channels init
mu_pos = torch.randn(dims, device=device); mu_pos /= mu_pos.norm()
mu_neg = mu_pos.clone()


def adf_update(mu_p, mu_n, α=alpha):
    Δ = mu_p - mu_n
    mu_p = mu_p + α * Δ
    mu_n = mu_n - α * Δ
    return mu_p / (mu_p.norm()+1e-9), mu_n / (mu_n.norm()+1e-9)


def encode_piece(sq, p):
    h = hashlib.sha256(f"{p.symbol()}_{sq}".encode()).digest()
    v = np.frombuffer(h[:dims], dtype=np.uint8).astype(np.float32)
    v = ((v/255)*2 - 1)
    x = torch.from_numpy(v).to(device)
    x = x / (x.norm()+1e-9)
    return torch.sinc(x)


def extract_features(board):
    v = torch.zeros(dims, device=device)
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            v += encode_piece(sq, p)
    sink   = v.dot(sink_axis)
    quarks = torch.stack([v.dot(q) for q in quark_axes])
    return torch.cat([
        sink.unsqueeze(0),
        quarks,
        v.dot(mu_pos).unsqueeze(0),
        v.dot(mu_neg).unsqueeze(0),
    ])


class BigMLP(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_dim = dims + 2  # 1 sink + 4 quarks + 2 ADF dims = 7
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_size), nn.ReLU()]
            in_dim = hidden_size
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_seed_data(path, engine, buf, max_games):
    if not os.path.exists(path):
        return
    with open(path, 'r') as pgn:
        g = 0
        while g < max_games:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            board = game.board()
            for mv in game.mainline_moves():
                board.push(mv)
                feat = extract_features(board).cpu().numpy()
                info = engine.analyse(board, chess.engine.Limit(depth=1))
                raw  = info["score"].white().score(mate_score=100000) / 100.0
                buf.append((feat, np.clip(raw, -10.0, 10.0)))
            g += 1


def train_with_bhre(model, opt, engine, buf, eps):
    global mu_pos, mu_neg
    collected = []
    for _ in range(games_per_iter):
        board = chess.Board()
        while not board.is_game_over():
            feats = []
            for mv in board.legal_moves:
                board.push(mv)
                feats.append(extract_features(board))
                board.pop()
            X = torch.stack(feats)
            with torch.no_grad():
                scores = model(X)
            move = random.choice(list(board.legal_moves)) if random.random() < eps else list(board.legal_moves)[scores.argmax().item()]
            board.push(move)
            for mv2, f in zip(list(board.legal_moves), feats):
                board.push(mv2)
                info = engine.analyse(board, chess.engine.Limit(depth=1))
                raw  = info["score"].white().score(mate_score=100000)
                board.pop()
                collected.append((f.cpu().numpy(), raw/100.0))
    for feat, sc in collected:
        buf.append((feat, np.clip(sc, -10.0, 10.0)))

    losses = []
    for _ in range(4):
        batch = random.sample(buf, min(len(buf), batch_size))
        Xb = torch.tensor([b[0] for b in batch], device=device)
        yb = torch.tensor([b[1] for b in batch], device=device)
        pred = model(Xb)
        loss = nn.functional.smooth_l1_loss(pred, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())

    mu_pos, mu_neg = adf_update(mu_pos, mu_neg)
    return sum(losses) / len(losses)


def eval_vs_sf(model, engine):
    model.eval()
    w = d = l = 0
    for _ in range(eval_games):
        board = chess.Board()
        while not board.is_game_over():
            if board.turn:
                feats = [extract_features(board.copy(stack=False).push(mv) or board) for mv in board.legal_moves]
                X = torch.stack(feats)
                idx = model(X).argmax().item()
                board.push(list(board.legal_moves)[idx])
            else:
                board.push(engine.play(board, chess.engine.Limit(depth=1)).move)
        res = board.result()
        if   res == "1-0":
            w += 1
        elif res == "0-1":
            l += 1
        else:
            d += 1
    print(f"Win/Draw/Loss: {w}/{d}/{l}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moves", type=int, default=50)
    args = parser.parse_args()
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads":2})
    model = BigMLP().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    buf = deque(maxlen=buffer_size)
    load_seed_data(seed_pgn, engine, buf, max_seed_games)
    for itr in range(num_iters):
        eps = start_eps * (1-itr/(num_iters-1))**2 + end_eps * (itr/(num_iters-1))
        loss = train_with_bhre(model, opt, engine, buf, eps)
        print(f"Iter {itr+1}/{num_iters}, Loss {loss:.3f}")
        eval_vs_sf(model, engine)
    engine.quit()


if __name__ == "__main__":
    main()

