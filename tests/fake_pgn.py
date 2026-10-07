"""Synthetic Lichess-style games for testing the pipeline end to end.

Higher-rated fake players grab material more often and play randomly less,
so a model trained on these should show at least some rating dependence.
Not real chess data; only for tests.
"""

import random

import chess

VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def pick(board, elo, rnd):
    moves = list(board.legal_moves)
    greedy = min(0.95, max(0.05, (elo - 600) / 1800))
    if rnd.random() < greedy:
        caps = [m for m in moves if board.is_capture(m)]
        if caps:
            def gain(m):
                p = board.piece_at(m.to_square)
                return VALUES[p.piece_type] if p else 1
            return max(caps, key=gain)
    return rnd.choice(moves)


def fake_game(i, rnd):
    w, b = rnd.randint(800, 2400), rnd.randint(800, 2400)
    base, inc = rnd.choice([(180, 2), (300, 0), (300, 3), (600, 0)])
    board, clocks, moves = chess.Board(), [base, base], []
    for ply in range(rnd.randint(20, 80)):
        if board.is_game_over():
            break
        mv = pick(board, w if board.turn else b, rnd)
        side = 0 if board.turn else 1
        clocks[side] = max(1, clocks[side] - rnd.uniform(0.5, 12) + inc)
        moves.append((board.san(mv), clocks[side]))
        board.push(mv)
    res = board.result() if board.is_game_over() else rnd.choice(["1-0", "0-1", "1/2-1/2"])
    body = []
    for k, (san, c) in enumerate(moves):
        h, m, s = int(c // 3600), int(c % 3600 // 60), int(c % 60)
        num = f"{k // 2 + 1}. " if k % 2 == 0 else f"{k // 2 + 1}... "
        body.append(f"{num}{san} {{ [%clk {h}:{m:02d}:{s:02d}] }}")
    return (f'[Event "Rated Blitz game"]\n[Site "https://lichess.org/fake{i:07d}"]\n'
            f'[White "w{i}"]\n[Black "b{i}"]\n[Result "{res}"]\n[WhiteElo "{w}"]\n[BlackElo "{b}"]\n'
            f'[TimeControl "{base}+{inc}"]\n[Termination "Normal"]\n\n' + " ".join(body) + f" {res}\n\n")


def write(path, n, seed=0):
    rnd = random.Random(seed)
    with open(path, "w") as f:
        for i in range(n):
            f.write(fake_game(i, rnd))


if __name__ == "__main__":
    import sys
    write(sys.argv[1], int(sys.argv[2]))
