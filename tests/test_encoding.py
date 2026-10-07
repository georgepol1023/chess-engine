import random

import chess
import numpy as np
import torch

from hc.data import game_records, split_games
from hc.encoding import (N_MOVES, encode_board, index_to_move, legal_indices, move_to_index,
                         oriented, planes_from_pieces, record_to_board)
from hc.model import HumanNet

from .fake_pgn import fake_game


def random_boards(n=300, seed=0):
    rnd = random.Random(seed)
    board = chess.Board()
    for _ in range(n):
        if board.is_game_over() or board.fullmove_number > 60:
            board = chess.Board()
        yield board.copy()
        board.push(rnd.choice(list(board.legal_moves)))


PROMO_FENS = ["8/1P6/8/8/8/8/k6K/8 w - - 0 1", "8/8/8/8/8/k7/1p5K/2R5 b - - 0 1",
              "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
              "8/8/8/3pP3/8/8/k6K/8 w - d6 0 1"]


def all_boards():
    yield from random_boards()
    for f in PROMO_FENS:
        yield chess.Board(f)


def test_move_index_roundtrip():
    for board in all_boards():
        seen = set()
        for m in board.legal_moves:
            i = move_to_index(board, m)
            assert 0 <= i < N_MOVES
            assert index_to_move(board, i) == m
            seen.add(i)
        assert len(seen) == board.legal_moves.count()  # no two legal moves share an index


def test_record_rebuilds_position():
    for board in all_boards():
        pieces, castling, ep = encode_board(board)
        rec = np.zeros(1, dtype=[("pieces", np.int8, 64), ("castling", np.uint8), ("ep_file", np.int8)])[0]
        rec["pieces"], rec["castling"], rec["ep_file"] = pieces, castling, ep
        rebuilt = record_to_board(rec)
        assert {m.uci() for m in rebuilt.legal_moves} == {m.uci() for m in oriented(board).legal_moves}


def test_planes_shape_and_content():
    board = chess.Board()
    pieces, castling, ep = encode_board(board)
    t = lambda a, dt: torch.tensor(np.asarray(a)[None], dtype=dt)
    planes = planes_from_pieces(t(pieces, torch.int8), t(castling, torch.uint8), t(ep, torch.int8))
    assert planes.shape == (1, 17, 8, 8)
    assert planes[0, :12].sum() == 32            # 32 pieces on the board
    assert planes[0, 12:16].sum() == 4 * 64      # all four castling rights


def test_pgn_parsing_with_clocks():
    rnd = random.Random(3)
    texts = [fake_game(i, rnd) for i in range(5)]
    games = list(split_games("".join(texts).splitlines(keepends=True)))
    assert len(games) == 5
    rec = game_records(games[0])
    assert len(rec) > 0
    assert (rec["clock"] > 0).all()
    assert set(np.unique(rec["result"])) <= {-1, 0, 1}


def test_model_forward():
    model = HumanNet(channels=16, blocks=2)
    planes, sc = torch.zeros(3, 17, 8, 8), torch.zeros(3, 6)
    pol, time, out = model(planes, sc)
    assert pol.shape == (3, N_MOVES) and time.shape == (3, 2) and out.shape == (3, 3)
