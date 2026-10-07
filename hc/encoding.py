"""Board and move encoding.

Everything is seen from the side to move: when it's Black's turn the board is
mirrored (colours swapped, ranks flipped), so the network only ever has to learn
to play as White. This halves what it has to learn and is standard practice.

A position is stored as a compact record (see RECORD_DTYPE) rather than as
planes, so millions of positions fit in RAM. Planes are built per batch on the
GPU in `planes_from_pieces`.

Moves are indexed 0..4167:
    from_square * 64 + to_square                for normal moves (queen promotions included)
    4096 + (from_file * 3 + dir) * 3 + piece    for under-promotions, dir = to_file - from_file + 1,
                                                piece 0/1/2 = knight/bishop/rook
"""

import numpy as np
import chess

N_MOVES = 4096 + 72
N_PIECE_PLANES = 12
N_PLANES = N_PIECE_PLANES + 4 + 1  # pieces, castling rights, en passant
N_SCALARS = 6

UNDERPROMOS = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}
UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

RECORD_DTYPE = np.dtype([
    ("pieces", np.int8, 64),   # 0 empty, 1-6 own PNBRQK, 7-12 opponent PNBRQK
    ("castling", np.uint8),    # bits: own K, own Q, opp K, opp Q
    ("ep_file", np.int8),      # en passant file, or -1
    ("elo", np.int16),         # rating of the player to move
    ("opp_elo", np.int16),
    ("clock", np.float32),     # seconds left for the player to move, before moving
    ("opp_clock", np.float32),
    ("base", np.int16),        # time control, seconds
    ("inc", np.int16),
    ("ply", np.int16),
    ("move", np.int16),        # move played, as an index
    ("spent", np.float32),     # seconds spent on the move, -1 if unknown
    ("result", np.int8),       # final result for the player to move: 1 win, 0 draw, -1 loss
])


def oriented(board: chess.Board) -> chess.Board:
    """The board from the side to move's point of view."""
    return board if board.turn == chess.WHITE else board.mirror()


def encode_board(board: chess.Board) -> tuple[np.ndarray, int, int]:
    """(pieces[64], castling bits, ep_file) for the side to move."""
    b = oriented(board)
    pieces = np.zeros(64, dtype=np.int8)
    for sq, p in b.piece_map().items():
        pieces[sq] = p.piece_type + (0 if p.color == chess.WHITE else 6)
    castling = (int(b.has_kingside_castling_rights(chess.WHITE))
                | int(b.has_queenside_castling_rights(chess.WHITE)) << 1
                | int(b.has_kingside_castling_rights(chess.BLACK)) << 2
                | int(b.has_queenside_castling_rights(chess.BLACK)) << 3)
    ep = chess.square_file(b.ep_square) if b.ep_square is not None and b.has_legal_en_passant() else -1
    return pieces, castling, ep


def _mirror_move(m: chess.Move) -> chess.Move:
    return chess.Move(chess.square_mirror(m.from_square), chess.square_mirror(m.to_square), m.promotion)


def move_to_index(board: chess.Board, move: chess.Move) -> int:
    """Index of a move played in `board` (by the side to move)."""
    m = move if board.turn == chess.WHITE else _mirror_move(move)
    if m.promotion in UNDERPROMOS:
        ff, tf = chess.square_file(m.from_square), chess.square_file(m.to_square)
        return 4096 + (ff * 3 + (tf - ff + 1)) * 3 + UNDERPROMOS[m.promotion]
    return m.from_square * 64 + m.to_square


def index_to_move(board: chess.Board, idx: int) -> chess.Move:
    """Inverse of move_to_index for a position where it's `board.turn` to move."""
    if idx < 4096:
        frm, to = divmod(idx, 64)
        promo = None
        b = oriented(board)
        p = b.piece_at(frm)
        if p and p.piece_type == chess.PAWN and chess.square_rank(to) == 7:
            promo = chess.QUEEN
        m = chess.Move(frm, to, promo)
    else:
        k = idx - 4096
        ff_dir, piece = divmod(k, 3)
        ff, d = divmod(ff_dir, 3)
        m = chess.Move(chess.square(ff, 6), chess.square(ff + d - 1, 7), UNDERPROMO_PIECES[piece])
    return m if board.turn == chess.WHITE else _mirror_move(m)


def legal_indices(board: chess.Board) -> np.ndarray:
    return np.array([move_to_index(board, m) for m in board.legal_moves], dtype=np.int64)


def record_to_board(rec) -> chess.Board:
    """Rebuild a board (oriented: side to move is White) from a record."""
    b = chess.Board(None)
    for sq in range(64):
        code = int(rec["pieces"][sq])
        if code:
            color = chess.WHITE if code <= 6 else chess.BLACK
            b.set_piece_at(sq, chess.Piece((code - 1) % 6 + 1, color))
    b.turn = chess.WHITE
    c = int(rec["castling"])
    fen_c = "".join(ch for bit, ch in zip(range(4), "KQkq") if c >> bit & 1) or "-"
    b.set_castling_fen(fen_c)
    if rec["ep_file"] >= 0:
        b.ep_square = chess.square(int(rec["ep_file"]), 5)
    return b


def scalars(elo, opp_elo, clock, opp_clock, base, inc) -> np.ndarray:
    """Conditioning inputs, roughly normalised to [-2, 2]."""
    elo, opp_elo = np.asarray(elo, np.float32), np.asarray(opp_elo, np.float32)
    return np.stack([
        (elo - 1500) / 500,
        (opp_elo - 1500) / 500,
        np.log1p(np.maximum(clock, 0)) / 5 - 0.8,
        np.log1p(np.maximum(opp_clock, 0)) / 5 - 0.8,
        np.log1p(np.asarray(base, np.float32)) / 5 - 1.2,
        np.asarray(inc, np.float32) / 10,
    ], axis=-1).astype(np.float32)


def record_scalars(rec) -> np.ndarray:
    return scalars(rec["elo"], rec["opp_elo"], rec["clock"], rec["opp_clock"], rec["base"], rec["inc"])


def planes_from_pieces(pieces, castling, ep_file):
    """Batch of records -> (B, 17, 8, 8) float tensor. Works on torch tensors (any device)."""
    import torch

    B = pieces.shape[0]
    codes = torch.arange(1, 13, device=pieces.device).view(1, 12, 1)
    piece_planes = (pieces.long().unsqueeze(1) == codes).float()          # (B, 12, 64)
    bits = torch.arange(4, device=pieces.device).view(1, 4, 1)
    castle = ((castling.long().view(B, 1, 1) >> bits) & 1).float().expand(B, 4, 64)
    ep = torch.zeros(B, 1, 64, device=pieces.device)
    has_ep = ep_file >= 0
    if has_ep.any():
        rows = has_ep.nonzero().flatten()
        ep[rows, 0, 40 + ep_file[rows].long()] = 1.0                       # rank 6 from mover's view
    return torch.cat([piece_planes, castle, ep], dim=1).view(B, N_PLANES, 8, 8)
