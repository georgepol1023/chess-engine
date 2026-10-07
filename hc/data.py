"""Turn Lichess PGN games into training records, and load them back for training."""

import io
import re
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch

from .encoding import RECORD_DTYPE, encode_board, move_to_index, planes_from_pieces

_ELO = re.compile(r'\[(White|Black)Elo "(\d+)"\]')
_TC = re.compile(r'\[TimeControl "(\d+)\+(\d+)"\]')
_RESULT = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}


@dataclass
class Filter:
    min_elo: int = 600
    max_elo: int = 2900
    min_base: int = 180     # 3 minutes: skip bullet, where play is mostly reflexes
    max_base: int = 1800
    min_plies: int = 10


def quick_accept(text: str, f: Filter) -> bool:
    """Cheap header checks on raw PGN text, before paying for a full parse."""
    if "%clk" not in text or '[Event "Rated' not in text or 'Termination "Abandoned"' in text:
        return False
    elos = [int(m.group(2)) for m in _ELO.finditer(text)]
    tc = _TC.search(text)
    if len(elos) != 2 or not tc:
        return False
    base = int(tc.group(1))
    return all(f.min_elo <= e <= f.max_elo for e in elos) and f.min_base <= base <= f.max_base


def game_records(text: str, f: Filter = Filter()) -> np.ndarray:
    """All positions of one game as records. Empty if the game is unusable."""
    game = chess.pgn.read_game(io.StringIO(text))
    empty = np.zeros(0, dtype=RECORD_DTYPE)
    if game is None or game.errors:
        return empty
    h = game.headers
    try:
        elo = {chess.WHITE: int(h["WhiteElo"]), chess.BLACK: int(h["BlackElo"])}
        base, inc = (int(x) for x in h["TimeControl"].split("+"))
        result = _RESULT[h["Result"]]
    except (KeyError, ValueError):
        return empty

    nodes = list(game.mainline())
    if len(nodes) < f.min_plies:
        return empty
    out = np.zeros(len(nodes), dtype=RECORD_DTYPE)
    board = game.board()
    clock = {chess.WHITE: float(base), chess.BLACK: float(base)}

    for i, node in enumerate(nodes):
        after = node.clock()
        if after is None:
            return empty
        turn = board.turn
        pieces, castling, ep = encode_board(board)
        r = out[i]
        r["pieces"], r["castling"], r["ep_file"] = pieces, castling, ep
        r["elo"], r["opp_elo"] = elo[turn], elo[not turn]
        r["clock"], r["opp_clock"] = clock[turn], clock[not turn]
        r["base"], r["inc"], r["ply"] = base, inc, i
        r["move"] = move_to_index(board, node.move)
        # Lichess clocks are recorded after the increment is added
        r["spent"] = max(0.0, clock[turn] - after + inc) if i >= 2 else -1.0
        r["result"] = result if turn == chess.WHITE else -result
        clock[turn] = after
        board.push(node.move)
    return out


def split_games(lines):
    """Yield raw PGN text, one game at a time, from an iterable of lines."""
    buf, in_moves = [], False
    for line in lines:
        if line.startswith("[Event ") and in_moves:
            yield "".join(buf)
            buf, in_moves = [], False
        if line.strip() and not line.startswith("["):
            in_moves = True
        buf.append(line)
    if buf and in_moves:
        yield "".join(buf)


class Positions:
    """Memory-mapped records with fast GPU batching."""

    def __init__(self, path, device="cpu"):
        self.rec = np.load(path, mmap_mode="r")
        self.device = device

    def __len__(self):
        return len(self.rec)

    def batch(self, idx):
        r = self.rec[np.sort(idx)]  # sorted reads are much kinder to the disk
        return to_tensors(r, self.device)


def to_tensors(r: np.ndarray, device):
    from .encoding import record_scalars

    t = lambda a, dt=None: torch.from_numpy(np.ascontiguousarray(a)).to(device, dtype=dt)
    planes = planes_from_pieces(t(r["pieces"]), t(r["castling"]), t(r["ep_file"]))
    return {
        "planes": planes,
        "scalars": t(record_scalars(r)),
        "move": t(r["move"], torch.long),
        "spent": t(r["spent"]),
        "result": t(r["result"].astype(np.int64) + 1),   # 0 loss, 1 draw, 2 win
    }


def save_chunk(records: list[np.ndarray], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.concatenate(records) if records else np.zeros(0, RECORD_DTYPE))


def merge_chunks(chunks: list[Path], out: Path):
    sizes = [len(np.load(c, mmap_mode="r")) for c in chunks]
    mm = np.lib.format.open_memmap(out, mode="w+", dtype=RECORD_DTYPE, shape=(sum(sizes),))
    pos = 0
    for c, n in zip(chunks, sizes):
        mm[pos:pos + n] = np.load(c)
        pos += n
    mm.flush()
    for c in chunks:
        c.unlink()
    return sum(sizes)
