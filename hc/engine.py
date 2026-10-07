"""Play like a human of a given rating."""

from dataclasses import dataclass

import chess
import numpy as np
import torch

from .encoding import encode_board, index_to_move, legal_indices, planes_from_pieces, scalars
from .model import best_device, load


@dataclass
class Decision:
    move: chess.Move
    probs: list[tuple[chess.Move, float]]   # legal moves, most likely first
    think_seconds: float
    outcome: dict                            # {"loss": p, "draw": p, "win": p} for the side to move


class HumanEngine:
    """
    mode="sample": pick moves in proportion to how often humans at this level play them.
                   Most human-like, including the occasional blunder.
    mode="top":    always the single most likely human move. Plays like a slightly
                   stronger, very consistent player at the given rating.
    temperature:   below 1 sharpens the distribution (fewer surprises), above 1 flattens it.
    """

    def __init__(self, checkpoint, device=None, mode="sample", temperature=1.0, seed=None):
        self.device = device or best_device()
        self.model = load(checkpoint, self.device)
        self.mode, self.temperature = mode, temperature
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def analyse(self, board, elo, opp_elo=None, clock=300.0, opp_clock=None, base=300, inc=0):
        opp_elo = elo if opp_elo is None else opp_elo
        opp_clock = clock if opp_clock is None else opp_clock
        pieces, castling, ep = encode_board(board)
        t = lambda a, dt: torch.tensor(np.asarray(a)[None], dtype=dt, device=self.device)
        planes = planes_from_pieces(t(pieces, torch.int8), t(castling, torch.uint8), t(ep, torch.int8))
        sc = torch.from_numpy(scalars([elo], [opp_elo], [clock], [opp_clock], [base], [inc])).to(self.device)
        policy, time_pred, outcome = self.model(planes, sc)

        legal = legal_indices(board)
        logits = policy[0, torch.from_numpy(legal).to(self.device)].float().cpu().numpy()
        probs = _softmax(logits)
        mean, log_std = time_pred[0].float().cpu().numpy()
        out = torch.softmax(outcome[0].float(), 0).cpu().numpy()
        return legal, logits, probs, (float(mean), float(log_std)), out

    def decide(self, board, elo, opp_elo=None, clock=300.0, opp_clock=None, base=300, inc=0) -> Decision:
        legal, logits, probs, (mean, log_std), out = self.analyse(board, elo, opp_elo, clock, opp_clock, base, inc)
        if self.mode == "top" or len(legal) == 1:
            k = int(np.argmax(probs))
        else:
            p = _softmax(logits / max(self.temperature, 1e-3))
            k = int(self.rng.choice(len(p), p=p))
        order = np.argsort(-probs)
        moves = [(index_to_move(board, int(legal[i])), float(probs[i])) for i in order]

        # sample a think time from the predicted distribution, never more than the clock allows
        secs = float(np.expm1(self.rng.normal(mean, np.exp(np.clip(log_std, -3, 3)))))
        secs = float(np.clip(secs, 0.1, max(0.1, clock * 0.5)))
        return Decision(index_to_move(board, int(legal[k])), moves, secs,
                        {"loss": float(out[0]), "draw": float(out[1]), "win": float(out[2])})

    def should_resign(self, decision: Decision, board: chess.Board, threshold=0.97) -> bool:
        """Resign when the model predicts a near-certain loss from here.

        The outcome head sees both ratings and clocks, so "lost" means lost for
        players of this level: a position a 2200 would convert, a 1000 might not.
        """
        return board.fullmove_number > 15 and decision.outcome["loss"] > threshold


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()
