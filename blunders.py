"""Do the model's mistakes look like human mistakes? Needs Stockfish installed.

    python blunders.py checkpoints/run/model.pt --stockfish /usr/local/bin/stockfish

For held-out positions in each rating band, Stockfish scores three moves:
  * the move the human actually played
  * the move the model samples at that rating (mode="sample")
  * the model's single most likely move (mode="top")

and reports, per band, the average centipawn loss and the blunder rate
(a move losing 200+ centipawns). A human-like model should track the human
numbers across bands: blundering more at 1000 than at 2000, but not more than
real 1000-rated players do.
"""

import argparse
import json
from pathlib import Path

import chess
import chess.engine
import numpy as np

from hc.encoding import index_to_move, legal_indices, record_to_board
from hc.engine import HumanEngine

BANDS = list(range(800, 2600, 200))
CAP = 1500


def score(info, pov):
    return max(-CAP, min(CAP, info["score"].pov(pov).score(mate_score=10_000)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--stockfish", default="stockfish")
    p.add_argument("--data", type=Path, default=Path("data/val.npy"))
    p.add_argument("--per-band", type=int, default=300)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--blunder-cp", type=int, default=200)
    args = p.parse_args()

    rec_all = np.load(args.data, mmap_mode="r")
    rng = np.random.default_rng(1)
    sample_eng = HumanEngine(args.checkpoint, mode="sample", seed=0)
    sf = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    limit = chess.engine.Limit(depth=args.depth)
    results = {}

    elo_all = np.asarray(rec_all["elo"])
    ply_all = np.asarray(rec_all["ply"])
    for b in BANDS:
        pool = np.nonzero((elo_all >= b) & (elo_all < b + 200) & (ply_all >= 10))[0]
        if len(pool) < 50:
            continue
        idx = rng.choice(pool, size=min(args.per_band, len(pool)), replace=False)
        losses = {"human": [], "model_sample": [], "model_top": []}
        for i in idx:
            r = rec_all[i]
            board = record_to_board(r)
            if len(legal_indices(board)) < 2:
                continue
            best = score(sf.analyse(board, limit), chess.WHITE)
            d = sample_eng.decide(board, int(r["elo"]), int(r["opp_elo"]), float(r["clock"]),
                                  float(r["opp_clock"]), int(r["base"]), int(r["inc"]))
            moves = {"human": index_to_move(board, int(r["move"])), "model_sample": d.move,
                     "model_top": d.probs[0][0]}
            for k, mv in moves.items():
                board.push(mv)
                after = score(sf.analyse(board, limit), chess.WHITE)
                board.pop()
                losses[k].append(max(0, best - after))
        results[f"{b}-{b + 199}"] = {
            k: {"mean_cp_loss": float(np.mean(v)), "blunder_rate": float(np.mean(np.array(v) >= args.blunder_cp)),
                "positions": len(v)} for k, v in losses.items()}
        row = results[f"{b}-{b + 199}"]
        print(f"{b}-{b + 199}: blunder rate  human {row['human']['blunder_rate']:.1%}  "
              f"model(sample) {row['model_sample']['blunder_rate']:.1%}  model(top) {row['model_top']['blunder_rate']:.1%}")
    sf.quit()

    out = Path("results") / args.checkpoint.parent.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "blunders.json").write_text(json.dumps(results, indent=1))
    plot(results, out / "blunders.png")


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bands = list(results)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for k, style in [("human", dict(color="#222", lw=2.5)), ("model_sample", dict(color="#d9730d", lw=2)),
                     ("model_top", dict(color="#8a94a6", lw=2, ls="--"))]:
        ax.plot(bands, [results[b][k]["blunder_rate"] * 100 for b in bands], marker="o", label=k.replace("_", " "), **style)
    ax.set_ylabel("blunder rate (%)")
    ax.set_xlabel("player rating")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
