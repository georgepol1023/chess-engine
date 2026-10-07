"""Measure how human-like the model is, broken down by rating.

    python evaluate.py checkpoints/run/model.pt

The core experiment: take held-out positions from players in each rating band,
then ask the model "what would a player of rating R play here?" for every R.
If rating conditioning works, accuracy is highest when R matches the real
player's rating, giving a bright diagonal in the heatmap.

When the model's rating input is changed, the opponent's rating is shifted by
the same amount, so the rating gap between the two players is preserved.

Positions with only one legal move are left out (they're free points).

Writes results/<run>/: eval.json, results.md, heatmap.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hc.data import to_tensors
from hc.encoding import N_MOVES, legal_indices, record_to_board
from hc.model import best_device, load

BANDS = list(range(800, 2600, 200))  # band [b, b+200)


def band_of(elo):
    return np.clip((np.asarray(elo) - BANDS[0]) // 200, 0, len(BANDS) - 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--data", type=Path, default=Path("data/val.npy"))
    p.add_argument("--positions", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", default=best_device())
    args = p.parse_args()

    dev = args.device
    model = load(args.checkpoint, dev)
    rec_all = np.load(args.data, mmap_mode="r")
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(rec_all), size=min(args.positions, len(rec_all)), replace=False))
    rec = np.array(rec_all[idx])

    print("generating legal moves...")
    legal = [legal_indices(record_to_board(r)) for r in rec]
    keep = np.array([len(l) > 1 for l in legal])
    rec, legal = rec[keep], [l for l, k in zip(legal, keep) if k]
    n = len(rec)
    band = band_of(rec["elo"])
    print(f"{n:,} positions (forced moves removed)")

    def run(elo_override=None):
        """Masked top-1 correctness, plus think-time and outcome predictions."""
        correct = np.zeros(n, dtype=bool)
        t_mean = np.zeros(n, dtype=np.float32)
        outcome = np.zeros((n, 3), dtype=np.float32)
        for i in range(0, n, args.batch_size):
            r = rec[i:i + args.batch_size].copy()
            if elo_override is not None:
                delta = elo_override - r["elo"].astype(np.int32)
                r["elo"] = elo_override  # a single rating for every position
                r["opp_elo"] = np.clip(r["opp_elo"] + delta, 400, 3200)
            b = to_tensors(r, dev)
            with torch.no_grad():
                pol, tp, out = model(b["planes"], b["scalars"])
            mask = torch.zeros(len(r), N_MOVES, dtype=torch.bool, device=dev)
            for j, l in enumerate(legal[i:i + len(r)]):
                mask[j, torch.from_numpy(l).to(dev)] = True
            pol = pol.float().masked_fill(~mask, float("-inf"))
            correct[i:i + len(r)] = (pol.argmax(1) == b["move"]).cpu().numpy()
            t_mean[i:i + len(r)] = tp[:, 0].float().cpu().numpy()
            outcome[i:i + len(r)] = torch.softmax(out.float(), 1).cpu().numpy()
        return correct, t_mean, outcome

    # 1. model with the true ratings
    correct, t_mean, outcome = run()
    results = {"positions": n, "accuracy": float(correct.mean())}

    # 2. rating sweep: every real band x every model rating
    print("rating sweep...")
    centres = [b + 100 for b in BANDS]
    matrix = np.full((len(BANDS), len(centres)), np.nan)
    counts = np.bincount(band, minlength=len(BANDS))
    for j, c in enumerate(centres):
        cor, _, _ = run(c)
        for i in range(len(BANDS)):
            if counts[i] >= 200:
                matrix[i, j] = cor[band == i].mean()
        print(f"  model rating {c}: {cor.mean():.4f}")
    results["bands"] = [f"{b}-{b + 199}" for b in BANDS]
    results["band_counts"] = counts.tolist()
    results["sweep_model_ratings"] = centres
    results["sweep_accuracy"] = [[None if np.isnan(x) else float(x) for x in row] for row in matrix]
    results["accuracy_by_band_true_rating"] = [
        float(correct[band == i].mean()) if counts[i] >= 200 else None for i in range(len(BANDS))]

    # 3. time pressure
    clock_edges = [0, 10, 30, 60, 1e9]
    names = ["<10s", "10-30s", "30-60s", ">60s"]
    cb = np.digitize(rec["clock"], clock_edges[1:-1])
    results["accuracy_by_clock"] = {nm: {"accuracy": float(correct[cb == k].mean()), "positions": int((cb == k).sum())}
                                    for k, nm in enumerate(names) if (cb == k).sum() >= 200}

    # 4. think time: rank correlation between predicted and actual
    known = rec["spent"] >= 0
    if known.sum() > 100:
        from scipy.stats import spearmanr
        rho = spearmanr(t_mean[known], np.log1p(rec["spent"][known])).correlation
        results["think_time_spearman"] = float(rho)
        results["think_time_median_abs_error_s"] = float(np.median(np.abs(np.expm1(t_mean[known]) - rec["spent"][known])))

    # 5. outcome head
    y = rec["result"].astype(int) + 1
    results["outcome_accuracy"] = float((outcome.argmax(1) == y).mean())

    out = Path("results") / args.checkpoint.parent.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(results, indent=1))
    write_markdown(results, out / "results.md")
    plot(matrix, results, out / "heatmap.png")
    print((out / "results.md").read_text())


def write_markdown(r, path):
    lines = [f"Overall move-matching accuracy: **{r['accuracy']:.1%}** on {r['positions']:,} held-out positions.", "",
             "| Player rating | Positions | Accuracy (true rating) | Accuracy (model told 1500) |", "|---|---|---|---|"]
    j1500 = r["sweep_model_ratings"].index(1500) if 1500 in r["sweep_model_ratings"] else None
    for i, b in enumerate(r["bands"]):
        a = r["accuracy_by_band_true_rating"][i]
        if a is None:
            continue
        fixed = r["sweep_accuracy"][i][j1500] if j1500 is not None else None
        lines.append(f"| {b} | {r['band_counts'][i]:,} | {a:.1%} | {fixed:.1%} |" if fixed is not None
                     else f"| {b} | {r['band_counts'][i]:,} | {a:.1%} | n/a |")
    lines += ["", "| Clock remaining | Positions | Accuracy |", "|---|---|---|"]
    lines += [f"| {k} | {v['positions']:,} | {v['accuracy']:.1%} |" for k, v in r["accuracy_by_clock"].items()]
    if "think_time_spearman" in r:
        lines += ["", f"Think time: Spearman correlation {r['think_time_spearman']:.2f} between predicted and actual, "
                      f"median error {r['think_time_median_abs_error_s']:.1f}s."]
    lines += ["", f"Outcome head: predicts the final result correctly {r['outcome_accuracy']:.1%} of the time."]
    Path(path).write_text("\n".join(lines) + "\n")


def plot(matrix, r, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [i for i in range(len(r["bands"])) if not np.all(np.isnan(matrix[i]))]
    m = matrix[rows]
    # show each row relative to its own best, so the diagonal is visible across bands
    rel = m - np.nanmax(m, axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(rel * 100, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(r["sweep_model_ratings"])), r["sweep_model_ratings"], rotation=45)
    ax.set_yticks(range(len(rows)), [r["bands"][i] for i in rows])
    ax.set_xlabel("rating the model was told")
    ax.set_ylabel("real player's rating")
    for i in range(len(rows)):
        for j in range(m.shape[1]):
            if not np.isnan(m[i, j]):
                ax.text(j, i, f"{m[i, j] * 100:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if rel[i, j] < -0.01 else "black")
    fig.colorbar(im, ax=ax, label="accuracy vs best in row (points)")
    ax.set_title("Move-matching accuracy (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
