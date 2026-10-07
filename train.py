"""Train the human move predictor.

    python train.py --epochs 2
    python train.py --channels 64 --blocks 6 --epochs 1     # quick, small model

Uses CUDA, Apple MPS or CPU automatically. Writes checkpoints/<name>/:
    model.pt      latest weights
    log.jsonl     training and validation metrics
"""

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hc.data import Positions
from hc.model import HumanNet, best_device, save


def losses(model, b, w_time, w_outcome):
    policy, time_pred, outcome = model(b["planes"], b["scalars"])
    l_policy = F.cross_entropy(policy.float(), b["move"])

    # think time: Gaussian negative log-likelihood on log(1 + seconds)
    known = b["spent"] >= 0
    if known.any():
        mean, log_std = time_pred[known, 0].float(), time_pred[known, 1].float().clamp(-3, 3)
        target = torch.log1p(b["spent"][known])
        l_time = (log_std + 0.5 * ((target - mean) / log_std.exp()) ** 2).mean()
    else:
        l_time = policy.new_zeros(())

    l_outcome = F.cross_entropy(outcome.float(), b["result"])
    total = l_policy + w_time * l_time + w_outcome * l_outcome
    acc = (policy.argmax(1) == b["move"]).float().mean()
    return total, {"policy": l_policy.item(), "time": l_time.item(),
                   "outcome": l_outcome.item(), "acc": acc.item()}


@torch.no_grad()
def validate(model, val, n, batch_size, args, amp_ctx):
    model.eval()
    rng = np.random.default_rng(0)  # same validation positions every time
    idx = rng.choice(len(val), size=min(n, len(val)), replace=False)
    sums, count = {}, 0
    for i in range(0, len(idx), batch_size):
        b = val.batch(idx[i:i + batch_size])
        with amp_ctx():
            _, m = losses(model, b, args.w_time, args.w_outcome)
        k = len(b["move"])
        for key, v in m.items():
            sums[key] = sums.get(key, 0.0) + v * k
        count += k
    model.train()
    return {f"val_{k}": v / count for k, v in sums.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--name", default="run")
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--blocks", type=int, default=8)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--w-time", type=float, default=0.1)
    p.add_argument("--w-outcome", type=float, default=0.25)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--val-positions", type=int, default=50_000)
    p.add_argument("--device", default=best_device())
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    dev = args.device
    torch.manual_seed(0)
    train, val = Positions(args.data / "train.npy", dev), Positions(args.data / "val.npy", dev)
    print(f"device {dev}: {len(train):,} train / {len(val):,} val positions")

    out = Path("checkpoints") / args.name
    out.mkdir(parents=True, exist_ok=True)
    model = HumanNet(args.channels, args.blocks).to(dev)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters")

    steps_per_epoch = len(train) // args.batch_size
    total_steps = max(1, int(args.epochs * steps_per_epoch))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = min(1000, total_steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warmup)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))

    step = 0
    if args.resume and (out / "model.pt").exists():
        ckpt = torch.load(out / "model.pt", map_location=dev)
        model.load_state_dict(ckpt["state"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        step = ckpt["step"]
        print(f"resumed at step {step}")

    use_bf16 = dev == "cuda" and torch.cuda.is_bf16_supported()
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 else nullcontext

    log = open(out / "log.jsonl", "a")
    rng = np.random.default_rng(step)
    t0, running = time.time(), {}
    model.train()
    while step < total_steps:
        perm = rng.permutation(len(train))
        for i in range(0, len(perm) - args.batch_size + 1, args.batch_size):
            b = train.batch(perm[i:i + args.batch_size])
            with amp_ctx():
                loss, m = losses(model, b, args.w_time, args.w_outcome)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            for k, v in m.items():
                running[k] = 0.98 * running.get(k, v) + 0.02 * v

            if step % 100 == 0:
                rate = step / (time.time() - t0) if step else 0
                print(f"step {step:7d}/{total_steps}  policy {running['policy']:.3f}  "
                      f"acc {running['acc']:.3f}  time {running['time']:.3f}  "
                      f"outcome {running['outcome']:.3f}  lr {sched.get_last_lr()[0]:.2e}")
            if step % args.eval_every == 0 or step == total_steps:
                v = validate(model, val, args.val_positions, args.batch_size, args, amp_ctx)
                row = {"step": step, "epoch": step / steps_per_epoch,
                       "seconds": time.time() - t0, **{f"train_{k}": x for k, x in running.items()}, **v}
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(f"  validation: acc {v['val_acc']:.4f}  policy loss {v['val_policy']:.4f}")
                save(model, out / "model.pt", {"opt": opt.state_dict(), "sched": sched.state_dict(),
                                              "step": step, "args": vars(args) | {"data": str(args.data)}})
            if step >= total_steps:
                break
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out / 'model.pt'}")


if __name__ == "__main__":
    main()
