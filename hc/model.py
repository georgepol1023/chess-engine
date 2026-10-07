"""A residual CNN that predicts what a human would play, conditioned on rating and clock.

The conditioning inputs (both ratings, both clocks, time control) don't go in as
extra board planes. Instead each residual block is modulated by them with FiLM
(feature-wise linear modulation): a small MLP turns the scalars into a per-channel
scale and shift for every block. The same board can then be "read" differently by
a 1000 player and a 2200 player, or by someone with 5 seconds left.

Three heads:
  policy   probability of every move (4168 outputs)
  time     how long the player will think (mean and spread of log-seconds)
  outcome  win / draw / loss for the player to move (used for human-like resigning)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import N_MOVES, N_PLANES, N_SCALARS


class FiLMBlock(nn.Module):
    def __init__(self, ch: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.film = nn.Linear(cond_dim, 2 * ch)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)  # starts as a plain residual block

    def forward(self, x, cond):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        gamma, beta = self.film(cond).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        return F.relu(x + h * (1 + gamma) + beta)


class HumanNet(nn.Module):
    def __init__(self, channels: int = 128, blocks: int = 8, cond_dim: int = 64, head_dim: int = 64):
        super().__init__()
        self.config = dict(channels=channels, blocks=blocks, cond_dim=cond_dim, head_dim=head_dim)
        self.stem = nn.Sequential(nn.Conv2d(N_PLANES, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.cond = nn.Sequential(nn.Linear(N_SCALARS, cond_dim), nn.ReLU(),
                                  nn.Linear(cond_dim, cond_dim), nn.ReLU())
        self.blocks = nn.ModuleList(FiLMBlock(channels, cond_dim) for _ in range(blocks))

        # policy: score(from, to) = <f(from), g(to)>, plus a small head for under-promotions
        self.head_dim = head_dim
        self.from_head = nn.Conv2d(channels, head_dim, 1)
        self.to_head = nn.Conv2d(channels, head_dim, 1)
        self.promo_head = nn.Conv2d(channels, 9, 1)

        pooled = channels + cond_dim
        self.time_head = nn.Sequential(nn.Linear(pooled, 128), nn.ReLU(), nn.Linear(128, 2))
        self.outcome_head = nn.Sequential(nn.Linear(pooled, 128), nn.ReLU(), nn.Linear(128, 3))

    def forward(self, planes, scalars):
        c = self.cond(scalars)
        x = self.stem(planes)
        for blk in self.blocks:
            x = blk(x, c)
        B = x.shape[0]

        f = self.from_head(x).view(B, self.head_dim, 64)
        t = self.to_head(x).view(B, self.head_dim, 64)
        from_to = torch.einsum("bdf,bdt->bft", f, t).reshape(B, 4096) / math.sqrt(self.head_dim)
        promo = self.promo_head(x).view(B, 9, 64)[:, :, 48:56]          # from-squares on rank 7
        promo = promo.permute(0, 2, 1).reshape(B, 72)                  # (from_file, dir, piece)
        policy = torch.cat([from_to, promo], dim=1)

        pooled = torch.cat([x.mean(dim=(2, 3)), c], dim=1)
        time = self.time_head(pooled)          # [mean, log_std] of log1p(seconds)
        outcome = self.outcome_head(pooled)    # logits for [loss, draw, win]
        return policy, time, outcome


def save(model: HumanNet, path, extra=None):
    torch.save({"config": model.config, "state": model.state_dict(), **(extra or {})}, path)


def load(path, device="cpu") -> HumanNet:
    ckpt = torch.load(path, map_location=device)
    model = HumanNet(**ckpt["config"])
    model.load_state_dict(ckpt["state"])
    return model.to(device).eval()


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
