"""Inference-time search: best-of-N, and best-of-N with early rejection.

Early rejection (our inference-time improvement): run N candidates for k of the T denoising steps, decode the
predicted clean image x0_hat with a tiny VAE (TAESD), score it, keep the `keep` best candidates and finish only those.
Cost in UNet evaluations: N*k + keep*(T-k) instead of N*T. With N=8, keep=2, k=0.4T that is 55% of the compute.
The hypothesis to test: the reward of x0_hat at step k already ranks the candidates well (measure the Spearman
correlation between early and final rewards, scripts/06_benchmark_inference.py --early-correlation).
Works for text-to-image and for img2img (init_image + strength).
"""

from __future__ import annotations

import torch

from .generator import Generator
from .scoring import Scorer


def best_of_n(gen: Generator, scorer: Scorer, prompt: str, true_idx: int, seeds: list[int],
              target_idx: int | None = None, prune_at: float | None = None, keep: int = 1,
              init_image=None, strength: float = 1.0):
    """Returns (records, sample_output). One record per FINAL image (all survivors), with its reward."""
    n = len(seeds)
    steps = gen.cfg.generator.steps
    run_steps = steps if init_image is None else max(1, min(steps, round(strength * steps)))
    tgt = None if target_idx is None else [target_idx] * n
    early_scores: dict[int, float] = {}
    prune = None
    if prune_at:
        k = max(1, int(round(prune_at * run_steps)))

        def prune_fn(x0: torch.Tensor) -> torch.Tensor:
            previews = gen.decode(x0, preview=True)
            s = scorer(previews, [true_idx] * len(previews), tgt)
            for j, r in enumerate(s["reward"].tolist()):
                early_scores[j] = r
            return s["reward"].topk(min(keep, len(previews))).indices.sort().values

        prune = (k, prune_fn)
    out = gen.sample([prompt] * n, seeds, prune=prune, init_images=None if init_image is None else [init_image] * n,
                     strength=strength)
    m = len(out.images)
    s = scorer(out.images, [true_idx] * m, None if target_idx is None else [target_idx] * m)
    records = []
    for j, orig in enumerate(out.keep_idx.tolist()):
        records.append(dict(seed=seeds[orig], reward=s["reward"][j].item(), early_reward=early_scores.get(orig),
                            p_target_true=s["p_target_true"][j].item(), p_oracle_true=s["p_oracle_true"][j].item(),
                            taxo_distance=int(s["taxo_distance"][j])))
    return records, out
