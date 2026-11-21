"""Evaluation metrics on a table of scored images (one row per image, written by scripts/03_score.py).

Definitions (the report must state them exactly like this):
  faithful        eval oracle p(true species) >= threshold            (VLM judge by default, NOT the reward oracle)
  fooled          target top-1 != true species
  fool_rate       P(fooled)                                  - over all images (contaminated by bad generations)
  ASR             P(fooled | faithful)                       - attack success rate, the headline number
  effective_ASR   P(fooled and faithful)                     - what an attacker actually gets per image generated
  ASR@rank        P(wrong at that rank | faithful)           - e.g. ASR@family: even the family is wrong
  taxo_dist       mean taxonomic distance of the errors      - 1 = same genus ... 7 = different kingdom
  confident_err   P(fooled with top-1 probability > 0.5 | faithful): errors the tool would show with confidence
  entropy         mean entropy of the target's prediction (uncertainty, Lab 3)
"fooled" is defined from the ranks shared with the truth (shared_ranks < 7), so it works for the open-set target
(top-1 among ~450k TreeOfLife species) as well as for closed-set classifiers.
Confidence intervals: cluster bootstrap over species (images of the same species are not independent).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .taxonomy import RANKS


def add_outcomes(df: pd.DataFrame, target: str, eval_oracle: str, threshold: float) -> pd.DataFrame:
    """Adds the boolean columns `faithful`, `fooled` and `wrong_<rank>` for a given target model."""
    df = df.copy()
    df["faithful"] = df[f"{eval_oracle}_p_true"] >= threshold
    shared = df[f"{target}_shared_ranks"]
    df["fooled"] = shared < len(RANKS)
    df["confident_error"] = df.fooled & (df[f"{target}_top1_prob"] > 0.5) if f"{target}_top1_prob" in df else False
    df["entropy"] = df[f"{target}_entropy"] if f"{target}_entropy" in df else np.nan
    for i, rank in enumerate(RANKS):
        df[f"wrong_{rank}"] = shared < i + 1
    df["taxo_dist"] = len(RANKS) - shared
    return df


def summary(df: pd.DataFrame) -> dict[str, float]:
    faithful = df[df.faithful]
    out = {
        "n": len(df),
        "faithful_rate": df.faithful.mean(),
        "fool_rate": df.fooled.mean(),
        "ASR": faithful.fooled.mean() if len(faithful) else np.nan,
        "effective_ASR": (df.fooled & df.faithful).mean(),
        "taxo_dist": faithful[faithful.fooled].taxo_dist.mean() if faithful.fooled.any() else np.nan,
        "confident_err": faithful.confident_error.mean() if len(faithful) else np.nan,
        "entropy": df.entropy.mean(),
    }
    for rank in ["genus", "family", "order", "class"]:
        out[f"ASR@{rank}"] = faithful[f"wrong_{rank}"].mean() if len(faithful) else np.nan
    return out


def bootstrap_ci(df: pd.DataFrame, stat: str, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """(low, high) percentile CI of summary(df)[stat], resampling species with replacement."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("latin")]
    vals = []
    for _ in range(n_boot):
        sample = pd.concat([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        v = summary(sample)[stat]
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return np.nan, np.nan
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def summary_table(df: pd.DataFrame, by: list[str], stats=("faithful_rate", "ASR", "effective_ASR", "ASR@family",
                                                          "taxo_dist"), ci: bool = True, n_boot: int = 500):
    rows = []
    for key, g in df.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        s = summary(g)
        row = dict(zip(by, key)) | {"n": s["n"]}
        for st in stats:
            row[st] = s[st]
            if ci and st in ("ASR", "effective_ASR"):
                row[f"{st}_lo"], row[f"{st}_hi"] = bootstrap_ci(g, st, n_boot=n_boot)
        rows.append(row)
    return pd.DataFrame(rows)


def spearman(a, b) -> float:
    a, b = pd.Series(a).rank(), pd.Series(b).rank()
    return float(np.corrcoef(a, b)[0, 1])


def lpips_diversity(images, device="cuda", max_pairs: int = 50, seed: int = 0) -> float:
    """Mean LPIPS distance between random pairs of images of the same set (higher = more diverse).
    Detects mode collapse of the fine-tuned generator. Needs `pip install lpips`."""
    import lpips
    import torch
    import torchvision.transforms.functional as TF

    if len(images) < 2:
        return float("nan")
    net = lpips.LPIPS(net="alex", verbose=False).to(device)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(min(max_pairs, len(images) * (len(images) - 1) // 2)):
        i, j = rng.choice(len(images), 2, replace=False)
        x = [TF.to_tensor(images[k].convert("RGB").resize((256, 256))) * 2 - 1 for k in (i, j)]
        with torch.no_grad():
            d.append(net(x[0][None].to(device), x[1][None].to(device)).item())
    return float(np.mean(d))
