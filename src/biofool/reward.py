"""The attack reward.

For one image of true species y (and, for a targeted attack, target species y*):
    p_T(y)   probability that the TARGET (BioCLIP, open set) gives to the true species
    p_T(y*)  ... to the target species
    p_O(y)   fidelity: probability that the REWARD ORACLE (BioCLIP-2 by default) gives to the true species

    fool       = 1 - p_T(y)               (untargeted)       or   p_T(y*)   (targeted)
    gated      = fool * p_O(y)            default: fooling only counts if the oracle still sees species y
    hard       = fool * [p_O(y) >= tau]
    margin     = log p_O(y) - log p_T(y)  (clipped to [-10, 10])
    fool_only  = fool                     ablation: no fidelity term -> reward hacking

An optional bonus rewards errors that are far away in the tree of life (taxonomic distance / 7).
"""

from __future__ import annotations

import torch


def attack_reward(p_target_true: torch.Tensor, p_oracle_true: torch.Tensor, p_target_target: torch.Tensor | None = None,
                  mode: str = "gated", tau: float = 0.5, taxo_distance: torch.Tensor | None = None,
                  taxo_bonus: float = 0.0) -> torch.Tensor:
    p_true = torch.nan_to_num(p_target_true.float(), nan=0.0)   # species unknown to the target: counts as fooled
    oracle = torch.nan_to_num(p_oracle_true.float(), nan=0.0)
    fool = torch.nan_to_num(p_target_target.float(), nan=0.0) if p_target_target is not None else 1 - p_true
    if mode == "gated":
        r = fool * oracle
    elif mode == "hard":
        r = fool * (oracle >= tau).float()
    elif mode == "margin":
        r = (oracle.clamp_min(1e-6).log() - p_true.clamp_min(1e-6).log()).clamp(-10, 10)
    elif mode == "fool_only":
        r = fool
    else:
        raise ValueError(f"unknown reward mode {mode!r}")
    if taxo_bonus and taxo_distance is not None:
        r = r + taxo_bonus * (taxo_distance.float() / 7) * (oracle >= tau).float()
    return r


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """GRPO / DDPO per-prompt normalisation. rewards: [P, G] -> advantages [P, G] (population std, as in Lab 4)."""
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    return (rewards - mean) / (std + eps)
