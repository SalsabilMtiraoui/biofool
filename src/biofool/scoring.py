"""Scores generated images with the target model and the reward oracle, and turns them into rewards."""

from __future__ import annotations

import torch
from PIL import Image

from .reward import attack_reward
from .taxonomy import Species


class Scorer:
    def __init__(self, cfg, species: list[Species], target=None, oracle=None):
        from .classifiers import build_classifier

        self.cfg, self.species = cfg, species
        self.target = target or build_classifier(cfg.roles.target, cfg, species)
        self.oracle = oracle or build_classifier(cfg.roles.reward_oracle, cfg, species)

    def __call__(self, images: list[Image.Image], true_idx, target_idx=None) -> dict[str, torch.Tensor]:
        true_idx = [int(i) for i in true_idx]
        t = self.target.predict(images, true_idx, target_idx)
        o = self.oracle.predict(images, true_idx)
        r = self.cfg.reward
        dist = 7 - t["shared_ranks"]
        reward = attack_reward(t["p_true"], o["p_true"], t["p_target"] if r.targeted and target_idx is not None
                               else None, mode=r.mode, tau=r.tau, taxo_distance=dist, taxo_bonus=r.taxo_bonus)
        return dict(reward=reward, p_target_true=t["p_true"], p_oracle_true=o["p_true"], target_top1=t["top1"],
                    target_shared_ranks=t["shared_ranks"], taxo_distance=dist)
