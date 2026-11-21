"""Shared command-line handling: every script takes --config and optional --set key=value overrides."""

from __future__ import annotations

import argparse

from .config import load_config
from .io import seed_everything
from .taxonomy import load_species


def parse(description: str, add_args=None):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/default.yaml")
    if add_args:
        add_args(p)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="config overrides, e.g. --set generator.steps=10 device=cuda:1")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg.seed)
    species = load_species(cfg.species_csv)
    return args, cfg, species


def select_species(species, split: str = "all", groups: str | None = None, limit: int | None = None):
    out = [s for s in species if split == "all" or s.split == split]
    if groups:
        out = [s for s in out if s.group in groups.split(",")]
    return out[:limit] if limit else out
