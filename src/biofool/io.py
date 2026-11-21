"""Run directories.

    outputs/<run>/images/<image_id>.png
    outputs/<run>/meta.jsonl       one JSON line per image (species_idx, dataset, method, prompt, seed, ...)
    outputs/<run>/species.csv      the label set of the run: species_idx indexes into it
    outputs/<run>/scores.csv       written by scripts/03_score.py

Datasets (the `dataset` column):
    synth80        text-to-image images of the 80 species of data/species.csv
    inat80         real iNaturalist photos of the same 80 species (and their img2img attacks)
    rare           real photos of the Rare Species benchmark (Imageomics, IUCN species held out from BioCLIP's
                   training) and their img2img attacks
    invasive       real iNaturalist photos of 40 invasive alien species (EU Union list) and their img2img attacks
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from .taxonomy import Species, load_species, save_species


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class RunWriter:
    def __init__(self, run_dir: str | Path, species: list[Species] | None = None):
        self.dir = Path(run_dir)
        (self.dir / "images").mkdir(parents=True, exist_ok=True)
        if species is not None:
            save_species(species, self.dir / "species.csv")
        meta = self.dir / "meta.jsonl"
        self.n = sum(1 for line in open(meta) if line.strip()) if meta.exists() else 0
        self.meta = open(meta, "a")

    def add(self, image, **meta) -> str:
        image_id = f"{self.n:06d}"
        image.save(self.dir / "images" / f"{image_id}.png")
        self.meta.write(json.dumps({"image_id": image_id, **meta}) + "\n")
        self.meta.flush()
        self.n += 1
        return image_id

    def close(self):
        self.meta.close()


def read_meta(run_dir: str | Path) -> list[dict]:
    with open(Path(run_dir) / "meta.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_species(run_dir: str | Path, default: str | Path = "data/species.csv") -> list[Species]:
    path = Path(run_dir) / "species.csv"
    return load_species(path if path.exists() else default)


def begin_run(run_dir: str | Path) -> bool:
    """Resumable steps (Colab sessions get cut). Returns False if the run is already complete (a DONE file exists).
    A run directory WITHOUT DONE is the leftover of an interrupted run: it is deleted and the run starts again."""
    run_dir = Path(run_dir)
    if (run_dir / "DONE").exists():
        print(f"{run_dir} is complete: skipping (delete {run_dir / 'DONE'} to run it again)")
        return False
    if run_dir.exists():
        print(f"{run_dir} was interrupted: starting it again")
        shutil.rmtree(run_dir)
    return True


def finish_run(run_dir: str | Path):
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "DONE").write_text("ok\n")
