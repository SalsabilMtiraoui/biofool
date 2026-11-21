"""Align data/species.csv with the taxonomy BioCLIP itself uses (the names of the TreeOfLife-10M embeddings).

Our csv was written by hand; BioCLIP's labels come from its own taxonomy. Any difference (a renamed family, a moved
genus) would make the closed-set labels slightly wrong and the rank-level metrics noisy. This script prints every
difference and, with --write, rewrites the csv with BioCLIP's names. Run it once, before the experiments.

Example:  python scripts/sync_taxonomy_tol.py --write
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biofool.cli import parse  # noqa: E402
from biofool.taxonomy import RANKS, Species, save_species  # noqa: E402


def add_args(p):
    p.add_argument("--write", action="store_true")
    p.add_argument("--tol-repo", default="imageomics/TreeOfLife-10M")


def main():
    from huggingface_hub import hf_hub_download

    args, cfg, species = parse(__doc__, add_args)
    with open(hf_hub_download(args.tol_repo, "embeddings/txt_emb_species.json", repo_type="dataset")) as f:
        names = json.load(f)
    tol = {}
    for n in names:
        ranks = n[0]
        if len(ranks) >= 7 and ranks[5] and ranks[6]:
            tol.setdefault(f"{ranks[5]} {ranks[6]}".lower(), (ranks, n[1]))
    fixed, n_diff = [], 0
    for s in species:
        hit = tol.get(s.latin.lower())
        if hit is None:
            print(f"{s.latin:30s} NOT IN TreeOfLife-10M: keep it only if you accept p_true = nan for the open-set target")
            fixed.append(s)
            continue
        ranks, common = hit
        diffs = [f"{r}: {a} -> {b}" for r, a, b in zip(RANKS[:6], s.lineage()[:6], ranks[:6]) if a != b]
        if diffs:
            n_diff += 1
            print(f"{s.latin:30s} " + "; ".join(diffs))
        fixed.append(Species(idx=s.idx, group=s.group, kingdom=ranks[0], phylum=ranks[1], cls=ranks[2], order=ranks[3],
                             family=ranks[4], genus=ranks[5], epithet=ranks[6], common_name=s.common_name,
                             split=s.split))
    print(f"{n_diff} species with differences")
    if args.write:
        save_species(fixed, cfg.species_csv)
        print(f"rewrote {cfg.species_csv}")


if __name__ == "__main__":
    main()
