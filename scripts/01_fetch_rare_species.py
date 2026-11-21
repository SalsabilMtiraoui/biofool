"""Dataset rare: the Rare Species benchmark of the BioCLIP paper (imageomics/rare-species on Hugging Face).

400 IUCN Red List species (all Animalia), 30 photos each, deliberately held out of BioCLIP's training set: the
hardest real-world test of the tool. We sample --species species and --per-species photos of each, write them as a
run directory with its own species.csv (closed-set label set = all 400 species, as in the paper's zero-shot protocol).

Structure used (dataset card, docs/imageomics/RareSpecies_DatasetCard.md in Imageomics/bioclip):
  metadata/rarespecies-catalog.csv: rarespecies_id, eol_content_id, eol_page_id, kingdom ... species, sciName, common
  <taxonomy folder>/<eol_content_id>_<eol_page_id>_eol_full-size-copy.jpg

Example:  python scripts/01_fetch_rare_species.py --out outputs/rare --species 100 --per-species 4
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from biofool.cli import parse  # noqa: E402
from biofool.io import RunWriter, begin_run, finish_run  # noqa: E402
from biofool.taxonomy import Species  # noqa: E402

REPO = "imageomics/rare-species"
GROUPS = {"Aves": "birds", "Mammalia": "mammals", "Insecta": "insects", "Reptilia": "reptiles",
          "Amphibia": "amphibians", "Actinopterygii": "fish", "Chondrichthyes": "fish", "Anthozoa": "corals"}


def add_args(p):
    p.add_argument("--out", required=True)
    p.add_argument("--species", type=int, default=100, help="species sampled among the 400")
    p.add_argument("--per-species", type=int, default=4)


def main():
    from huggingface_hub import hf_hub_download, list_repo_files

    args, cfg, _ = parse(__doc__, add_args)
    if not begin_run(args.out):
        return
    rng = random.Random(cfg.seed)
    cat = pd.read_csv(hf_hub_download(REPO, "metadata/rarespecies-catalog.csv", repo_type="dataset"))
    cat["binomial"] = cat.genus.str.strip() + " " + cat.species.str.strip()
    taxa = cat.drop_duplicates("binomial").sort_values("binomial").reset_index(drop=True)
    species = [Species(idx=i, group=GROUPS.get(r["class"], "other"), kingdom=r.kingdom, phylum=r.phylum,
                       cls=r["class"], order=r.order, family=r.family, genus=r.genus.strip(),
                       epithet=r.species.strip(), common_name=r.common if isinstance(r.common, str) else r.binomial,
                       split="B") for i, r in taxa.iterrows()]
    idx = {s.latin: s.idx for s in species}
    files = [f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith(".jpg")]
    by_key = {Path(f).name.split("_eol")[0]: f for f in files}  # "<content_id>_<page_id>" -> path
    writer = RunWriter(args.out, species)
    for binomial in rng.sample(sorted(idx), min(args.species, len(idx))):
        rows = cat[cat.binomial == binomial]
        n = 0
        for r in rows.sample(frac=1, random_state=cfg.seed).itertuples():
            path = by_key.get(f"{r.eol_content_id}_{r.eol_page_id}")
            if path is None or n >= args.per_species:
                continue
            im = Image.open(hf_hub_download(REPO, path, repo_type="dataset")).convert("RGB")
            s = species[idx[binomial]]
            writer.add(im, species_idx=s.idx, latin=s.latin, group=s.group, split="B", dataset="rare", family="real",
                       modifier="", prompt="", target_idx=None, seed=None, method="real", source=path,
                       rarespecies_id=r.rarespecies_id)
            n += 1
        print(f"{binomial:32s} {n} photos")
    writer.close()
    finish_run(args.out)


if __name__ == "__main__":
    main()
