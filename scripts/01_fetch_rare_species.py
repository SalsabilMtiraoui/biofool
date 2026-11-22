"""Dataset rare: the Rare Species benchmark of the BioCLIP paper (imageomics/rare-species on Hugging Face).

400 IUCN Red List species (all Animalia), 30 photos each, deliberately held out of BioCLIP's training set: the
hardest real-world test of the tool. We sample --species species and --per-species photos of each, write them as a
run directory with its own species.csv (closed-set label set = all 400 species, as in the paper's zero-shot protocol).

Images are found from the folder of each species in the repository (<kingdom>-<phylum>-...-<genus>-<species>/), so the
script does not depend on the exact form of the ids in the catalog. Resumable: species already downloaded are kept.

Example:  python scripts/01_fetch_rare_species.py --out outputs/rare --species 100 --per-species 4
"""

import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from biofool.cli import parse  # noqa: E402
from biofool.io import RunWriter, finish_run, read_meta  # noqa: E402
from biofool.taxonomy import Species  # noqa: E402

REPO = "imageomics/rare-species"
GROUPS = {"Aves": "birds", "Mammalia": "mammals", "Insecta": "insects", "Reptilia": "reptiles",
          "Amphibia": "amphibians", "Actinopterygii": "fish", "Chondrichthyes": "fish", "Anthozoa": "corals"}


def add_args(p):
    p.add_argument("--out", required=True)
    p.add_argument("--species", type=int, default=100, help="species sampled among the 400")
    p.add_argument("--per-species", type=int, default=4)


def norm(text: str) -> str:
    return re.sub(r"[^a-z]+", " ", str(text).lower()).strip()


def main():
    from huggingface_hub import hf_hub_download, list_repo_files

    args, cfg, _ = parse(__doc__, add_args)
    out = Path(args.out)
    if (out / "DONE").exists():
        print(f"{out} is complete: skipping (delete {out / 'DONE'} to run it again)")
        return
    rng = random.Random(cfg.seed)
    files = list_repo_files(REPO, repo_type="dataset")
    catalog = next(f for f in files if f.endswith("rarespecies-catalog.csv"))
    cat = pd.read_csv(hf_hub_download(REPO, catalog, repo_type="dataset"), dtype=str)
    cat["binomial"] = cat.genus.str.strip() + " " + cat.species.str.strip()
    taxa = cat.drop_duplicates("binomial").sort_values("binomial").reset_index(drop=True)
    species = [Species(idx=i, group=GROUPS.get(r["class"], "other"), kingdom=r.kingdom, phylum=r.phylum,
                       cls=r["class"], order=r.order, family=r.family, genus=r.genus.strip(),
                       epithet=r.species.strip(), common_name=r.common if isinstance(r.common, str) else r.binomial,
                       split="B") for i, r in taxa.iterrows()]
    idx = {s.latin: s.idx for s in species}

    # images grouped by species folder; the folder name ends with "<genus>-<species>" (any separator)
    by_binomial: dict[str, list[str]] = {}
    for f in files:
        if f.lower().endswith(".jpg"):
            folder = norm(Path(f).parent.name)
            for binomial in idx:
                if folder.endswith(norm(binomial)):
                    by_binomial.setdefault(binomial, []).append(f)
                    break
    print(f"{len(files)} files in the repository, images found for {len(by_binomial)} / {len(idx)} species")
    if not by_binomial:
        sys.exit("no image matched a species: send the first lines of this log, the repository layout has changed")

    done = {}
    if (out / "meta.jsonl").exists():  # resume
        for m in read_meta(out):
            done.setdefault(m["latin"], set()).add(m["source"])
    writer = RunWriter(out, species)
    chosen = rng.sample(sorted(by_binomial), min(args.species, len(by_binomial)))
    for binomial in chosen:
        paths = sorted(by_binomial[binomial])
        rng.shuffle(paths)
        have = done.get(binomial, set())
        s = species[idx[binomial]]
        for path in paths:
            if len(have) >= args.per_species:
                break
            if path in have:
                continue
            im = Image.open(hf_hub_download(REPO, path, repo_type="dataset")).convert("RGB")
            writer.add(im, species_idx=s.idx, latin=s.latin, group=s.group, split="B", dataset="rare", family="real",
                       modifier="", prompt="", target_idx=None, seed=None, method="real", source=path)
            have.add(path)
        print(f"{binomial:32s} {len(have)} photos")
    writer.close()
    finish_run(out)


if __name__ == "__main__":
    main()