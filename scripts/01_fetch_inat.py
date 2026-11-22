"""Real photos from iNaturalist (research grade, CC0 / CC-BY / CC-BY-NC only; the source URL is kept per image).

Two datasets are built with this script:
  inat80     the 80 species of data/species.csv (taxonomy from the csv)
             python scripts/01_fetch_inat.py --out outputs/inat80 --per-species 10
  invasive   40 invasive alien species of the EU Union list (data/invasive_species.txt); their taxonomy and common
             names are resolved with the iNaturalist taxa API and written to the run's species.csv
             python scripts/01_fetch_inat.py --out outputs/invasive --dataset invasive \
                 --species-list data/invasive_species.txt --per-species 8

The photos give BioCLIP's accuracy on real data (the reference) and are the starting images of img2img attacks.
"""

import http.client
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image  # noqa: E402

from biofool.cli import parse  # noqa: E402
from biofool.io import RunWriter, finish_run, read_meta  # noqa: E402
from biofool.taxonomy import Species, load_species  # noqa: E402

API = "https://api.inaturalist.org/v1"
HEADERS = {"User-Agent": "biofool-course-project"}
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]
GROUPS = {"Insecta": "insects", "Aves": "birds", "Plantae": "plants", "Fungi": "fungi", "Mammalia": "mammals",
          "Reptilia": "reptiles", "Amphibia": "amphibians", "Mollusca": "molluscs", "Actinopterygii": "fish",
          "Arachnida": "arachnids", "Animalia": "other animals"}


def add_args(p):
    p.add_argument("--out", required=True)
    p.add_argument("--per-species", type=int, default=10)
    p.add_argument("--dataset", default="inat80")
    p.add_argument("--species-list", default=None, help="text file with one binomial per line (default: species csv)")


def get(url, raw=False):
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            time.sleep(1.0)  # iNaturalist asks for at most ~1 request per second
            return data if raw else json.loads(data)
        except (OSError, http.client.HTTPException):  # includes IncompleteRead: retry
            time.sleep(2 ** attempt)
    raise RuntimeError(url)


def resolve(binomial: str, idx: int):
    """Species with its full taxonomy from iNaturalist, or None."""
    res = get(f"{API}/taxa?" + urllib.parse.urlencode(dict(q=binomial, rank="species", per_page=5)))["results"]
    hit = next((t for t in res if t["name"].lower() == binomial.lower()), None)
    if hit is None:
        return None
    full = get(f"{API}/taxa/{hit['id']}")["results"][0]
    by_rank = {a["rank"]: a["name"] for a in full.get("ancestors", [])}
    if not all(r in by_rank for r in RANKS):
        return None
    genus, epithet = hit["name"].split(" ")[:2]
    return Species(idx=idx, group=GROUPS.get(hit.get("iconic_taxon_name"), "other"), kingdom=by_rank["kingdom"],
                   phylum=by_rank["phylum"], cls=by_rank["class"], order=by_rank["order"], family=by_rank["family"],
                   genus=genus, epithet=epithet, common_name=hit.get("preferred_common_name") or hit["name"],
                   split="B")


def main():
    args, cfg, species = parse(__doc__, add_args)
    out = Path(args.out)
    if (out / "DONE").exists():
        print(f"{out} is complete: skipping (delete {out / 'DONE'} to run it again)")
        return
    done = {}
    if (out / "meta.jsonl").exists():  # resume: keep the photos already downloaded, never download one twice
        for m in read_meta(out):
            done.setdefault(m["latin"], set()).add(m["source"])
    if (out / "species.csv").exists():
        species = load_species(out / "species.csv")
    elif args.species_list:
        names = [l.strip() for l in open(args.species_list) if l.strip() and not l.startswith("#")]
        species = []
        for name in names:
            s = resolve(name, len(species))
            print(f"{name:32s} {'-> ' + ' > '.join(s.lineage()) if s else 'NOT FOUND, skipped'}")
            if s:
                species.append(s)
    writer = RunWriter(args.out, species)
    for s in species:
        have = done.get(s.latin, set())
        if len(have) >= args.per_species:
            continue
        q = urllib.parse.urlencode(dict(taxon_name=s.latin, quality_grade="research", photos="true",
                                        photo_license="cc0,cc-by,cc-by-nc", per_page=args.per_species * 2,
                                        order_by="votes"))
        for obs in get(f"{API}/observations?{q}")["results"]:
            source = f"https://www.inaturalist.org/observations/{obs['id']}"
            if len(have) >= args.per_species or not obs.get("photos") or source in have:
                continue
            url = obs["photos"][0]["url"].replace("square", "medium")
            try:
                im = Image.open(io.BytesIO(get(url, raw=True))).convert("RGB")
            except (OSError, RuntimeError):
                continue
            writer.add(im, species_idx=s.idx, latin=s.latin, group=s.group, split=s.split, dataset=args.dataset,
                       family="real", modifier="", prompt="", target_idx=None, seed=None, method="real",
                       source=source, license=obs["photos"][0].get("license_code"))
            have.add(source)
        print(f"{s.latin:28s} {len(have)} photos")
    writer.close()
    if writer.n == 0:
        sys.exit("no photo downloaded: check the internet connection (api.inaturalist.org)")
    finish_run(out)


if __name__ == "__main__":
    main()