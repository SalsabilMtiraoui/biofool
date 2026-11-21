"""Species list, label strings and taxonomic distance."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]


@dataclass(frozen=True)
class Species:
    idx: int
    group: str
    kingdom: str
    phylum: str
    cls: str
    order: str
    family: str
    genus: str
    epithet: str
    common_name: str
    split: str

    @property
    def latin(self) -> str:
        return f"{self.genus} {self.epithet}"

    @property
    def slug(self) -> str:
        return f"{self.genus}_{self.epithet}".lower()

    def lineage(self) -> tuple[str, ...]:
        """The 7 ranks, kingdom to species. The species rank is the binomial, so that two
        species of different genera never 'match' on the epithet alone."""
        return (self.kingdom, self.phylum, self.cls, self.order, self.family, self.genus, self.latin)

    def label(self, fmt: str = "taxonomic_common") -> str:
        """Class name given to a zero-shot classifier (the classifier wraps it in prompt templates).

        taxonomic_common  full taxonomy plus common name: BioCLIP's "mixed" text type (best for BioCLIP / BioCLIP-2)
        taxonomic         full taxonomy only
        scientific        binomial only
        common            common name only (best for generic CLIP models)
        """
        taxo = " ".join(self.lineage()[:-1] + (self.epithet,))
        if fmt == "taxonomic_common":
            return f"{taxo} with common name {self.common_name}"
        if fmt == "taxonomic":
            return taxo
        if fmt == "scientific":
            return self.latin
        if fmt == "common":
            return self.common_name
        raise ValueError(f"unknown label format {fmt!r}")


FIELDS = ["group", "kingdom", "phylum", "class", "order", "family", "genus", "epithet", "common_name", "split"]


def load_species(path: str | Path) -> list[Species]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        Species(idx=i, group=r["group"], kingdom=r["kingdom"], phylum=r["phylum"], cls=r["class"],
                order=r["order"], family=r["family"], genus=r["genus"], epithet=r["epithet"],
                common_name=r.get("common_name") or f"{r['genus']} {r['epithet']}", split=r.get("split") or "B")
        for i, r in enumerate(rows)
    ]


def save_species(species: list[Species], path: str | Path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for s in species:
            w.writerow([s.group, s.kingdom, s.phylum, s.cls, s.order, s.family, s.genus, s.epithet, s.common_name,
                        s.split])


def load_pairs(path: str | Path, species: list[Species]) -> dict[int, list[int]]:
    """Curated confusable pairs: source species index -> list of target species indices."""
    by_latin = {s.latin: s.idx for s in species}
    pairs: dict[int, list[int]] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            pairs.setdefault(by_latin[r["source_latin"]], []).append(by_latin[r["target_latin"]])
    return pairs


def shared_prefix(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    """Number of leading ranks two lineages (kingdom, ..., genus, binomial) share, compared case-insensitively."""
    n = 0
    for x, y in zip(a, b):
        if x.strip().lower() != y.strip().lower():
            break
        n += 1
    return n


def shared_ranks(a: Species, b: Species) -> int:
    """Number of leading ranks (from kingdom) that a and b share: 7 = same species, 0 = different kingdoms."""
    return shared_prefix(a.lineage(), b.lineage())


def taxonomic_distance(a: Species, b: Species) -> int:
    """0 = same species, 1 = same genus, 2 = same family, ..., 7 = different kingdom."""
    return len(RANKS) - shared_ranks(a, b)


def wrong_at_rank(true: Species, pred: Species, rank: str) -> bool:
    """True if the prediction is wrong at `rank` (e.g. 'family': the predicted family differs)."""
    return shared_ranks(true, pred) < RANKS.index(rank) + 1
