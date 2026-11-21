"""Check data/species.csv against the GBIF backbone taxonomy (needs internet). Prints every disagreement.

Example:  python scripts/verify_taxonomy.py
"""

import csv
import json
import urllib.parse
import urllib.request

RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]


def main(path="data/species.csv"):
    bad = 0
    for row in csv.DictReader(open(path)):
        name = f"{row['genus']} {row['epithet']}"
        url = "https://api.gbif.org/v1/species/match?" + urllib.parse.urlencode({"name": name, "strict": "true"})
        g = json.load(urllib.request.urlopen(url, timeout=30))
        diffs = [f"{r}: csv={row[r]} gbif={g.get(r)}" for r in RANKS if g.get(r) and g.get(r) != row[r]]
        if g.get("matchType") != "EXACT" or diffs:
            bad += 1
            print(f"{name:30s} match={g.get('matchType')} status={g.get('status')}  " + "; ".join(diffs))
    print(f"{bad} species with differences (BioCLIP uses its own taxonomy: small differences at family level "
          f"are expected; check them with BioCLIP's TreeOfLife names if in doubt)")


if __name__ == "__main__":
    main()
