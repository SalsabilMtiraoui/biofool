"""Remove duplicated or broken rows from run directories (meta.jsonl and scores.csv, kept in sync).

Example:  python scripts/clean_run.py outputs/i2i_invasive
"""

import json
import sys
from pathlib import Path

import pandas as pd


def key(m: dict):
    if m.get("method") == "real":
        return ("real", m.get("source"))
    return (m.get("prompt"), m.get("seed"), m.get("strength"), m.get("source_image_id"), m.get("job"))


def clean(run: Path):
    meta_path = run / "meta.jsonl"
    rows = [json.loads(line) for line in open(meta_path, encoding="utf-8") if line.strip()]
    seen, keep = set(), []
    for m in rows:
        k = key(m)
        if k in seen or not (run / "images" / f"{m['image_id']}.png").exists():
            continue
        src = m.get("source_run")
        if src and not (Path(src) / "images" / f"{m['source_image_id']}.png").exists():
            continue
        seen.add(k)
        keep.append(m)
    meta_path.write_text("".join(json.dumps(m) + "\n" for m in keep), encoding="utf-8")
    ids = {m["image_id"] for m in keep}
    scores = run / "scores.csv"
    if scores.exists():
        df = pd.read_csv(scores, dtype={"image_id": str, "source_image_id": str})
        df = df[df.image_id.isin(ids)].drop_duplicates("image_id")
        df.to_csv(scores, index=False)
    print(f"{run}: {len(rows)} -> {len(keep)} rows")


if __name__ == "__main__":
    for r in sys.argv[1:]:
        clean(Path(r))