"""Score every image of one or more runs with the target, the oracles and the transfer models.

Writes <run>/scores.csv: the metadata of each image, plus for every classifier c
    c_p_true, c_p_target, c_top1 (-1: not one of the run's species), c_top1_name, c_top1_prob,
    c_shared_ranks (ranks shared by the top-1 and the truth: 7 = correct species), c_entropy
(the VLM judge only has c_p_true = p("yes")). And the columns used by RAFT and the analysis:
    target_fooled, reward_oracle_p_true, reward.
Each run is scored against its own label set (<run>/species.csv). One model is in memory at a time.
Incremental: a classifier whose columns are already in <run>/scores.csv is skipped (use --force to redo), and
scores.csv is saved after every classifier, so an interrupted scoring resumes where it stopped.

Example:  python scripts/03_score.py --runs outputs/baseline outputs/templates_B
"""

import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from biofool.classifiers import build_classifier  # noqa: E402
from biofool.cli import parse  # noqa: E402
from biofool.io import read_meta, run_species  # noqa: E402
from biofool.reward import attack_reward  # noqa: E402


def add_args(p):
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--classifiers", nargs="*", default=None,
                   help="default: target, reward oracle, eval oracle and transfer models of the config")
    p.add_argument("--chunk", type=int, default=256, help="images loaded in memory at once")
    p.add_argument("--force", action="store_true", help="recompute classifiers already in scores.csv")


def main():
    args, cfg, _ = parse(__doc__, add_args)
    r = cfg.roles
    names = args.classifiers or list(dict.fromkeys([r.target, r.reward_oracle, r.eval_oracle, *r.transfer]))
    runs = {}
    for run in args.runs:
        if not (Path(run) / "meta.jsonl").exists():
            print(f"{run}: no images, skipped")
            continue
        meta = pd.DataFrame(read_meta(run))
        old = Path(run) / "scores.csv"
        df = meta
        if old.exists() and not args.force:
            prev = pd.read_csv(old, dtype={"image_id": str})
            if len(prev) == len(meta):
                df = prev
        runs[run] = (df, run_species(run, cfg.species_csv))
    for name in names:
        clf = None
        for run, (df, species) in runs.items():
            if f"{name}_p_true" in df and not args.force:
                continue
            if clf is None:
                clf = build_classifier(name, cfg, species)
            else:
                clf.set_species(species)
            cols: dict[str, list] = {}
            tgt_all = df.target_idx.tolist() if "target_idx" in df else [None] * len(df)
            for i in range(0, len(df), args.chunk):
                ims = [Image.open(Path(run) / "images" / f"{iid}.png") for iid in df.image_id[i:i + args.chunk]]
                idx = df.species_idx[i:i + args.chunk].tolist()
                tgt = tgt_all[i:i + args.chunk]
                tgt = None if all(pd.isna(t) for t in tgt) else [int(t) if pd.notna(t) else 0 for t in tgt]
                pred = clf.predict(ims, idx, tgt)
                for k, v in pred.items():
                    cols.setdefault(k, []).extend(v if isinstance(v, list) else v.tolist())
            for k, v in cols.items():
                df[f"{name}_{k}"] = v
            df.to_csv(Path(run) / "scores.csv", index=False)
            print(f"{name:12s} {run}: done")
        if clf is not None and getattr(clf, "n_calls", 0):
            print(f"{name}: {clf.n_calls} API calls (the others came from the cache)")
        del clf
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for run, (df, _) in runs.items():
        if f"{r.target}_p_true" in df and f"{r.reward_oracle}_p_true" in df:
            df["reward_oracle_p_true"] = df[f"{r.reward_oracle}_p_true"]
            df["target_fooled"] = df[f"{r.target}_shared_ranks"] < 7
            targeted = cfg.reward.targeted and "target_idx" in df and df.target_idx.notna().any()
            df["reward"] = attack_reward(torch.tensor(df[f"{r.target}_p_true"].values),
                                         torch.tensor(df.reward_oracle_p_true.values),
                                         torch.tensor(df[f"{r.target}_p_target"].values) if targeted else None,
                                         mode=cfg.reward.mode, tau=cfg.reward.tau).numpy()
        df.to_csv(Path(run) / "scores.csv", index=False)
        print(f"wrote {Path(run) / 'scores.csv'} ({len(df)} images)")


if __name__ == "__main__":
    main()
