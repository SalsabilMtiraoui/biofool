"""RAFT: select the best attacks of a scored best-of-N run and fine-tune a LoRA on them.

Example:
  python scripts/04_train_raft.py --run outputs/bon_A --out checkpoints/raft_r1
  python scripts/02_generate.py --config configs/experiments/templates.yaml --split B --tag raft_r1 \
      --out outputs/raft_r1_B --set generator.lora_path=checkpoints/raft_r1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from biofool.cli import parse  # noqa: E402
from biofool.generator import Generator  # noqa: E402
from biofool.raft import select_examples, train_raft  # noqa: E402


def add_args(p):
    p.add_argument("--run", required=True, help="a scored run (best_of_n on split A)")
    p.add_argument("--out", required=True)


def main():
    args, cfg, species = parse(__doc__, add_args)
    if (Path(args.out) / "pytorch_lora_weights.safetensors").exists():
        print(f"{args.out} already trained: skipping")
        return
    df = pd.read_csv(Path(args.run) / "scores.csv", dtype={"image_id": str})
    df = df[df.split == "A"]  # never train on the held-out species
    c = cfg.raft
    ex = select_examples(df, c.select, c.top_k, c.min_fidelity)
    print(f"selected {len(ex)} / {len(df)} images from {ex.species_idx.nunique()} species")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    ex.to_csv(Path(args.out) / "examples.csv", index=False)
    gen = Generator(cfg)
    train_raft(gen, ex, Path(args.run) / "images", Path(args.out), c.lora_rank, c.lr, c.steps, c.batch_size, cfg.seed,
               gradient_checkpointing=c.get("gradient_checkpointing", False))
    print(f"LoRA saved to {args.out}")


if __name__ == "__main__":
    main()
