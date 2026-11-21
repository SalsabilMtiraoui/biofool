"""DDPO / GRPO fine-tuning of a LoRA to fool BioCLIP while staying faithful (reward oracle).

Example:  python scripts/05_train_ddpo.py --config configs/experiments/ddpo.yaml --out checkpoints/ddpo
Ablation: python scripts/05_train_ddpo.py --config configs/experiments/ddpo_no_fidelity.yaml --out checkpoints/ddpo_nofid
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biofool.cli import parse  # noqa: E402
from biofool.ddpo import train_ddpo  # noqa: E402
from biofool.generator import Generator  # noqa: E402
from biofool.scoring import Scorer  # noqa: E402


def add_args(p):
    p.add_argument("--out", required=True)


def main():
    args, cfg, species = parse(__doc__, add_args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    gen = Generator(cfg)
    scorer = Scorer(cfg, species)
    train_ddpo(gen, scorer, species, cfg, out)


if __name__ == "__main__":
    main()
