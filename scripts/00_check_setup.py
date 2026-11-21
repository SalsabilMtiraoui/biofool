"""Check the installation: GPU, model downloads, one image generated and scored. Run it first.

Example:  python scripts/00_check_setup.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from biofool.cli import parse  # noqa: E402


def add_args(p):
    p.add_argument("--skip-vlm", action="store_true")


def main():
    args, cfg, species = parse(__doc__, add_args)
    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name}, {p.total_memory / 2**30:.0f} GiB")
    print(f"{len(species)} species, groups: {sorted({s.group for s in species})}")

    from biofool.generator import Generator
    gen = Generator(cfg)
    s = species[0]
    out = gen.sample([f"a high quality nature photograph of a {s.common_name} ({s.latin})"], [0])
    Path("outputs").mkdir(exist_ok=True)
    out.images[0].save("outputs/check_setup.png")
    print(f"generated outputs/check_setup.png in {out.seconds:.1f}s")

    from biofool.classifiers import build_classifier
    names = [cfg.roles.target, cfg.roles.closed_target, cfg.roles.reward_oracle, *cfg.roles.transfer]
    if not args.skip_vlm:
        names.append(cfg.roles.eval_oracle)
    for name in dict.fromkeys(names):
        clf = build_classifier(name, cfg, species)
        pred = clf.predict(out.images, [s.idx])
        if clf.kind == "judge":
            print(f"{name:12s} p(yes) = {pred['p_true'][0].item():.3f}")
        else:
            print(f"{name:12s} top-1 {pred['top1_name'][0]:28s} p={pred['top1_prob'][0].item():.3f}   "
                  f"p(true)={pred['p_true'][0].item():.3f}   shared ranks={int(pred['shared_ranks'][0])}")
        del clf
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("setup OK")


if __name__ == "__main__":
    main()
