"""Which pixels drive BioCLIP's decision on the attacks? (the attention-inspection idea of Lab 3, Exercise 4)

For K faithful-and-fooled images per method, SmoothGrad saliency of the closed-set BioCLIP logit of
(a) the true species and (b) the predicted species, w.r.t. the input pixels. A recurring pattern (the saliency of the
wrong class sits on the background, the light or the context added by the modifier) is evidence for a shortcut.
Writes figures/saliency_<method>.png.

Example:  python scripts/11_saliency.py --runs outputs/templates_B outputs/i2i_inat --k 6
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from biofool.classifiers import build_classifier  # noqa: E402
from biofool.cli import parse  # noqa: E402
from biofool.io import run_species  # noqa: E402


def add_args(p):
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default="results")
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--samples", type=int, default=16)


def smoothgrad(clf, x: torch.Tensor, cls: int, n: int, sigma: float) -> torch.Tensor:
    grads = torch.zeros_like(x)
    for _ in range(n):
        xi = (x + sigma * torch.randn_like(x)).requires_grad_(True)
        f = torch.nn.functional.normalize(clf.model.encode_image(xi[None]).float(), dim=-1)
        (clf.logit_scale * f @ clf.text_features[cls]).sum().backward()
        grads += xi.grad.abs()
    return grads.mean(0).cpu()  # [H, W], averaged over channels


def main():
    args, cfg, _ = parse(__doc__, add_args)
    out = Path(args.out) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    r, thr = cfg.roles, cfg.fidelity.threshold
    clf = None
    for run in args.runs:
        df = pd.read_csv(Path(run) / "scores.csv", dtype={"image_id": str})
        species = run_species(run, cfg.species_csv)
        if clf is None:
            clf = build_classifier(r.closed_target, cfg, species)
            clf.model.float()
            clf.dtype = torch.float32
        else:
            clf.set_species(species)
        fid = f"{r.eval_oracle}_p_true" if f"{r.eval_oracle}_p_true" in df else "reward_oracle_p_true"
        good = df[(df[fid] >= thr) & (df[f"{r.target}_shared_ranks"] < 7)]
        for method, g in good.groupby("method"):
            g = g.head(args.k)
            fig, axes = plt.subplots(len(g), 3, figsize=(7.5, 2.5 * len(g)), squeeze=False)
            for ax_row, row in zip(axes, g.itertuples()):
                im = Image.open(Path(run) / "images" / f"{row.image_id}.png").convert("RGB")
                x = clf.image_tensor([im])[0]
                pred = int(clf.probs([im]).argmax())
                ax_row[0].imshow(im.resize((224, 224)))
                ax_row[0].set_title(f"true: {species[row.species_idx].common_name}", fontsize=8)
                for ax, c, title in ((ax_row[1], row.species_idx, "saliency of true"),
                                     (ax_row[2], pred, f"saliency of {species[pred].common_name}")):
                    ax.imshow(im.resize((224, 224)))
                    ax.imshow(smoothgrad(clf, x, c, args.samples, args.noise), cmap="inferno", alpha=0.6)
                    ax.set_title(title, fontsize=8)
                for ax in ax_row:
                    ax.axis("off")
            fig.tight_layout()
            name = f"saliency_{Path(run).name}_{method}.png".replace(":", "_")
            fig.savefig(out / name, dpi=150)
            plt.close(fig)
            print("wrote", out / name)


if __name__ == "__main__":
    main()
