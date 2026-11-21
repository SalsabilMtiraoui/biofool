"""Where do the attacks move images in BioCLIP's representation space? (the analyses of Lab 3, Exercise 3)

1. t-SNE of BioCLIP image embeddings (final layer) per dataset: real photos, clean generations and attacks,
   fooled images marked with a cross.                                        -> figures/tsne_<dataset>.png
2. Text alignment: cos(image, text of true species) and cos(image, text of predicted species).
   A successful attack keeps the image close to real photos of its species while pushing it across the decision
   boundary.                                                                 -> tables/embedding_alignment.md
3. Layer-wise drift for img2img attacks (the "early = lexical, late = semantic" conjecture of Lab 3, here for a
   vision transformer): cosine similarity between the CLS token of the real photo and of its attacked version
   after every transformer block, for fooled vs not-fooled attacks.        -> figures/layer_drift.png

Example:  python scripts/10_embeddings.py --runs outputs/inat80 outputs/baseline outputs/templates_B outputs/i2i_inat
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from biofool.classifiers import build_classifier  # noqa: E402
from biofool.cli import parse  # noqa: E402
from biofool.io import run_species  # noqa: E402


def add_args(p):
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default="results")
    p.add_argument("--max-per-dataset", type=int, default=1500)
    p.add_argument("--max-pairs", type=int, default=300)


def load(run, ids):
    return [Image.open(Path(run) / "images" / f"{i}.png") for i in ids]


@torch.no_grad()
def cls_per_layer(clf, images) -> torch.Tensor:
    """[n_images, n_blocks, dim] normalised CLS token after every transformer block of the vision tower."""
    blocks = clf.model.visual.transformer.resblocks
    acts = []
    hooks = [b.register_forward_hook(lambda m, i, o: acts.append(o)) for b in blocks]
    try:
        clf.model.encode_image(clf.image_tensor(images))
    finally:
        for h in hooks:
            h.remove()
    n = len(images)
    cls = [(a[:, 0] if a.shape[0] == n else a[0]).float() for a in acts]  # batch-first or sequence-first layouts
    return torch.nn.functional.normalize(torch.stack(cls, 1), dim=-1).cpu()


def main():
    from sklearn.manifold import TSNE

    args, cfg, _ = parse(__doc__, add_args)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    r = cfg.roles
    clf = None
    frames, feats = [], []
    for run in args.runs:
        df = pd.read_csv(Path(run) / "scores.csv", dtype={"image_id": str, "source_image_id": str})
        df["run_dir"] = run
        df["dataset"] = df.get("dataset", pd.Series(index=df.index, dtype=object)).fillna("synth80")
        df = df.sample(min(len(df), args.max_per_dataset), random_state=cfg.seed)
        species = run_species(run, cfg.species_csv)
        if clf is None:
            clf = build_classifier(r.closed_target, cfg, species)
        else:
            clf.set_species(species)
        f = clf.image_features(load(run, df.image_id))
        t = clf.text_features.cpu()
        top = (f.cpu() @ t.T).argmax(-1)
        df["cos_true"] = (f.cpu() * t[df.species_idx.values]).sum(-1).numpy()
        df["cos_pred"] = (f.cpu() * t[top]).sum(-1).numpy()
        df["fooled"] = df[f"{r.target}_shared_ranks"] < 7 if f"{r.target}_shared_ranks" in df else False
        frames.append(df)
        feats.append(f.cpu().numpy())
    df = pd.concat(frames, ignore_index=True)
    F = np.concatenate(feats)

    align = df.groupby(["dataset", "method", "fooled"])[["cos_true", "cos_pred"]].mean().reset_index().round(4)
    (out / "tables" / "embedding_alignment.md").write_text(align.to_markdown(index=False))
    print(align.to_markdown(index=False))

    for ds in sorted(df.dataset.unique()):
        idx = np.where((df.dataset == ds).values)[0]
        if len(idx) < 10:
            continue
        xy = TSNE(n_components=2, perplexity=min(30, len(idx) // 4), random_state=cfg.seed).fit_transform(F[idx])
        sub = df.iloc[idx]
        fig, ax = plt.subplots(figsize=(7, 6))
        for method in sorted(sub.method.unique()):
            m = (sub.method == method).values
            ax.scatter(xy[m & ~sub.fooled.values, 0], xy[m & ~sub.fooled.values, 1], s=8, alpha=0.6, label=method)
            ax.scatter(xy[m & sub.fooled.values, 0], xy[m & sub.fooled.values, 1], s=14, marker="x", alpha=0.8,
                       color=ax.collections[-1].get_facecolor()[0])
        ax.set_title(f"BioCLIP image embeddings, {ds} (x = fooled)")
        ax.legend(fontsize=7, markerscale=2)
        fig.tight_layout()
        fig.savefig(out / "figures" / f"tsne_{ds}.png", dpi=200)
        plt.close(fig)

    pairs = df[df.get("source_run", pd.Series(index=df.index, dtype=object)).notna()] if "source_run" in df else []
    if len(pairs):
        pairs = pairs.sample(min(len(pairs), args.max_pairs), random_state=cfg.seed)
        sims = []
        for row in pairs.itertuples():
            a = cls_per_layer(clf, load(row.source_run, [row.source_image_id]) + load(row.run_dir, [row.image_id]))
            sims.append((a[0] * a[1]).sum(-1).numpy())
        sims = np.stack(sims)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for fooled, label in ((False, "not fooled"), (True, "fooled")):
            m = pairs.fooled.values == fooled
            if m.any():
                mu, sd = sims[m].mean(0), sims[m].std(0)
                ax.plot(range(1, len(mu) + 1), mu, marker="o", label=f"{label} (n={m.sum()})")
                ax.fill_between(range(1, len(mu) + 1), mu - sd, mu + sd, alpha=0.2)
        ax.set(xlabel="transformer block of BioCLIP's vision tower", ylabel="cos(real photo, attack) of CLS token")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "figures" / "layer_drift.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
