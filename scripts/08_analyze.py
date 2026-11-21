"""Aggregate scored runs into the tables and figures of the report.

Outputs in --out (default results/):
  tables/main.{md,tex}         per dataset x method: faithful rate, ASR [95% CI], effective ASR, ASR@family,
                               confident errors, taxonomic distance          (Table 1 of the report)
  tables/reference.{md,tex}    BioCLIP on the REAL photos of each dataset (accuracy = 1 - fool rate)
  tables/by_group.{md,tex}     per method x taxonomic group
  tables/transfer.{md,tex}     ASR of the same images against every other classifier
  tables/families.{md,tex}     ASR per modifier family (which context breaks BioCLIP?)
  tables/speed, early_correlation, attackable_species.csv (if the inputs are given)
  figures/asr_by_method.png, asr_by_group.png, transfer_heatmap.png, taxo_distance.png, img2img_strength.png,
          pareto_speed.png, early_correlation.png

Example:  see `python run.py analyze`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from biofool.cli import parse  # noqa: E402
from biofool.metrics import add_outcomes, spearman, summary, summary_table  # noqa: E402


def add_args(p):
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default="results")
    p.add_argument("--bench", default=None)
    p.add_argument("--early", default=None)
    p.add_argument("--best-only", action="store_true", help="best_of_n runs: keep only the selected image per prompt")
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--attackable-from", default=None,
                   help="scored synth80 baseline run: keep only the synth80 species whose clean images are faithful "
                        "AND correctly classified (the 'attackable set'); species the generator cannot draw are out")


def save(df: pd.DataFrame, path: Path):
    df = df.round(3)
    path.with_suffix(".md").write_text(df.to_markdown(index=False))
    path.with_suffix(".tex").write_text(df.to_latex(index=False, float_format="%.3f", na_rep="--"))
    print(f"\n== {path.name}\n{df.to_markdown(index=False)}")


def main():
    args, cfg, _ = parse(__doc__, add_args)
    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_csv(Path(r) / "scores.csv", dtype={"image_id": str}).assign(run=Path(r).name)
                    for r in args.runs if (Path(r) / "scores.csv").exists()], ignore_index=True)
    df["dataset"] = df.get("dataset", pd.Series(index=df.index, dtype=object)).fillna("synth80")
    if args.best_only and "is_best" in df:
        df = df[df.is_best.fillna(True).astype(bool)]
    r, thr = cfg.roles, cfg.fidelity.threshold
    eval_oracle = r.eval_oracle if f"{r.eval_oracle}_p_true" in df else r.reward_oracle
    if eval_oracle != r.eval_oracle:
        print(f"WARNING: {r.eval_oracle} scores missing, fidelity measured with {eval_oracle} (= reward oracle: biased)")

    if args.attackable_from:
        base = add_outcomes(pd.read_csv(Path(args.attackable_from) / "scores.csv"), r.target, eval_oracle, thr)
        per = base.groupby("latin").agg(faithful=("faithful", "mean"), correct=("fooled", lambda x: 1 - x.mean()))
        keep = set(per[(per.faithful >= 0.5) & (per.correct >= 0.5)].index)
        print(f"attackable set: {len(keep)} / {len(per)} synth80 species")
        pd.Series(sorted(keep), name="latin").to_csv(out / "tables" / "attackable_species.csv", index=False)
        df = df[(df.dataset != "synth80") | df.latin.isin(keep)]
    all_df = add_outcomes(df, r.target, eval_oracle, thr)
    real = all_df[all_df.method == "real"]
    main_df = all_df[all_df.method != "real"]

    # ---- reference: the tool on real photos (no attack). Faithful = all, by construction.
    if len(real):
        ref = real.groupby("dataset").agg(n=("fooled", "size"), accuracy=("fooled", lambda x: 1 - x.mean()),
                                          genus_acc=("wrong_genus", lambda x: 1 - x.mean()),
                                          vlm_agrees=("faithful", "mean")).reset_index()
        save(ref, out / "tables" / "reference")

    # ---- main tables
    save(summary_table(main_df, ["dataset", "method"], n_boot=args.n_boot), out / "tables" / "main")
    save(summary_table(main_df, ["method", "group"], ci=False), out / "tables" / "by_group")
    if "family" in main_df:
        save(summary_table(main_df, ["family"], ci=False).sort_values("ASR", ascending=False), out / "tables" / "families")

    # ---- transfer: the same images, other classifiers as "target"
    rows = []
    for model in dict.fromkeys([r.target, *r.transfer]):
        if f"{model}_shared_ranks" not in df:
            continue
        t = add_outcomes(df[df.method != "real"], model, eval_oracle, thr)
        for method, g in t.groupby("method"):
            rows.append(dict(method=method, model=model, ASR=summary(g)["ASR"]))
    transfer = pd.DataFrame(rows).pivot(index="method", columns="model", values="ASR").reset_index()
    save(transfer, out / "tables" / "transfer")

    # ---- figures
    tab = summary_table(main_df, ["method"], n_boot=args.n_boot)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(tab.method, tab.ASR, yerr=[(tab.ASR - tab.ASR_lo).clip(0), (tab.ASR_hi - tab.ASR).clip(0)], capsize=3,
           color="#2a78d6", label="ASR = P(fooled | faithful)")
    ax.scatter(tab.method, tab.faithful_rate, color="#e0702a", zorder=3, label="faithful rate")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out / "figures" / "asr_by_method.png", dpi=200)
    plt.close(fig)

    by_group = summary_table(main_df, ["method", "group"], ci=False).pivot(index="method", columns="group", values="ASR")
    ax = by_group.plot.bar(figsize=(8, 3.5), ylim=(0, 1), ylabel="ASR", rot=30)
    ax.figure.tight_layout()
    ax.figure.savefig(out / "figures" / "asr_by_group.png", dpi=200)
    plt.close(ax.figure)

    m = transfer.set_index("method")
    fig, ax = plt.subplots(figsize=(1.4 * len(m.columns) + 3, 0.4 * len(m) + 1.5))
    im = ax.imshow(m.values, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(m.columns)), m.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(m.index)), m.index)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="ASR")
    fig.tight_layout()
    fig.savefig(out / "figures" / "transfer_heatmap.png", dpi=200)
    plt.close(fig)

    fooled = main_df[main_df.faithful & main_df.fooled]
    if len(fooled):
        ax = fooled.groupby("method").taxo_dist.value_counts(normalize=True).unstack(fill_value=0).plot.bar(
            stacked=True, figsize=(8, 3.5), ylabel="share of errors", rot=30)
        ax.legend(title="taxonomic distance", fontsize=7)
        ax.figure.tight_layout()
        ax.figure.savefig(out / "figures" / "taxo_distance.png", dpi=200)
        plt.close(ax.figure)

    # ---- img2img: fidelity / ASR trade-off against strength
    i2i = main_df[main_df.method.str.startswith("img2img") & main_df.strength.notna()] if "strength" in main_df else []
    if len(i2i):
        st = summary_table(i2i, ["dataset", "strength"], ci=False)
        save(st, out / "tables" / "img2img_strength")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for ds, g in st.groupby("dataset"):
            ax.plot(g.strength, g.ASR, marker="o", label=f"{ds}: ASR")
            ax.plot(g.strength, g.faithful_rate, marker="s", linestyle="--", label=f"{ds}: faithful")
        ax.set(xlabel="img2img strength", ylim=(0, 1))
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "figures" / "img2img_strength.png", dpi=200)
        plt.close(fig)

    # ---- speed / quality trade-off
    if args.bench and Path(args.bench).exists():
        bench = pd.read_csv(args.bench)
        q = summary_table(main_df[main_df.method.str.startswith("fast:")], ["method"], ci=False)
        q["mode"] = q.method.str.removeprefix("fast:")
        bench = bench.merge(q, on="mode")
        bench["faithful_attacks_per_hour"] = bench.images_per_hour * bench.effective_ASR
        save(bench, out / "tables" / "speed")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.scatter(bench.seconds_per_image, bench.faithful_rate, label="faithful rate")
        ax.scatter(bench.seconds_per_image, bench.ASR, label="ASR")
        for _, row in bench.iterrows():
            ax.annotate(row["mode"], (row.seconds_per_image, row.faithful_rate), fontsize=8, xytext=(4, 4),
                        textcoords="offset points")
        ax.set(xlabel="seconds per image", ylim=(0, 1))
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "figures" / "pareto_speed.png", dpi=200)
        plt.close(fig)

    if args.early and Path(args.early).exists():
        e = pd.read_csv(args.early)
        # rank correlation WITHIN each prompt (best-of-N only compares candidates of the same prompt), then averaged
        rho = e.groupby(["fraction", "latin"]).apply(lambda g: spearman(g.early_reward, g.final_reward))
        rho = rho.groupby("fraction").mean().rename("spearman")
        save(rho.reset_index(), out / "tables" / "early_correlation")
        ax = rho.plot(marker="o", figsize=(5, 3), ylim=(0, 1), xlabel="fraction of denoising steps",
                      ylabel="Spearman(early, final reward)")
        ax.figure.tight_layout()
        ax.figure.savefig(out / "figures" / "early_correlation.png", dpi=200)
        plt.close(ax.figure)


if __name__ == "__main__":
    main()
