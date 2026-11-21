"""Blind human evaluation of fidelity: "does this image show <species>?"

1. sample:  python scripts/09_human_eval.py sample --runs outputs/baseline outputs/bon outputs/ddpo_B --n 40
            -> human_eval/manifest.csv (n images per method, shuffled, method hidden from annotators)
2. annotate (each team member, ~15 min for 160 images):
            python scripts/09_human_eval.py annotate --annotator alice      (opens a Gradio app in the browser)
3. report:  python scripts/09_human_eval.py report
            -> human fidelity per method, agreement between annotators and with the VLM judge (Cohen's kappa)
"""

import argparse
import csv
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path("human_eval")


def species_of(run):
    """The label set of a run (each dataset has its own species list)."""
    path = Path(run) / "species.csv"
    return pd.read_csv(path if path.exists() else "data/species.csv")


def sample(runs, n, seed=0):
    df = pd.concat([pd.read_csv(Path(r) / "scores.csv", dtype={"image_id": str}).assign(run=r) for r in runs])
    df["common_name"] = [species_of(r).iloc[i].common_name for r, i in zip(df.run, df.species_idx)]
    df["binomial"] = [f"{species_of(r).iloc[i].genus} {species_of(r).iloc[i].epithet}"
                      for r, i in zip(df.run, df.species_idx)]
    if "is_best" in df:
        df = df[df.is_best.fillna(True).astype(bool)]
    df = df.groupby("method", group_keys=False).apply(lambda g: g.sample(min(n, len(g)), random_state=seed))
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    (ROOT / "images").mkdir(parents=True, exist_ok=True)
    df["item"] = [f"{i:04d}" for i in range(len(df))]
    for row in df.itertuples():
        shutil.copy(Path(row.run) / "images" / f"{row.image_id}.png", ROOT / "images" / f"{row.item}.png")
    df.to_csv(ROOT / "manifest.csv", index=False)
    print(f"{len(df)} items -> {ROOT / 'manifest.csv'}")


def annotate(annotator):
    import gradio as gr

    man = pd.read_csv(ROOT / "manifest.csv", dtype={"item": str})
    out = ROOT / f"answers_{annotator}.csv"
    done = set(pd.read_csv(out, dtype={"item": str}).item) if out.exists() else set()
    todo = [r for r in man.itertuples() if r.item not in done]

    def question(i):
        if i >= len(todo):
            return None, "All done, thank you!", i
        row = todo[i]
        return (str(ROOT / "images" / f"{row.item}.png"),
                f"### {len(done) + i + 1}/{len(man)}: is this a **{row.common_name}** (*{row.binomial}*)?\n"
                "Answer yes only if the organism is clearly this species and the photo looks realistic.", i)

    def answer(label, i):
        if i < len(todo):
            new = not out.exists()
            with open(out, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["item", "annotator", "answer"])
                w.writerow([todo[i].item, annotator, label])
        return question(i + 1)

    with gr.Blocks() as app:
        idx = gr.State(0)
        img, txt = gr.Image(height=512), gr.Markdown()
        with gr.Row():
            buttons = [gr.Button(lbl) for lbl in ("yes", "no", "unsure")]
        for b in buttons:
            b.click(answer, [gr.State(b.value), idx], [img, txt, idx])
        app.load(question, idx, [img, txt, idx])
    app.launch()


def report():
    from sklearn.metrics import cohen_kappa_score

    man = pd.read_csv(ROOT / "manifest.csv", dtype={"item": str})
    ans = pd.concat([pd.read_csv(f, dtype={"item": str}) for f in ROOT.glob("answers_*.csv")])
    ans = ans[ans.answer != "unsure"]
    ans["yes"] = ans.answer == "yes"
    votes = ans.groupby("item").yes.mean().rename("human_yes")
    df = man.merge(votes, left_on="item", right_index=True)
    df["human_faithful"] = df.human_yes >= 0.5
    print(df.groupby("method").human_faithful.agg(["mean", "count"]).round(3).to_markdown())
    wide = ans.pivot_table(index="item", columns="annotator", values="yes", aggfunc="first").dropna()
    names = list(wide.columns)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            print(f"kappa {names[i]} vs {names[j]}: {cohen_kappa_score(wide[names[i]], wide[names[j]]):.3f}")
    for judge in ("gpt4o_judge", "vlm_judge"):
        if f"{judge}_p_true" in df:
            k = cohen_kappa_score(df.human_faithful, df[f"{judge}_p_true"] >= 0.5)
            print(f"kappa humans (majority) vs {judge}: {k:.3f}   <- validates the automatic evaluation oracle")
    df.to_csv(ROOT / "report.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--n", type=int, default=40)
    a = sub.add_parser("annotate")
    a.add_argument("--annotator", required=True)
    sub.add_parser("report")
    args = p.parse_args()
    {"sample": lambda: sample(args.runs, args.n), "annotate": lambda: annotate(args.annotator),
     "report": report}[args.cmd]()
