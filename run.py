"""BioFool launcher: every step of the protocol, in order. Works on Linux, macOS, Windows and Colab.

    python run.py list                       the steps and what they do
    python run.py check --profile t4         one step (or several: python run.py baseline templates)
    python run.py all --profile t4           every step that needs no human, in order
    python run.py gpu --profile t4           only the heavy GPU steps (for Colab)
    python run.py light --profile rtx2060    the steps that fit a 6 GB GPU
    python run.py baseline --set generator.steps=10     extra config overrides for every command

Steps are resumable: a finished step is skipped (its output has a DONE file), an interrupted one starts again,
DDPO resumes from its last checkpoint, and scoring only computes what is missing. So after a Colab disconnection,
just run the same command again.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
E = "configs/experiments"
REAL = ["inat80", "invasive", "rare"]


def gen(*args, sets=()):
    return (["scripts/02_generate.py", *args], list(sets))


# step name -> (description, [(script argv, step-specific --set overrides), ...])
STEPS = {
    "test": ("offline CPU tests", [(["-m", "pytest", "-q"], None)]),
    "check": ("download every model, generate and score one image", [(["scripts/00_check_setup.py"], [])]),
    "taxonomy": ("align data/species.csv with BioCLIP's TreeOfLife names",
                 [(["scripts/sync_taxonomy_tol.py", "--write"], [])]),
    "smoke": ("the whole pipeline on 2 species, a few minutes", [
        gen("--config", f"{E}/templates.yaml", "--out", "outputs/smoke", "--limit", "2",
            sets=["attack.n_per_prompt=2", "attack.modifiers_per_species=2", "generator.steps=10"]),
        (["scripts/03_score.py", "--runs", "outputs/smoke"], []),
        (["scripts/08_analyze.py", "--runs", "outputs/smoke", "--out", "outputs/smoke_results", "--n-boot", "50"], []),
    ]),
    # ---- datasets
    "real_inat": ("dataset inat80: iNaturalist photos of the 80 species",
                  [(["scripts/01_fetch_inat.py", "--out", "outputs/inat80", "--per-species", "10"], [])]),
    "real_invasive": ("dataset invasive: 40 invasive alien species (EU list)",
                      [(["scripts/01_fetch_inat.py", "--out", "outputs/invasive", "--dataset", "invasive",
                         "--species-list", "data/invasive_species.txt", "--per-species", "8"], [])]),
    "real_rare": ("dataset rare: BioCLIP's Rare Species benchmark",
                  [(["scripts/01_fetch_rare_species.py", "--out", "outputs/rare", "--species", "100",
                     "--per-species", "4"], [])]),
    # ---- text-to-image attacks
    "baseline": ("clean prompts, 80 species (defines the attackable set)",
                 [gen("--config", f"{E}/baseline.yaml", "--out", "outputs/baseline")]),
    "templates": ("context modifiers, held-out species",
                  [gen("--config", f"{E}/templates.yaml", "--split", "B", "--out", "outputs/templates_B")]),
    "targeted": ("look-alike targets", [gen("--config", f"{E}/targeted.yaml", "--split", "B", "--out",
                                            "outputs/targeted_B", sets=["attack.n_per_prompt=2"])]),
    "llm": ("LLM attacker (OpenAI) then generation", [
        (["scripts/07_llm_attacker.py", "--out", "outputs/llm_modifiers_B.json", "--split", "B"], []),
        gen("--method", "llm", "--modifiers", "outputs/llm_modifiers_B.json", "--split", "B", "--tag", "llm",
            "--out", "outputs/llm_B", sets=["attack.n_per_prompt=2"]),
    ]),
    "bon": ("best-of-N", [gen("--config", f"{E}/bon.yaml", "--split", "B", "--tag", "bon", "--out",
                              "outputs/bon_B")]),
    "bon_early": ("best-of-N with early rejection", [gen("--config", f"{E}/bon_early.yaml", "--split", "B", "--tag",
                                                         "bon_early", "--out", "outputs/bon_early_B")]),
    # ---- attacks on real photos
    "i2i": ("img2img on the 3 real datasets, strengths 0.3/0.5/0.7", [
        gen("--method", "img2img", "--source", f"outputs/{d}", "--strengths", "0.3", "0.5", "0.7",
            "--per-species", "2", "--out", f"outputs/i2i_{d}", sets=["attack.modifiers_per_species=2"]) for d in REAL]),
    "i2i_bon": ("img2img + best-of-N + early rejection, strength 0.5", [
        gen("--config", f"{E}/bon_early.yaml", "--method", "img2img_bon", "--source", f"outputs/{d}", "--strengths",
            "0.5", "--per-species", "1", "--out", f"outputs/i2i_bon_{d}", sets=["attack.modifiers_per_species=2"])
        for d in REAL]),
    # ---- post-training
    "raft": ("RAFT: search on split A, select, LoRA fine-tuning", [
        gen("--config", f"{E}/bon.yaml", "--split", "A", "--tag", "bon_A", "--out", "outputs/bon_A"),
        (["scripts/03_score.py", "--runs", "outputs/bon_A", "--classifiers", "bioclip_tol", "bioclip2"], []),
        (["scripts/04_train_raft.py", "--run", "outputs/bon_A", "--out", "checkpoints/raft"], []),
    ]),
    "ddpo": ("DDPO / GRPO LoRA training", [(["scripts/05_train_ddpo.py", "--config", f"{E}/ddpo.yaml", "--out",
                                             "checkpoints/ddpo"], [])]),
    "ddpo_ablation": ("DDPO without the fidelity term (reward hacking)", [
        (["scripts/05_train_ddpo.py", "--config", f"{E}/ddpo_no_fidelity.yaml", "--out", "checkpoints/ddpo_nofid"],
         ["ddpo.epochs=15"])]),
    "eval_lora": ("the trained LoRAs on held-out species", [
        gen("--config", f"{E}/templates.yaml", "--split", "B", "--tag", tag, "--out", f"outputs/{tag}_B",
            sets=[f"generator.lora_path=checkpoints/{ck}"])
        for tag, ck in (("raft", "raft"), ("ddpo", "ddpo/lora_final"), ("ddpo_nofid", "ddpo_nofid/lora_final"))]),
    # ---- inference optimisation
    "bench": ("sampling modes: speed, memory, images", [
        (["scripts/06_benchmark_inference.py", "--out", "outputs/bench", "--limit", "24"], []),
    ] + ([] if sys.platform == "win32" else [   # torch.compile needs Triton, which is not available on Windows
        (["scripts/06_benchmark_inference.py", "--out", "outputs/bench_compile", "--modes", "ddim", "--limit", "24"],
         ["generator.compile=true"])])),
    "early": ("early-vs-final reward correlation", [
        (["scripts/06_benchmark_inference.py", "--out", "outputs/early", "--early-correlation", "--limit", "24"], [])]),
    "vae": ("full VAE vs TAESD decoding", [
        (["scripts/06_benchmark_inference.py", "--out", "outputs/vae", "--vae-fidelity", "--limit", "24"], [])]),
    # ---- evaluation
    "score": ("score every run with every classifier (incremental)", "SCORE"),
    "analyze": ("tables and figures for the report", "ANALYZE"),
    "interpret": ("embeddings, layer drift, saliency", "INTERPRET"),
    "human_sample": ("sample images for the human study", "HUMAN"),
}

RUNS = (["outputs/" + d for d in REAL] +
        ["outputs/" + r for r in ("baseline", "templates_B", "targeted_B", "llm_B", "bon_B", "bon_early_B")] +
        [f"outputs/i2i_{d}" for d in REAL] + [f"outputs/i2i_bon_{d}" for d in REAL] +
        ["outputs/raft_B", "outputs/ddpo_B", "outputs/ddpo_nofid_B", "outputs/bench"])

GROUPS = {
    "data": ["real_inat", "real_invasive", "real_rare"],
    "attacks": ["baseline", "templates", "targeted", "llm", "bon", "bon_early", "i2i", "i2i_bon"],
    "training": ["raft", "ddpo", "ddpo_ablation", "eval_lora"],
    "speed": ["bench", "early", "vae"],
    "eval": ["score", "analyze", "interpret", "human_sample"],
    # what to run where, for a T4 on Colab + a 6 GB GPU at home
    "gpu": ["bon", "bon_early", "i2i_bon", "raft", "ddpo", "ddpo_ablation", "eval_lora", "bench", "early"],
    "light": ["real_inat", "real_invasive", "real_rare", "baseline", "templates", "targeted", "llm", "i2i", "vae"],
}
GROUPS["all"] = ["check", "taxonomy"] + GROUPS["data"] + GROUPS["attacks"] + GROUPS["training"] + GROUPS["speed"] + \
    GROUPS["eval"]


def dynamic(kind):
    runs = [r for r in RUNS if Path(r, "DONE").exists()]
    if kind == "SCORE":
        return [(["scripts/03_score.py", "--runs", *runs], [])]
    if kind == "ANALYZE":
        return [(["scripts/08_analyze.py", "--runs", *runs, "--attackable-from", "outputs/baseline", "--best-only",
                  "--bench", "outputs/bench/bench.csv", "--early", "outputs/early/early_correlation.csv"], [])]
    if kind == "INTERPRET":
        sal = [r for r in ("outputs/templates_B", "outputs/i2i_inat80", "outputs/ddpo_B") if r in runs]
        return [(["scripts/10_embeddings.py", "--runs", *runs], []), (["scripts/11_saliency.py", "--runs", *sal], [])]
    if kind == "HUMAN":
        hs = [r for r in ("outputs/baseline", "outputs/templates_B", "outputs/bon_early_B", "outputs/i2i_inat80",
                          "outputs/i2i_rare", "outputs/ddpo_B", "outputs/ddpo_nofid_B") if r in runs]
        return [(["scripts/09_human_eval.py", "sample", "--runs", *hs, "--n", "30"], None)]
    raise ValueError(kind)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("steps", nargs="+", help="step or group names, or 'list'")
    p.add_argument("--profile", default=os.environ.get("BIOFOOL_PROFILE"), help="t4 | rtx2060 (hardware profile)")
    p.add_argument("--set", nargs="*", default=[], help="config overrides added to every command")
    p.add_argument("--keep-going", action="store_true", help="continue with the next step if one fails")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.steps == ["list"]:
        for name, (desc, _) in STEPS.items():
            print(f"  {name:15s} {desc}")
        for name, members in GROUPS.items():
            print(f"  {name:15s} = {' '.join(members)}")
        return
    names = [s for name in args.steps for s in GROUPS.get(name, [name])]
    unknown = [n for n in names if n not in STEPS]
    if unknown:
        sys.exit(f"unknown step(s): {unknown}. See: python run.py list")
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"            # Windows: UTF-8 for every file the scripts open, and for their output
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.profile:
        env["BIOFOOL_PROFILE"] = args.profile
    Path("logs").mkdir(exist_ok=True)
    for name in names:
        desc, cmds = STEPS[name]
        cmds = dynamic(cmds) if isinstance(cmds, str) else cmds
        print(f"\n=== {name}: {desc}  (profile: {args.profile or 'default'})", flush=True)
        t0 = time.time()
        for argv, sets in cmds:
            cmd = [PY, *argv]
            if sets is not None and (sets or args.set):
                cmd += ["--set", *sets, *args.set]
            print("$", " ".join(cmd), flush=True)
            if args.dry_run:
                continue
            with open(Path("logs") / f"{name}.log", "a", encoding="utf-8") as log:
                proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                        encoding="utf-8", errors="replace", bufsize=1)
                for line in proc.stdout:
                    sys.stdout.write(line)
                    log.write(line)
                code = proc.wait()
            if code != 0:
                print(f"!!! step {name} failed (exit {code}); see logs/{name}.log", flush=True)
                if not args.keep_going:
                    sys.exit(code)
                break
        print(f"=== {name} finished in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
