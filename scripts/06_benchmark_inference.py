"""Inference acceleration study.

Part 1 (--modes): time / memory / throughput of each sampling mode, and the images it produces (scored afterwards
with 03_score.py, so that every mode gets an ASR and a fidelity: the quality-vs-cost trade-off).
    ddim (25 steps, as trained), ddim_10, dpm (12 steps), lcm (4 steps), and the same with the tiny VAE (_taesd),
    in fp32 vs fp16, with or without torch.compile.
Part 2 (--early-correlation): does the reward of the x0 preview after k steps predict the final reward?
    This is the hypothesis behind early rejection (search.py). Writes early_correlation.csv.
Part 3 (--vae-fidelity): the latent-diffusion trade-off of Lab 2. The same final latents are decoded by the full SD
    VAE and by the tiny TAESD decoder: PSNR between the two images, decoding time, and whether BioCLIP's top-1 agrees.

Example:
  python scripts/06_benchmark_inference.py --out outputs/bench --modes ddim ddim_10 dpm lcm lcm_taesd --limit 20
  python scripts/03_score.py --runs outputs/bench
  python scripts/06_benchmark_inference.py --out outputs/early --early-correlation --limit 20
  python scripts/06_benchmark_inference.py --out outputs/vae --vae-fidelity --limit 20
"""

import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from biofool.cli import parse, select_species  # noqa: E402
from biofool.generator import Generator, set_fast_mode  # noqa: E402
from biofool.io import RunWriter, begin_run, finish_run  # noqa: E402
from biofool.prompts import attack_prompts  # noqa: E402


def add_args(p):
    p.add_argument("--out", required=True)
    p.add_argument("--modes", nargs="*", default=["ddim_50", "ddim", "ddim_10", "dpm", "lcm", "lcm_taesd"])
    p.add_argument("--limit", type=int, default=20, help="number of species")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--early-correlation", action="store_true")
    p.add_argument("--vae-fidelity", action="store_true")
    p.add_argument("--fractions", nargs="*", type=float, default=[0.2, 0.3, 0.4, 0.6])


def bench_modes(args, cfg, species):
    rng = random.Random(cfg.seed)
    chosen = select_species(species, "all", None, args.limit)
    jobs = [(s, *attack_prompts(s, 1, rng)[0]) for s in chosen]
    seeds = [rng.randrange(2**31) for _ in jobs]
    writer = RunWriter(args.out, species)
    rows = []
    for mode in args.modes:
        gen = Generator(cfg)  # fresh pipeline for each mode (modes change the scheduler / VAE / adapters)
        base, _, n_steps = mode.partition("_") if mode.split("_")[-1].isdigit() else (mode, "", "")
        steps, guidance = set_fast_mode(gen, base)
        steps = int(n_steps) if n_steps else steps
        gen.generate_fast([jobs[0][3]], [0], steps, guidance)  # warm-up (and compilation)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(0, len(jobs), args.batch):
            chunk = jobs[i:i + args.batch]
            ims = gen.generate_fast([j[3] for j in chunk], seeds[i:i + args.batch], steps, guidance)
            for (s, fam, mod, prompt), seed, im in zip(chunk, seeds[i:i + args.batch], ims):
                writer.add(im, species_idx=s.idx, latin=s.latin, group=s.group, split=s.split, dataset="synth80",
                           family=fam,
                           modifier=mod, prompt=prompt, target_idx=None, seed=seed, method=f"fast:{mode}",
                           steps=steps, guidance=guidance)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sec = (time.perf_counter() - t0) / len(jobs)
        vram = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else float("nan")
        rows.append(dict(mode=mode, steps=steps, guidance=guidance, dtype=cfg.generator.dtype,
                         compile=cfg.generator.compile, seconds_per_image=sec, images_per_hour=3600 / sec,
                         peak_vram_gib=vram))
        print(rows[-1])
        del gen
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    writer.close()
    with open(Path(args.out) / "bench.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def early_correlation(args, cfg, species):
    from biofool.scoring import Scorer

    rng = random.Random(cfg.seed)
    gen, scorer = Generator(cfg), Scorer(cfg, species)
    steps = cfg.generator.steps
    rows = []
    for s in select_species(species, "all", None, args.limit):
        _, _, prompt = attack_prompts(s, 1, rng)[0]
        seeds = [rng.randrange(2**31) for _ in range(cfg.best_of_n.n)]
        final = scorer(gen.sample([prompt] * len(seeds), seeds).images, [s.idx] * len(seeds))["reward"]
        for frac in args.fractions:
            early = {}

            def record(x0):  # keep every candidate, just record the early reward
                early["r"] = scorer(gen.decode(x0, preview=True), [s.idx] * len(x0))["reward"]
                return torch.arange(len(x0))

            gen.sample([prompt] * len(seeds), seeds, prune=(max(1, round(frac * steps)), record))
            for j in range(len(seeds)):
                rows.append(dict(latin=s.latin, group=s.group, seed=seeds[j], fraction=frac,
                                 early_reward=early["r"][j].item(), final_reward=final[j].item()))
        print(s.latin, "done")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "early_correlation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def vae_fidelity(args, cfg, species):
    import numpy as np

    from biofool.classifiers import build_classifier

    rng = random.Random(cfg.seed)
    gen = Generator(cfg)
    target = build_classifier(cfg.roles.target, cfg, species)
    rows = []
    for s in select_species(species, "all", None, args.limit):
        _, _, prompt = attack_prompts(s, 1, rng)[0]
        out = gen.sample([prompt] * args.batch, [rng.randrange(2**31) for _ in range(args.batch)])
        timings = {}
        for name, preview in (("vae", False), ("taesd", True)):
            gen.decode(out.final_latents[:1], preview=preview)  # warm-up / load
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            ims = gen.decode(out.final_latents, preview=preview)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timings[name] = ((time.perf_counter() - t0) / len(ims), ims)
        full, tiny = timings["vae"][1], timings["taesd"][1]
        pf = target.predict(full, [s.idx] * len(full))
        pt = target.predict(tiny, [s.idx] * len(tiny))
        for j in range(len(full)):
            a, b = (np.asarray(x, dtype=np.float64) / 255 for x in (full[j], tiny[j]))
            mse = ((a - b) ** 2).mean()
            rows.append(dict(latin=s.latin, psnr=-10 * np.log10(max(mse, 1e-12)),
                             sec_vae=timings["vae"][0], sec_taesd=timings["taesd"][0],
                             top1_agree=pf["top1_name"][j] == pt["top1_name"][j],
                             p_true_vae=float(pf["p_true"][j]), p_true_taesd=float(pt["p_true"][j])))
        print(s.latin, "done")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "vae_fidelity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    args, cfg, species = parse(__doc__, add_args)
    if not begin_run(args.out):
        return
    if args.vae_fidelity:
        vae_fidelity(args, cfg, species)
    elif args.early_correlation:
        early_correlation(args, cfg, species)
    else:
        bench_modes(args, cfg, species)
    finish_run(args.out)


if __name__ == "__main__":
    main()
