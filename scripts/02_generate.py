"""Generate attack images.

Text-to-image (dataset synth80, the 80 species of data/species.csv):
  baseline     the plain prompt, n_per_prompt seeds per species
  templates    context modifiers (background, lighting, viewpoint, camera, style, life stage)
  targeted     prompts that steer towards a confusable species (curated pairs + BioCLIP text-embedding neighbours)
  best_of_n    templates + best-of-N search scored by the reward; with best_of_n.prune_at set: early rejection
  llm          modifiers proposed by scripts/07_llm_attacker.py (--modifiers file)
Image-to-image on REAL photos (datasets inat80 and rare; --source = a run written by 01_fetch_*.py):
  img2img      SDEdit: each photo is noised up to `strength` and re-generated with an attack modifier
  img2img_bon  the same with best-of-N (+ early rejection if best_of_n.prune_at is set)
Resumable image by image: if the run is interrupted (Ctrl+C, Colab disconnection, crash), run the same command
again and it continues after the last saved image (best-of-N: after the last complete prompt).
Add --set generator.lora_path=<dir> to sample from a RAFT / DDPO LoRA, and --tag to name the run's method.

Examples:
  python scripts/02_generate.py --config configs/experiments/templates.yaml --split B --out outputs/templates_B
  python scripts/02_generate.py --method img2img --source outputs/inat80 --strengths 0.4 0.6 --out outputs/i2i_inat
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image  # noqa: E402

from biofool.cli import parse, select_species  # noqa: E402
from biofool.generator import Generator  # noqa: E402
from biofool.io import RunWriter, finish_run, read_meta, run_species  # noqa: E402
from biofool.prompts import attack_prompts, baseline_prompt, targeted_prompts  # noqa: E402
from biofool.taxonomy import load_pairs  # noqa: E402


def add_args(p):
    p.add_argument("--out", required=True)
    p.add_argument("--method", default=None, help="overrides cfg.method")
    p.add_argument("--tag", default=None, help="name stored in the 'method' column (default: the method)")
    p.add_argument("--split", default="all", choices=["all", "A", "B"])
    p.add_argument("--groups", default=None, help="comma-separated, e.g. insects,fungi")
    p.add_argument("--limit", type=int, default=None, help="first N species only (smoke tests)")
    p.add_argument("--batch", type=int, default=None, help="images per batch (default: generator.batch, or 8)")
    p.add_argument("--modifiers", default=None, help="JSON {latin: [modifier, ...]} from 07_llm_attacker.py")
    p.add_argument("--source", default=None, help="img2img: run directory of real photos")
    p.add_argument("--strengths", nargs="*", type=float, default=[0.5])
    p.add_argument("--per-species", type=int, default=2, help="img2img: source photos per species")


def text_to_image_jobs(method, args, cfg, species, rng):
    """(species, family, modifier, prompt, target_idx) for every prompt."""
    chosen = select_species(species, args.split, args.groups, args.limit)
    jobs = []
    if method == "baseline":
        jobs = [(s, "none", "", baseline_prompt(s), None) for s in chosen]
    elif method in ("templates", "best_of_n"):
        for s in chosen:
            jobs += [(s, f, m, p, None) for f, m, p in attack_prompts(s, cfg.attack.modifiers_per_species, rng)]
    elif method == "targeted":
        from biofool.classifiers import build_classifier
        closed = build_classifier(cfg.roles.closed_target, cfg, species)
        pairs = load_pairs(cfg.pairs_csv, species)
        for s in chosen:
            targets = list(dict.fromkeys(pairs.get(s.idx, []) + closed.nearest_species(s.idx, 2)))[:3]
            for t in targets:
                jobs += [(s, f, m, p, t) for f, m, p in targeted_prompts(s, species[t])]
        del closed
    elif method == "llm":
        mods = json.loads(Path(args.modifiers).read_text())
        for s in chosen:
            jobs += [(s, "llm", m, f"{baseline_prompt(s)}, {m}", None) for m in mods.get(s.latin, [])]
    else:
        raise ValueError(method)
    return jobs


def main():
    args, cfg, species = parse(__doc__, add_args)
    out_dir = Path(args.out)
    if (out_dir / "DONE").exists():
        print(f"{out_dir} is complete: skipping (delete {out_dir / 'DONE'} to run it again)")
        return
    previous = read_meta(out_dir) if (out_dir / "meta.jsonl").exists() else []
    args.batch = args.batch or cfg.generator.get("batch", 8)
    method = args.method or cfg.get("method", "baseline")
    tag = args.tag or method
    rng = random.Random(cfg.seed)
    n = cfg.attack.n_per_prompt
    common = dict(method=tag, steps=cfg.generator.steps, guidance=cfg.generator.guidance_scale,
                  lora=cfg.generator.get("lora_path"))

    # ------------------------------------------------------------------ jobs
    if method.startswith("img2img"):
        species = run_species(args.source)
        src = [m for m in read_meta(args.source)]
        by_species = {}
        for m in src:
            by_species.setdefault(m["species_idx"], []).append(m)
        chosen = {s.idx for s in select_species(species, args.split, args.groups, args.limit)}
        jobs = []  # (species, family, modifier, prompt, target, source_meta, strength)
        for idx, items in by_species.items():
            if idx not in chosen:
                continue
            s = species[idx]
            for m in rng.sample(items, min(args.per_species, len(items))):
                for fam, mod, prompt in attack_prompts(s, cfg.attack.modifiers_per_species, rng):
                    jobs += [(s, fam, mod, prompt, None, m, st) for st in args.strengths]
        dataset = src[0]["dataset"]
    else:
        jobs = [(*j, None, 1.0) for j in text_to_image_jobs(method, args, cfg, species, rng)]
        dataset = "synth80"
    # ---- resume: the job list is deterministic (same seed), so we skip what is already on disk
    search = method in ("best_of_n", "img2img_bon")
    if previous and search:
        expected = cfg.best_of_n.keep if cfg.best_of_n.prune_at else cfg.best_of_n.n
        counts = {}
        for m in previous:
            counts[m["job"]] = counts.get(m["job"], 0) + 1
        complete = [m for m in previous if counts[m["job"]] == expected]  # drop a half-written last prompt
        with open(out_dir / "meta.jsonl", "w") as f:
            f.writelines(json.dumps(m) + "\n" for m in complete)
        done_jobs = {m["job"] for m in complete}
    else:
        done_jobs = set()
    if previous:
        print(f"resuming {out_dir}: {len(done_jobs) if search else len(previous)} "
              f"{'prompts' if search else 'images'} already done")
    gen = Generator(cfg)
    writer = RunWriter(args.out, species)
    (Path(args.out) / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"{method}: {len(jobs)} prompts, {dataset} -> {args.out}")

    def load_source(m):
        return Image.open(Path(args.source) / "images" / f"{m['image_id']}.png") if m else None

    def meta(s, fam, mod, prompt, tgt, m, st):
        return dict(species_idx=s.idx, latin=s.latin, group=s.group, split=s.split, dataset=dataset, family=fam,
                    modifier=mod, prompt=prompt, target_idx=tgt, strength=st,
                    source_image_id=m["image_id"] if m else None, source_run=args.source if m else None, **common)

    # ------------------------------------------------------------------ search methods
    if method in ("best_of_n", "img2img_bon"):
        from biofool.scoring import Scorer
        from biofool.search import best_of_n
        scorer = Scorer(cfg, species)
        b = cfg.best_of_n
        for j, (s, fam, mod, prompt, tgt, m, st) in enumerate(jobs):
            seeds = [rng.randrange(2**31) for _ in range(b.n)]  # drawn even when skipped: keeps the seeds identical
            if j in done_jobs:
                continue
            records, out = best_of_n(gen, scorer, prompt, s.idx, seeds, tgt, prune_at=b.prune_at,
                                     keep=b.keep if b.prune_at else b.n, init_image=load_source(m), strength=st)
            best = max(range(len(records)), key=lambda i: records[i]["reward"])
            for i, (rec, im) in enumerate(zip(records, out.images)):
                writer.add(im, **meta(s, fam, mod, prompt, tgt, m, st), job=j, seed=rec["seed"], is_best=i == best,
                           bon_n=b.n, prune_at=b.prune_at, unet_evals_per_prompt=out.unet_evals,
                           seconds_per_prompt=out.seconds, search_reward=rec["reward"],
                           early_reward=rec["early_reward"])
            print(f"[{j + 1}/{len(jobs)}] {s.latin:28s} best reward {records[best]['reward']:.3f} "
                  f"({out.unet_evals} UNet evals, {out.seconds:.1f}s)")
    # ------------------------------------------------------------------ plain sampling
    else:
        flat = [(job, rng.randrange(2**31)) for job in jobs for _ in range(n if not method.startswith("img2img") else 1)]
        # img2img: one sample per (photo, modifier, strength); batches must share the strength
        flat.sort(key=lambda x: x[0][6])
        i = len(previous)  # images are written in this exact order: continue after the last one
        while i < len(flat):
            st = flat[i][0][6]
            chunk = [x for x in flat[i:i + args.batch] if x[0][6] == st]
            inits = [load_source(job[5]) for job, _ in chunk] if method.startswith("img2img") else None
            out = gen.sample([job[3] for job, _ in chunk], [seed for _, seed in chunk], init_images=inits, strength=st)
            for (job, seed), im in zip(chunk, out.images):
                writer.add(im, **meta(*job), seed=seed, seconds_per_image=out.seconds / len(chunk))
            i += len(chunk)
            print(f"{i}/{len(flat)} images ({out.seconds / len(chunk):.2f} s/image)")
    writer.close()
    finish_run(args.out)


if __name__ == "__main__":
    main()