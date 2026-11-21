"""Integration test of the scoring and analysis scripts, with randomly initialised CLIP models (offline)."""

import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image

from biofool.classifiers import ZeroShotCLIP
from biofool.io import RunWriter

RANDOM = ["--set", "classifiers.bioclip.model_id=ViT-B-32/none", "classifiers.bioclip2.model_id=ViT-B-32/none",
          "classifiers.bioclip.templates=single", "classifiers.bioclip2.templates=single",
          "roles.target=bioclip", "roles.transfer=[bioclip2]", "roles.eval_oracle=bioclip2", "device=cpu"]


def test_zero_shot_clip(species, tmp_path):
    clf = ZeroShotCLIP("t", "ViT-B-32/none", species[:10], device="cpu", cache_dir=tmp_path, templates="single")
    ims = [Image.new("RGB", (64, 64), "red"), Image.new("RGB", (64, 64), "blue")]
    p = clf.probs(ims)
    assert p.shape == (2, 10) and np.allclose(p.sum(-1).numpy(), 1, atol=1e-4)
    pred = clf.predict(ims, [0, 1], [1, 0])
    assert set(pred) == {"p_true", "p_target", "top1", "top1_name", "top1_prob", "shared_ranks", "entropy"}
    assert pred["shared_ranks"].max() <= 7 and len(pred["top1_name"]) == 2
    assert len(list(tmp_path.glob("text_t_*.pt"))) == 1          # text embeddings cached on disk
    assert all(species[i].group == species[0].group for i in clf.nearest_species(0, 3))


def test_score_and_analyze(tmp_path, species):
    run = tmp_path / "run"
    w = RunWriter(run, species)
    rng = np.random.default_rng(0)
    for s in species[::10]:
        for method in ("baseline", "templates"):
            im = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
            w.add(im, species_idx=s.idx, latin=s.latin, group=s.group, split=s.split, family="none", modifier="",
                  prompt="p", target_idx=None, seed=0, method=method, dataset="synth80")
    w.close()
    py = sys.executable
    subprocess.run([py, "scripts/03_score.py", "--runs", str(run), *RANDOM, f"cache_dir={tmp_path}/c"], check=True)
    df = pd.read_csv(run / "scores.csv")
    assert {"bioclip_top1", "bioclip_p_true", "bioclip_shared_ranks", "bioclip2_p_true", "reward",
            "target_fooled"} <= set(df.columns)
    assert (run / "species.csv").exists()
    out = tmp_path / "results"
    subprocess.run([py, "scripts/08_analyze.py", "--runs", str(run), "--out", str(out), "--n-boot", "20", *RANDOM],
                   check=True)
    assert (out / "tables" / "main.tex").exists() and (out / "figures" / "transfer_heatmap.png").exists()


def test_tree_of_life_open_set(species, tmp_path, monkeypatch):
    """The open-set target, with a fake 6-taxon TreeOfLife (no download): p_true, top-1 outside our list, ranks."""
    import json

    import huggingface_hub
    import torch

    from biofool.classifiers import TreeOfLifeClassifier

    names = [[list(s.lineage()[:6]) + [s.epithet], s.common_name] for s in species[:4]]
    names += [[["Animalia", "Arthropoda", "Insecta", "Lepidoptera", "Nymphalidae", "Fakeus", "imaginarius"], "fake"],
              [["Plantae", "Tracheophyta", "Magnoliopsida", "Rosales", "Rosaceae", "Malus", "domestica"], "apple"]]
    emb = torch.nn.functional.normalize(torch.randn(512, len(names)), dim=0).numpy()
    np.save(tmp_path / "e.npy", emb)
    (tmp_path / "n.json").write_text(json.dumps(names))
    files = {"embeddings/txt_emb_species.npy": tmp_path / "e.npy", "embeddings/txt_emb_species.json": tmp_path / "n.json"}
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, f, repo_type=None: str(files[f]))
    clf = TreeOfLifeClassifier("tol", "ViT-B-32/none", species[:10], device="cpu")
    ims = [Image.new("RGB", (64, 64), c) for c in ("red", "green", "blue")]
    pred = clf.predict(ims, [0, 1, 7])
    assert np.isnan(pred["p_true"][2].item())                 # species 7 is not in the fake tree
    assert all(0 <= p <= 1 for p in pred["p_true"][:2].tolist())
    for name, top, shared in zip(pred["top1_name"], pred["top1"].tolist(), pred["shared_ranks"].tolist()):
        assert (top == -1) == (name in ("Fakeus imaginarius", "Malus domestica"))
        assert 0 <= shared <= 7


def test_analyze_all_branches(tmp_path, species):
    """08_analyze on hand-made scores: real photos, img2img strengths, fast modes, benchmark and early-rejection csv."""
    rng = np.random.default_rng(0)
    rows = []
    for method, ds, strength in [("real", "inat80", None), ("baseline", "synth80", None), ("templates", "synth80", None),
                                 ("img2img", "inat80", 0.3), ("img2img", "inat80", 0.7), ("fast:lcm", "synth80", None)]:
        for s in species[:12]:
            for k in range(3):
                shared = int(rng.choice([7, 7, 6, 4, 0]))
                rows.append(dict(image_id=f"{len(rows):06d}", species_idx=s.idx, latin=s.latin, group=s.group,
                                 dataset=ds, method=method, family="background", strength=strength,
                                 bioclip_tol_shared_ranks=shared, bioclip_tol_top1_prob=rng.random(),
                                 bioclip_tol_entropy=rng.random(), bioclip_shared_ranks=int(rng.choice([7, 5])),
                                 vlm_judge_p_true=rng.random(), bioclip2_p_true=rng.random()))
    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame(rows).to_csv(run / "scores.csv", index=False)
    pd.DataFrame([dict(mode="lcm", steps=4, guidance=1.0, dtype="float16", compile=False, seconds_per_image=0.3,
                       images_per_hour=12000, peak_vram_gib=3.1)]).to_csv(tmp_path / "bench.csv", index=False)
    pd.DataFrame([dict(latin=s.latin, group=s.group, seed=j, fraction=f, early_reward=rng.random(),
                       final_reward=rng.random()) for s in species[:5] for j in range(4) for f in (0.2, 0.4)]
                 ).to_csv(tmp_path / "early.csv", index=False)
    out = tmp_path / "results"
    subprocess.run([sys.executable, "scripts/08_analyze.py", "--runs", str(run), "--out", str(out), "--n-boot", "20",
                    "--bench", str(tmp_path / "bench.csv"), "--early", str(tmp_path / "early.csv"),
                    "--attackable-from", str(run), "--set", "roles.transfer=[bioclip]"], check=True)
    for name in ("reference", "main", "by_group", "transfer", "families", "img2img_strength", "speed",
                 "early_correlation"):
        assert (out / "tables" / f"{name}.tex").exists(), name
    for name in ("asr_by_method", "asr_by_group", "transfer_heatmap", "taxo_distance", "img2img_strength",
                 "pareto_speed", "early_correlation"):
        assert (out / "figures" / f"{name}.png").exists(), name
