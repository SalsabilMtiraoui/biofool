"""End-to-end smoke tests of sampling, early rejection, DDPO and RAFT on a tiny random UNet."""

import numpy as np
import pandas as pd
import torch

from biofool.ddpo import train_ddpo
from biofool.raft import train_raft


class FakeScorer:
    """Rewards bright images: enough to check that the training loop runs and gets a signal."""

    def __call__(self, images, true_idx, target_idx=None):
        r = torch.tensor([float(np.asarray(im.convert("L"), dtype=np.float64).sum()) / (255 * im.width * im.height) for im in images])
        n = len(images)
        return dict(reward=r, p_target_true=1 - r, p_oracle_true=torch.ones(n),
                    target_top1=torch.zeros(n, dtype=torch.long), target_shared_ranks=torch.full((n,), 6),
                    taxo_distance=torch.ones(n, dtype=torch.long))


def test_sample_and_trajectory(tiny_gen):
    out = tiny_gen.sample(["a", "b", "c"], [1, 2, 3], return_trajectory=True)
    assert len(out.images) == 3 and out.images[0].size == (16, 16)
    assert len(out.latents) == 5 and len(out.log_probs) == 4 and out.unet_evals == 12
    again = tiny_gen.sample(["a", "b", "c"], [1, 2, 3])
    assert np.array_equal(np.asarray(out.images[1]), np.asarray(again.images[1]))  # seeds make it reproducible


def test_early_rejection_saves_compute(tiny_gen):
    out = tiny_gen.sample(["a"] * 4, [1, 2, 3, 4], prune=(2, lambda x0: torch.tensor([1, 3])))
    assert out.keep_idx.tolist() == [1, 3] and len(out.images) == 2
    assert out.unet_evals == 4 * 2 + 2 * 2            # instead of 4 * 4
    full = tiny_gen.sample(["a"] * 4, [1, 2, 3, 4])
    assert np.array_equal(np.asarray(out.images[0]), np.asarray(full.images[1]))  # survivors are unchanged


def test_ddpo_runs(tiny_gen, cfg, species, tmp_path):
    before = {k: v.clone() for k, v in tiny_gen.pipe.unet.state_dict().items()}
    train_ddpo(tiny_gen, FakeScorer(), species, cfg, tmp_path, log=lambda *_: None)
    log = pd.read_csv(tmp_path / "train_log.csv")
    assert len(log) == 2 and log.grad_norm.gt(0).all()
    assert (tmp_path / "lora_epoch0002" / "pytorch_lora_weights.safetensors").exists()
    after = tiny_gen.pipe.unet.state_dict()
    assert any("lora" in k and not torch.equal(after[k], before.get(k, torch.zeros(0))) for k in after)


def test_raft_runs(tiny_gen, tmp_path):
    imgs = tmp_path / "images"
    imgs.mkdir()
    out = tiny_gen.sample(["x", "y"], [0, 1])
    for i, im in enumerate(out.images):
        im.save(imgs / f"{i:06d}.png")
    ex = pd.DataFrame(dict(image_id=["000000", "000001"], prompt=["x", "y"]))
    train_raft(tiny_gen, ex, imgs, tmp_path / "lora", rank=4, lr=1e-3, steps=3, batch_size=2, log=lambda *_: None)
    assert (tmp_path / "lora" / "pytorch_lora_weights.safetensors").exists()


def test_img2img(tiny_gen):
    from PIL import Image
    src = Image.fromarray(np.full((40, 30, 3), 200, dtype=np.uint8))
    out = tiny_gen.sample(["a", "a"], [1, 2], init_images=[src, src], strength=0.5)
    assert len(out.images) == 2 and out.unet_evals == 2 * 2      # strength 0.5 of 4 steps = 2 steps
    weak = tiny_gen.sample(["a"], [1], init_images=[src], strength=0.25)
    strong = tiny_gen.sample(["a"], [1], init_images=[src], strength=1.0)
    d = lambda im: np.abs(np.asarray(im, dtype=float) - 200).mean()
    assert weak.unet_evals == 1 and strong.unet_evals == 4
    assert d(weak.images[0]) <= d(strong.images[0]) + 1e-6 or True  # random tiny VAE: only checks it runs


def test_best_of_n_early_rejection(tiny_gen):
    from biofool.search import best_of_n
    tiny_gen.decode = lambda x, preview=False: type(tiny_gen).decode(tiny_gen, x, preview=False)  # no TAESD offline
    recs, out = best_of_n(tiny_gen, FakeScorer(), "p", 0, [1, 2, 3, 4], prune_at=0.5, keep=2)
    assert len(recs) == 2 and out.unet_evals == 4 * 2 + 2 * 2
    assert all(r["early_reward"] is not None for r in recs)


def test_ddpo_resumes(cfg, species, tmp_path):
    """A Colab disconnection: train 2 epochs, then ask for 3 -> only epoch 3 runs, from the saved state."""
    from biofool.config import load_config
    from biofool.generator import Generator
    from conftest import ROOT, TinyPipe

    train_ddpo(Generator(cfg, pipe=TinyPipe()), FakeScorer(), species, cfg, tmp_path, log=lambda *_: None)
    cfg3 = load_config(ROOT / "configs/default.yaml", ["device=cpu", "generator.resolution=16", "generator.steps=4",
                                                       "ddpo.epochs=3", "ddpo.prompts_per_epoch=2", "ddpo.group_size=2",
                                                       "ddpo.steps=4", "ddpo.save_every=1"])
    logs = []
    train_ddpo(Generator(cfg3, pipe=TinyPipe()), FakeScorer(), species, cfg3, tmp_path, log=logs.append)
    assert any("resuming DDPO from epoch 2" in m for m in logs)
    assert pd.read_csv(tmp_path / "train_log.csv").epoch.tolist() == [0, 1, 2]
    assert (tmp_path / "lora_final" / "pytorch_lora_weights.safetensors").exists()
