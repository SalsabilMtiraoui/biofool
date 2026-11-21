"""Tiny, randomly initialised Stable Diffusion components: the tests run on a CPU, offline, in seconds."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class TinyPipe:
    """Duck-typed stand-in for StableDiffusionPipeline (no tokenizer download needed)."""

    def __init__(self):
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel

        torch.manual_seed(0)
        self.unet = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
                                         block_out_channels=(32, 32), down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                                         up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=16,
                                         attention_head_dim=8, norm_num_groups=8)
        self.vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4, block_out_channels=(8, 8),
                                 down_block_types=("DownEncoderBlock2D",) * 2, up_block_types=("UpDecoderBlock2D",) * 2,
                                 norm_num_groups=8)
        self.scheduler = DDIMScheduler(beta_schedule="scaled_linear", beta_start=0.00085, beta_end=0.012,
                                       clip_sample=False, set_alpha_to_one=False, steps_offset=1)
        self.vae_scale_factor = 2

    def encode_prompt(self, prompt, device, n, cfg, negative_prompt=None):
        prompts = [prompt] if isinstance(prompt, str) else prompt
        emb = torch.stack([torch.randn(5, 16, generator=torch.Generator().manual_seed(hash(p) % 2**31))
                           for p in prompts])
        return emb, torch.zeros_like(emb)

    def set_progress_bar_config(self, **_):
        pass

    def to(self, device):
        self.unet.to(device)
        self.vae.to(device)
        return self


@pytest.fixture
def cfg():
    from biofool.config import load_config

    return load_config(ROOT / "configs/default.yaml", ["device=cpu", "generator.resolution=16", "generator.steps=4",
                                                       "ddpo.epochs=2", "ddpo.prompts_per_epoch=2",
                                                       "ddpo.group_size=2", "ddpo.steps=4", "ddpo.save_every=1"])


@pytest.fixture
def species():
    from biofool.taxonomy import load_species

    return load_species(ROOT / "data/species.csv")


@pytest.fixture
def tiny_gen(cfg):
    from biofool.generator import Generator

    return Generator(cfg, pipe=TinyPipe())
