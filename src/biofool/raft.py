"""RAFT (reward-ranked fine-tuning): select the best attacks found by search, then fine-tune a LoRA on them
with the ordinary denoising loss. One round = search -> select -> SFT; several rounds can be chained by
sampling again with the new LoRA. Simple and stable: the safe version of the post-training step.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from .lora import add_lora, grad_scaler, save_lora


def select_examples(df: pd.DataFrame, mode: str = "top_per_species", top_k: int = 4, min_fidelity: float = 0.5,
                    quantile: float = 0.8, fidelity_col: str = "reward_oracle_p_true") -> pd.DataFrame:
    """Keep the images that fool the target AND are judged faithful by the REWARD oracle.
    (The evaluation oracle is never used here, so the reported fidelity stays an independent measurement.)"""
    ok = df[(df[fidelity_col] >= min_fidelity) & (df["target_fooled"])]
    if mode == "top_per_species":
        return ok.sort_values("reward", ascending=False).groupby("species_idx").head(top_k)
    if mode == "quantile":
        return ok[ok.reward >= ok.reward.quantile(quantile)]
    raise ValueError(mode)


def train_raft(gen, examples: pd.DataFrame, image_dir: Path, out_dir: Path, rank: int, lr: float, steps: int,
               batch_size: int, seed: int = 0, log=print, gradient_checkpointing: bool = False):
    from diffusers import DDPMScheduler, StableDiffusionPipeline

    torch.manual_seed(seed)
    rng = random.Random(seed)
    pipe, device = gen.pipe, gen.device
    unet = pipe.unet
    params = add_lora(unet, rank)
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    scaler = grad_scaler(device, gen.dtype)
    noise_sched = DDPMScheduler.from_config(pipe.scheduler.config)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)

    # Pre-encode the training images into latents and the prompts into text embeddings, once.
    rows = list(examples.itertuples())
    with torch.no_grad():
        latents, embeds = [], []
        for r in rows:
            im = Image.open(image_dir / f"{r.image_id}.png").convert("RGB").resize((gen.resolution,) * 2)
            x = torch.from_numpy(np.asarray(im, dtype=np.float32))
            x = (x.permute(2, 0, 1) / 127.5 - 1)[None].to(device, pipe.vae.dtype)
            latents.append(pipe.vae.encode(x).latent_dist.sample() * pipe.vae.config.scaling_factor)
            embeds.append(pipe.encode_prompt(r.prompt, device, 1, False)[0])
        latents, embeds = torch.cat(latents), torch.cat(embeds)
    log(f"RAFT: {len(rows)} examples, {sum(p.numel() for p in params) / 1e6:.2f} M trainable parameters")

    unet.train()
    for step in range(steps):
        idx = torch.tensor(rng.sample(range(len(rows)), min(batch_size, len(rows))))
        x0, emb = latents[idx], embeds[idx]
        noise = torch.randn_like(x0)
        t = torch.randint(0, noise_sched.config.num_train_timesteps, (len(idx),), device=device)
        xt = noise_sched.add_noise(x0, noise, t)
        with torch.autocast(device.split(":")[0], dtype=gen.dtype, enabled=device.startswith("cuda")):
            pred = unet(xt, t, encoder_hidden_states=emb).sample
        loss = F.mse_loss(pred.float(), noise.float())
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        for g in opt.param_groups:  # cosine decay
            g["lr"] = lr * 0.5 * (1 + math.cos(math.pi * step / steps))
        scaler.step(opt)
        scaler.update()
        if step % 100 == 0 or step == steps - 1:
            log(f"step {step:5d}  loss {loss.item():.4f}")
    unet.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_lora(StableDiffusionPipeline, unet, str(out_dir))
    return out_dir
