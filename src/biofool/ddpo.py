"""DDPO with group-relative advantages (GRPO for diffusion), on a LoRA of the Stable Diffusion UNet.

One epoch:
  1. pick P prompts (species of split A + a random context modifier) and sample G images per prompt with the
     current policy, keeping every latent x_T..x_0 and every log p(x_{t-1}|x_t)          (rollout, no gradient)
  2. reward each image (attack reward, see reward.py) and normalise the rewards within each group of G:
     A_i = (r_i - mean_G) / std_G                                                     (exactly Lab 4's GRPO)
  3. for a random subset of the denoising steps, recompute log p_theta(x_{t-1}|x_t) and minimise the clipped
     PPO surrogate  max(-A * ratio, -A * clip(ratio, 1-eps, 1+eps)),  ratio = p_theta / p_old.
There is no KL term: the LoRA rank, the small learning rate and the fidelity factor in the reward are what keep
the policy close to the base model. The ablation configs/experiments/ddpo_no_fidelity.yaml shows what happens
without the fidelity factor (reward hacking, as in Lab 4 Exercise 2b).
"""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path

import torch

from .ddim import ddim_step_with_logprob
from .lora import add_lora, grad_scaler, save_lora
from .prompts import all_modifiers, baseline_prompt
from .reward import group_advantages


def make_prompts(species, n: int, rng: random.Random, split: str = "A"):
    pool = [s for s in species if s.split == split]
    out = []
    for s in rng.sample(pool, n) if n <= len(pool) else [rng.choice(pool) for _ in range(n)]:
        fam, mod = rng.choice([("none", "")] + all_modifiers(s))
        out.append((s.idx, f"{baseline_prompt(s)}, {mod}" if mod else baseline_prompt(s), fam))
    return out


def train_ddpo(gen, scorer, species, cfg, out_dir: Path, log=print):
    from diffusers import StableDiffusionPipeline

    d = cfg.ddpo
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    unet, device = gen.pipe.unet, gen.device
    params = add_lora(unet, d.lora_rank)
    if d.get("gradient_checkpointing"):
        unet.enable_gradient_checkpointing()
    opt = torch.optim.AdamW(params, lr=d.lr, weight_decay=1e-4)
    scaler = grad_scaler(device, gen.dtype)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [n for n, p in unet.named_parameters() if p.requires_grad]

    # ---- resume after an interruption (Colab): LoRA weights, optimizer, scaler, epoch and RNG states
    start, last = 0, out_dir / "last.pt"
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        with torch.no_grad():
            for n, p in unet.named_parameters():
                if n in state["lora"]:
                    p.copy_(state["lora"][n])
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        rng.setstate(state["rng"])
        torch.set_rng_state(state["torch_rng"])
        start = state["epoch"]
        log(f"resuming DDPO from epoch {start}")
    if start >= d.epochs:
        log("DDPO already finished")
        return out_dir
    log_file = open(out_dir / "train_log.csv", "a" if start else "w", newline="")
    writer = None
    P, G = d.prompts_per_epoch, d.group_size

    for epoch in range(start, d.epochs):
        t_start = time.perf_counter()
        # ---- 1. rollout
        unet.eval()
        batch = make_prompts(species, P, rng)
        prompts = [p for _, p, _ in batch for _ in range(G)]
        true_idx = [i for i, _, _ in batch for _ in range(G)]
        seeds = [rng.randrange(2**31) for _ in prompts]
        roll = gen.sample(prompts, seeds, steps=d.steps, return_trajectory=True)
        # ---- 2. rewards and group advantages
        s = scorer(roll.images, true_idx)
        adv = group_advantages(s["reward"].view(P, G)).view(-1).clamp(-d.adv_clip, d.adv_clip).to(device)
        old_logp = torch.stack(roll.log_probs, dim=1)                  # [B, T]
        lat = torch.stack(roll.latents, dim=1)                         # [B, T+1, C, H, W]
        pos, neg = gen.encode(prompts)
        # ---- 3. PPO update on a random subset of the steps
        unet.train()
        n_steps = len(roll.timesteps)
        train_steps = sorted(rng.sample(range(n_steps), max(1, int(d.timestep_fraction * n_steps))))
        order = torch.randperm(len(prompts))
        n_mb = (len(prompts) + d.train_batch_size - 1) // d.train_batch_size
        opt.zero_grad()
        stats = {"approx_kl": 0.0, "clipfrac": 0.0}
        for j in train_steps:
            t = roll.timesteps[j]
            for mb in order.split(d.train_batch_size):
                with torch.autocast(device.split(":")[0], dtype=gen.dtype, enabled=device.startswith("cuda")):
                    eps = gen.predict_eps(lat[mb, j], torch.tensor(t, device=device), pos[mb], neg[mb],
                                          cfg.generator.guidance_scale, unet=unet)
                _, logp, _ = ddim_step_with_logprob(gen.pipe.scheduler, eps.float(), t, lat[mb, j].float(),
                                                    cfg.generator.eta, prev_sample=lat[mb, j + 1].float())
                ratio = (logp - old_logp[mb, j]).exp()
                a = adv[mb]
                loss = torch.max(-a * ratio, -a * ratio.clamp(1 - d.clip_range, 1 + d.clip_range)).mean()
                scaler.scale(loss / (len(train_steps) * n_mb)).backward()
                stats["approx_kl"] += 0.5 * ((logp - old_logp[mb, j]) ** 2).mean().item() / (len(train_steps) * n_mb)
                stats["clipfrac"] += ((ratio - 1).abs() > d.clip_range).float().mean().item() / (len(train_steps) * n_mb)
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, d.max_grad_norm).item()
        scaler.step(opt)
        scaler.update()
        # ---- logging
        fooled = (s["target_shared_ranks"] < 7).float()
        faithful = (s["p_oracle_true"] >= cfg.reward.tau).float()
        row = dict(epoch=epoch, reward=s["reward"].mean().item(), fool_rate=fooled.mean().item(),
                   reward_oracle_faithful=faithful.mean().item(), train_asr=(fooled * faithful).mean().item(),
                   p_target_true=s["p_target_true"].mean().item(), grad_norm=grad_norm,
                   seconds=time.perf_counter() - t_start, **stats)
        if writer is None:
            writer = csv.DictWriter(log_file, fieldnames=list(row))
            if not start:
                writer.writeheader()
        writer.writerow(row)
        log_file.flush()
        log(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        if (epoch + 1) % d.save_every == 0 or epoch == d.epochs - 1:
            save_lora(StableDiffusionPipeline, unet, str(out_dir / f"lora_epoch{epoch + 1:04d}"))
            roll.images[0].save(out_dir / f"sample_epoch{epoch + 1:04d}.png")
            lora = {n: p.detach().cpu() for n, p in unet.named_parameters() if n in names}
            torch.save(dict(lora=lora, opt=opt.state_dict(), scaler=scaler.state_dict(), rng=rng.getstate(),
                            torch_rng=torch.get_rng_state(), epoch=epoch + 1), last)
    log_file.close()
    unet.eval()
    save_lora(StableDiffusionPipeline, unet, str(out_dir / "lora_final"))
    return out_dir
