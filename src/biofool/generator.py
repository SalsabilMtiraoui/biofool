"""Stable Diffusion wrapper with our own DDIM sampling loop.

Our loop (`sample`) is needed for two things that the diffusers pipeline does not expose:
  * the log-probability of every denoising step (DDPO training, see ddim.py);
  * pruning candidates in the middle of the chain (early-rejection best-of-N, see search.py).
`generate_fast` calls the stock diffusers pipeline and is used for the inference-acceleration benchmark
(other schedulers, LCM-LoRA, fewer steps).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from PIL import Image

from .ddim import ddim_step_with_logprob

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


@dataclass
class SampleOutput:
    images: list[Image.Image]
    keep_idx: torch.Tensor                 # indices (into the initial batch) of the samples that survived pruning
    latents: list[torch.Tensor] = field(default_factory=list)    # x_T, ..., x_0  (only if return_trajectory)
    log_probs: list[torch.Tensor] = field(default_factory=list)  # one [batch] tensor per step
    timesteps: list[int] = field(default_factory=list)
    unet_evals: int = 0                    # number of (batch element x step) UNet evaluations: our compute unit
    seconds: float = 0.0
    final_latents: torch.Tensor | None = None


def square(im: Image.Image, size: int) -> Image.Image:
    """Center crop to a square and resize (img2img inputs)."""
    im = im.convert("RGB")
    side = min(im.size)
    left, top = (im.width - side) // 2, (im.height - side) // 2
    return im.crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.LANCZOS)


def latents_to_pil(x: torch.Tensor) -> list[Image.Image]:
    x = (x.float() / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray((im * 255).round().astype("uint8")) for im in x]


class Generator:
    def __init__(self, cfg, pipe=None):
        from diffusers import DDIMScheduler, StableDiffusionPipeline

        g = cfg.generator
        self.cfg, self.device = cfg, cfg.device
        self.dtype = DTYPES[g.dtype] if cfg.device.startswith("cuda") else torch.float32
        if pipe is None:
            pipe = StableDiffusionPipeline.from_pretrained(g.model_id, torch_dtype=self.dtype, safety_checker=None,
                                                           requires_safety_checker=False)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe.to(self.device)
        if g.get("lora_path"):
            self.pipe.load_lora_weights(g.lora_path)
        if g.get("compile"):
            self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
        self.preview_vae = None
        self.resolution = g.resolution

    # ------------------------------------------------------------------ prompts
    @torch.no_grad()
    def encode(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        neg = [self.cfg.generator.negative_prompt] * len(prompts)
        pos, negative = self.pipe.encode_prompt(prompts, self.device, 1, True, negative_prompt=neg)
        return pos, negative

    def initial_latents(self, seeds: list[int]) -> torch.Tensor:
        c = self.pipe.unet.config.in_channels
        h = self.resolution // self.pipe.vae_scale_factor
        x = torch.stack([torch.randn(c, h, h, generator=torch.Generator().manual_seed(int(s))) for s in seeds])
        return x.to(self.device, self.dtype)

    # ------------------------------------------------------------------ sampling
    def predict_eps(self, latents, t, pos, neg, guidance: float, unet=None) -> torch.Tensor:
        unet = unet or self.pipe.unet
        x = torch.cat([latents, latents])
        emb = torch.cat([neg, pos])
        eps_uncond, eps_cond = unet(x, t, encoder_hidden_states=emb).sample.chunk(2)
        return eps_uncond + guidance * (eps_cond - eps_uncond)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, preview: bool = False) -> list[Image.Image]:
        if preview:
            if self.preview_vae is None:
                from diffusers import AutoencoderTiny
                self.preview_vae = AutoencoderTiny.from_pretrained(self.cfg.generator.preview_vae,
                                                                   torch_dtype=self.dtype).to(self.device)
            vae = self.preview_vae
        else:
            vae = self.pipe.vae
        return latents_to_pil(vae.decode(latents.to(vae.dtype) / vae.config.scaling_factor).sample)

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        """Pixels -> SD latents (posterior mean), for img2img."""
        x = torch.stack([torch.from_numpy(np.asarray(square(im, self.resolution), dtype=np.float32)) for im in images])
        x = (x.permute(0, 3, 1, 2) / 127.5 - 1).to(self.device, self.pipe.vae.dtype)
        return self.pipe.vae.encode(x).latent_dist.mean * self.pipe.vae.config.scaling_factor

    @torch.no_grad()
    def sample(self, prompts: list[str], seeds: list[int], steps: int | None = None, guidance: float | None = None,
               eta: float | None = None, return_trajectory: bool = False,
               prune: tuple[int, Callable[[torch.Tensor], torch.Tensor]] | None = None,
               init_images: list[Image.Image] | None = None, strength: float = 1.0) -> SampleOutput:
        """DDIM sampling, one image per (prompt, seed).

        prune = (step_index, fn): after `step_index` steps, fn(x0_prediction_latents) returns the indices of the
        samples to keep; the others are dropped and cost no more UNet evaluations.
        init_images + strength: img2img (SDEdit). The real photo is noised up to the timestep at `strength` of the
        chain and denoised from there with the attack prompt: low strength keeps the organism, high strength
        changes more. Only round(strength * steps) steps are run.
        """
        g = self.cfg.generator
        steps, guidance, eta = steps or g.steps, guidance or g.guidance_scale, g.eta if eta is None else eta
        t0 = time.perf_counter()
        sched = self.pipe.scheduler
        sched.set_timesteps(steps, device=self.device)
        timesteps = sched.timesteps
        pos, neg = self.encode(prompts)
        latents = self.initial_latents(seeds) * sched.init_noise_sigma
        if init_images is not None:
            n_run = max(1, min(steps, round(strength * steps)))
            timesteps = timesteps[steps - n_run:]
            x0 = self.encode_images(init_images).to(latents.dtype)
            latents = sched.add_noise(x0, latents, timesteps[:1].repeat(len(x0)))
        noise_gens = [torch.Generator().manual_seed(int(s) + 1_000_003) for s in seeds]
        keep = torch.arange(len(prompts))
        out = SampleOutput(images=[], keep_idx=keep)
        if return_trajectory:
            out.latents.append(latents)
        for i, t in enumerate(timesteps):
            eps = self.predict_eps(latents, t, pos, neg, guidance)
            noise = torch.stack([torch.randn(latents.shape[1:], generator=gen) for gen in noise_gens])
            latents, log_prob, x0 = ddim_step_with_logprob(sched, eps, int(t), latents, eta,
                                                           noise=noise.to(self.device))
            out.unet_evals += len(latents)
            if return_trajectory:
                out.latents.append(latents)
                out.log_probs.append(log_prob)
                out.timesteps.append(int(t))
            if prune is not None and i + 1 == prune[0]:
                sel = prune[1](x0).cpu()
                latents, pos, neg, keep = latents[sel], pos[sel], neg[sel], keep[sel]
                noise_gens = [noise_gens[j] for j in sel.tolist()]
        out.images = self.decode(latents)
        out.final_latents = latents
        out.keep_idx = keep
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        out.seconds = time.perf_counter() - t0
        return out

    # ------------------------------------------------------------------ stock pipeline (acceleration study)
    @torch.no_grad()
    def generate_fast(self, prompts: list[str], seeds: list[int], steps: int, guidance: float) -> list[Image.Image]:
        gens = [torch.Generator(self.device).manual_seed(int(s)) for s in seeds]
        neg = [self.cfg.generator.negative_prompt] * len(prompts) if guidance > 1 else None
        return self.pipe(prompts, negative_prompt=neg, num_inference_steps=steps, guidance_scale=guidance,
                         generator=gens, height=self.resolution, width=self.resolution).images


def set_fast_mode(gen: Generator, mode: str) -> tuple[int, float]:
    """Switch the pipeline to an acceleration mode. Returns the (steps, guidance) that mode is meant for.

    ddim       DDIM, as trained                        dpm        DPM-Solver++ (2M Karras)
    lcm        LCM-LoRA (4 steps, guidance 1)          *_taesd    the same, decoded with the tiny VAE
    """
    from diffusers import AutoencoderTiny, DDIMScheduler, DPMSolverMultistepScheduler, LCMScheduler

    pipe = gen.pipe
    base = mode.replace("_taesd", "")
    if base == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        setting = (gen.cfg.generator.steps, gen.cfg.generator.guidance_scale)
    elif base == "dpm":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True,
                                                                 algorithm_type="dpmsolver++")
        setting = (12, gen.cfg.generator.guidance_scale)
    elif base == "lcm":
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        try:
            active = pipe.get_active_adapters()
        except (ValueError, AttributeError):
            active = []
        if "lcm" not in active:
            pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5", adapter_name="lcm")
            if active:  # an attack LoRA is loaded too: keep both
                pipe.set_adapters(active + ["lcm"])
        setting = (4, 1.0)
    else:
        raise ValueError(mode)
    if mode.endswith("_taesd"):
        pipe.vae = AutoencoderTiny.from_pretrained(gen.cfg.generator.preview_vae, torch_dtype=gen.dtype).to(gen.device)
    return setting
