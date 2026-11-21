"""One DDIM step that also returns log p(x_{t-1} | x_t) under the policy.

DDPO (Black et al., 2023) views the denoising chain as a policy: with eta > 0 every DDIM step samples
x_{t-1} ~ N(mu_theta(x_t, t), sigma_t^2 I), so its log-probability is differentiable in the UNet parameters.
The formulas follow diffusers' DDIMScheduler.step (tests/test_ddim.py checks that they match exactly).
"""

from __future__ import annotations

import math

import torch


def _alpha(scheduler, t: int, device) -> torch.Tensor:
    if t >= 0:
        return scheduler.alphas_cumprod[t].to(device)
    return scheduler.final_alpha_cumprod.to(device)


def ddim_step_with_logprob(scheduler, eps: torch.Tensor, t: int, sample: torch.Tensor, eta: float,
                           prev_sample: torch.Tensor | None = None, noise: torch.Tensor | None = None):
    """Returns (prev_sample, log_prob [batch], pred_original_sample).

    eps          the (guided) noise prediction of the UNet, epsilon parameterisation
    prev_sample  if given, the log-probability of this sample is computed (training); else one is drawn
    noise        standard normal noise used to draw prev_sample (else torch.randn)
    """
    assert scheduler.config.prediction_type == "epsilon", "only epsilon-prediction models are supported"
    t = int(t)
    prev_t = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    alpha_t, alpha_prev = _alpha(scheduler, t, eps.device), _alpha(scheduler, prev_t, eps.device)
    beta_t, beta_prev = 1 - alpha_t, 1 - alpha_prev

    sample32, eps32 = sample.float(), eps.float()
    x0 = (sample32 - beta_t.sqrt() * eps32) / alpha_t.sqrt()
    if scheduler.config.clip_sample:
        x0 = x0.clamp(-scheduler.config.clip_sample_range, scheduler.config.clip_sample_range)
        eps32 = (sample32 - alpha_t.sqrt() * x0) / beta_t.sqrt()
    variance = (beta_prev / beta_t) * (1 - alpha_t / alpha_prev)
    std = eta * variance.sqrt()
    mean = alpha_prev.sqrt() * x0 + (1 - alpha_prev - std ** 2).clamp(min=0).sqrt() * eps32

    if prev_sample is None:
        if noise is None:
            noise = torch.randn_like(mean)
        prev_sample = mean + std * noise.float()
    prev32 = prev_sample.detach().float()
    std_safe = std.clamp(min=1e-8)
    log_prob = (-((prev32 - mean) ** 2) / (2 * std_safe ** 2) - torch.log(std_safe) - 0.5 * math.log(2 * math.pi))
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return prev_sample.to(sample.dtype), log_prob, x0.to(sample.dtype)
