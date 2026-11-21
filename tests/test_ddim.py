import torch
from diffusers import DDIMScheduler

from biofool.ddim import ddim_step_with_logprob


def make_scheduler(steps=10):
    s = DDIMScheduler(beta_schedule="scaled_linear", beta_start=0.00085, beta_end=0.012, clip_sample=False,
                      set_alpha_to_one=False, steps_offset=1)
    s.set_timesteps(steps)
    return s


def test_matches_diffusers():
    """Our step must give exactly diffusers' DDIM sample, for eta = 0 and eta = 1, at every timestep."""
    torch.manual_seed(0)
    for eta in (0.0, 1.0):
        sched = make_scheduler()
        x = torch.randn(2, 4, 8, 8)
        for t in sched.timesteps:
            eps, noise = torch.randn_like(x), torch.randn_like(x)
            ref = sched.step(eps, t, x, eta=eta, variance_noise=noise).prev_sample
            ours, logp, _ = ddim_step_with_logprob(sched, eps, int(t), x, eta, noise=noise)
            assert torch.allclose(ours, ref, atol=1e-5), (eta, int(t))
            assert logp.shape == (2,)
            x = ref


def test_logprob_is_maximal_at_the_mean_and_differentiable():
    sched = make_scheduler()
    x = torch.randn(1, 4, 8, 8)
    t = int(sched.timesteps[3])
    eps = torch.randn_like(x, requires_grad=True)
    mean, lp_mean, _ = ddim_step_with_logprob(sched, eps, t, x, 1.0, noise=torch.zeros_like(x))
    _, lp_far, _ = ddim_step_with_logprob(sched, eps, t, x, 1.0, prev_sample=mean + 0.5)
    assert lp_mean.item() > lp_far.item()
    lp_far.sum().backward()
    assert eps.grad is not None and eps.grad.abs().sum() > 0
