"""LoRA helpers shared by RAFT and DDPO."""

from __future__ import annotations

import torch


def add_lora(unet, rank: int, train_dtype=torch.float32):
    """Freezes the UNet and adds trainable LoRA adapters on the attention projections."""
    from peft import LoraConfig

    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian",
                                target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    params = [p for p in unet.parameters() if p.requires_grad]
    for p in params:  # trainable weights in fp32 even when the frozen UNet is fp16 (as in diffusers' LoRA scripts)
        p.data = p.data.to(train_dtype)
    return params


def grad_scaler(device: str, dtype):
    """fp16 training (T4, RTX 20xx have no bf16): scale the loss so that small gradients do not underflow."""
    enabled = device.startswith("cuda") and dtype == torch.float16
    return torch.amp.GradScaler("cuda", enabled=enabled)


def save_lora(pipe_cls, unet, path: str):
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    pipe_cls.save_lora_weights(path, unet_lora_layers=layers)
