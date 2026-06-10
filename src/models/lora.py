import logging
from typing import List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def apply_lora(
    backbone: nn.Module,
    r: int = 8,
    alpha: int = 16,
    target_blocks: int = 4,
) -> Tuple[nn.Module, List[torch.nn.Parameter]]:
    from peft import LoraConfig, get_peft_model

    inner_model = backbone.model if hasattr(backbone, "model") else backbone

    target_modules = _find_target_modules(inner_model, target_blocks)

    if not target_modules:
        raise ValueError(
            "LoRA: could not find attention projection modules"
        )

    logger.info(f"LoRA target modules ({len(target_modules)}): {target_modules[:4]}")

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
    )

    for param in inner_model.parameters():
        param.requires_grad = True

    peft_model = get_peft_model(inner_model, config)

    lora_params = []
    for name, param in peft_model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False

    backbone.model = peft_model

    logger.info(
        f"LoRA applied: r={r}, alpha={alpha}, "
        f"trainable params: {sum(p.numel() for p in lora_params):,}"
    )

    return backbone, lora_params


def _find_target_modules(model: nn.Module, target_blocks: int) -> List[str]:
    all_names = [name for name, _ in model.named_modules()]

    patterns = [
        ("attn.qkv",),
        ("attn.q_proj", "attn.v_proj"),
        ("attention.query", "attention.value"),
    ]

    block_names = []
    for name in all_names:
        for prefix in ("blocks.", "encoder.layer."):
            if prefix in name:
                parts = name.split(prefix)
                if len(parts) > 1:
                    block_idx = parts[1].split(".")[0]
                    if block_idx.isdigit():
                        block_names.append(int(block_idx))

    if not block_names:
        return []

    max_block = max(block_names)
    target_block_indices = set(range(max_block - target_blocks + 1, max_block + 1))

    target_modules = []
    for pat_group in patterns:
        matches = []
        for name in all_names:
            for pat in pat_group:
                if pat in name:
                    for prefix in ("blocks.", "encoder.layer."):
                        if prefix in name:
                            block_idx = int(name.split(prefix)[1].split(".")[0])
                            if block_idx in target_block_indices:
                                matches.append(name)
        if matches:
            target_modules = matches
            break

    return target_modules
