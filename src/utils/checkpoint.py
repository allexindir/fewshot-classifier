import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    config: Dict[str, Any],
    seed: int,
    val_acc: float,
    epoch: int,
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "seed": seed,
            "val_acc": val_acc,
            "epoch": epoch,
        },
        path,
    )


def load_checkpoint(path: str, device: str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)
