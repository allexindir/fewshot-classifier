import os

import torch


def get_device() -> str:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
