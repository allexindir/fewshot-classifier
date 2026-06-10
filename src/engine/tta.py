from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


@torch.no_grad()
def predict_with_tta(
    model: nn.Module,
    image: Image.Image,
    tta_transforms: List[transforms.Compose],
    device: str,
) -> torch.Tensor:
    model.eval()
    probs_sum = None

    for tfm in tta_transforms:
        img_tensor = tfm(image).unsqueeze(0).to(device)
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

        if probs_sum is None:
            probs_sum = probs
        else:
            probs_sum = probs_sum + probs

    return (probs_sum / len(tta_transforms)).cpu()
