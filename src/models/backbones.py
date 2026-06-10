import logging
from abc import ABC, abstractmethod

import timm
import torch
import torch.nn as nn

from transformers import AutoModel

logger = logging.getLogger(__name__)


class BackboneWrapper(ABC, nn.Module):
    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        ...


class TimmBackbone(BackboneWrapper):
    def __init__(self, model: nn.Module, dim: int):
        super().__init__()
        self.model = model
        self._embed_dim = dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model.forward_features(x)
        if features.dim() == 3:
            features = features[:, 0]
        return features

    @property
    def embed_dim(self) -> int:
        return self._embed_dim


def load_backbone(
    name: str,
    pretrained: bool = True,
    freeze: bool = True,
    model_name: str = None,
) -> BackboneWrapper:
    """load a pretrained backbone by name:
        - "dinov3_vitl16": DINOv3 ViT-L/16
        - "siglip2": SigLIP2 vision encoder 
    """
    if name == "dinov3_vitl16":
        backbone = _load_dinov3_vitl16(pretrained)
    elif name == "siglip2":
        backbone = _load_siglip2(model_name or "google/siglip2-so400m-patch14-384")
    else:
        raise ValueError(f"Unknown backbone: {name}")

    if freeze:
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval()
        logger.info(f"Backbone {name} frozen ({sum(p.numel() for p in backbone.parameters()):.1e} params)")

    return backbone


def _load_dinov3_vitl16(pretrained: bool) -> TimmBackbone:
    # Load DINOv3 ViT-L/16 from timm.
    model = timm.create_model(
        "vit_large_patch16_dinov3.lvd1689m", pretrained=pretrained, num_classes=0
    )
    dim = model.num_features
    logger.info(f"Loaded DINOv3 ViT-L/16 from timm (embed_dim={dim})")
    return TimmBackbone(model, dim)

class Siglip2Backbone(BackboneWrapper):
    def __init__(self, model_name: str = "google/siglip2-so400m-patch14-384"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self._embed_dim = self.model.config.vision_config.hidden_size

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.get_image_features(pixel_values=x)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim


def _load_siglip2(model_name: str) -> Siglip2Backbone:
    backbone = Siglip2Backbone(model_name)
    logger.info(f"Loaded SigLIP2 '{model_name}' (embed_dim={backbone.embed_dim})")
    return backbone
