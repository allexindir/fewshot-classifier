import logging
from typing import Any, Dict

import pandas as pd
import torch
from tqdm import tqdm

from src.data.dataset import TestImageDataset
from src.data.transforms import build_eval_transform_for, build_tta_transforms_for
from src.engine.tta import predict_with_tta
from src.models.backbones import load_backbone
from src.models.heads import build_head

logger = logging.getLogger(__name__)


def build_model_from_config(cfg: Dict[str, Any]) -> torch.nn.Module:
    from src.engine.train import FewShotClassifier, FewShotClassifierWithLoRA

    backbone = load_backbone(
        cfg["backbone"]["name"],
        cfg["backbone"].get("pretrained", True),
        cfg["backbone"].get("freeze", True),
        model_name=cfg["backbone"].get("model_name"),
    )

    head_kwargs = {}
    if cfg["head"]["type"] == "mlp":
        head_kwargs = {
            "hidden": cfg["head"].get("hidden", 512),
            "dropout": cfg["head"].get("dropout", 0.2),
        }
    head = build_head(cfg["head"]["type"], backbone.embed_dim, cfg["num_classes"], **head_kwargs)

    if cfg.get("lora", {}).get("enabled", False):
        from src.models.lora import apply_lora

        backbone, _ = apply_lora(
            backbone,
            r=cfg["lora"]["r"],
            alpha=cfg["lora"]["alpha"],
            target_blocks=cfg["lora"]["target_blocks"],
        )
        return FewShotClassifierWithLoRA(backbone, head)
    return FewShotClassifier(backbone, head)


def load_trained_model(ckpt: Dict[str, Any]) -> torch.nn.Module:
    cfg = ckpt["config"]
    model = build_model_from_config(cfg)
    state = ckpt["state_dict"]

    if state and all(k.startswith("head.") for k in state):
        head_state = {k[len("head."):]: v for k, v in state.items()}
        model.head.load_state_dict(head_state)
    else:
        model.load_state_dict(state)
    return model


@torch.no_grad()
def generate_submission(
    model: torch.nn.Module,
    test_dir: str,
    output_path: str,
    device: str,
    use_tta: bool = True,
    image_size: int = 224,
    backbone_name: str = "dinov3_vitl16",
    model_name: str = None,
):
    model.eval()
    model.to(device)

    test_dataset = TestImageDataset(test_dir, transform=None)

    if use_tta:
        tta_transforms = build_tta_transforms_for(backbone_name, model_name, image_size)
    else:
        eval_transform = build_eval_transform_for(backbone_name, model_name, image_size)

    ids = []
    labels = []

    for idx in tqdm(range(len(test_dataset)), desc="Predicting"):
        image, image_id = test_dataset[idx]

        if use_tta:
            probs = predict_with_tta(model, image, tta_transforms, device)
            pred = probs.argmax().item()
        else:
            img_tensor = eval_transform(image).unsqueeze(0).to(device)
            logits = model(img_tensor)
            pred = logits.argmax(dim=1).item()

        ids.append(image_id)
        labels.append(pred)

    df = pd.DataFrame({"ID": ids, "Label": labels})
    df = df.sort_values("ID").reset_index(drop=True)
    df["ID"] = df["ID"].astype(str) + ".jpg"
    df.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path} ({len(df)} predictions)")
