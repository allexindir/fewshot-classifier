"""Generate a submission CSV by ensembling multiple trained checkpoints"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import torch
from tqdm import tqdm

from src.data.dataset import TestImageDataset
from src.data.transforms import build_eval_transform_for, build_tta_transforms_for
from src.engine.predict import load_trained_model
from src.engine.tta import predict_with_tta
from src.models.ensemble import ProbEnsemble
from src.utils.checkpoint import load_checkpoint
from src.utils.device import get_device
from src.utils.logging import setup_logger


def _load_member(checkpoint_path: str, device: str, logger):
    """Load a checkpoint and return (model, backbone_name, model_name, image_size, test_dir)."""
    ckpt = load_checkpoint(checkpoint_path, device="cpu")
    cfg = ckpt["config"]
    model = load_trained_model(ckpt)
    model.eval()
    model.to(device)
    logger.info(
        f"Loaded {checkpoint_path}: backbone={cfg['backbone']['name']}, "
        f"epoch={ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f}"
    )
    return {
        "model": model,
        "backbone_name": cfg["backbone"]["name"],
        "model_name": cfg["backbone"].get("model_name"),
        "image_size": cfg.get("image_size", 224),
        "test_dir": cfg["paths"]["test_dir"],
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Ensemble prediction from multiple checkpoints")
    parser.add_argument(
        "--checkpoints", required=True, nargs="+",
    )
    parser.add_argument(
        "--weights", type=float, nargs="+", default=None,
    )
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    parser.add_argument("--test-dir", default=None, help="Override test image directory")
    parser.add_argument("--no-tta", action="store_true", help="Disable TTA")
    args = parser.parse_args()

    weights = args.weights if args.weights is not None else [1.0] * len(args.checkpoints)
    if len(weights) != len(args.checkpoints):
        parser.error(
            f"--weights ({len(weights)}) must match --checkpoints ({len(args.checkpoints)})"
        )

    logger = setup_logger("predict_ensemble")
    device = get_device()
    logger.info(f"Using device: {device}")

    members = [_load_member(ckpt, device, logger) for ckpt in args.checkpoints]

    test_dir = args.test_dir or members[0]["test_dir"]
    test_dataset = TestImageDataset(test_dir, transform=None)

    for m in members:
        if args.no_tta:
            m["transforms"] = [
                build_eval_transform_for(m["backbone_name"], m["model_name"], m["image_size"])
            ]
        else:
            m["transforms"] = build_tta_transforms_for(
                m["backbone_name"], m["model_name"], m["image_size"]
            )

    ensemble = ProbEnsemble(weights)
    logger.info(f"ensembling {len(members)} models with weights {weights} (TTA={not args.no_tta})")

    ids = []
    labels = []
    for idx in tqdm(range(len(test_dataset)), desc="Predicting"):
        image, image_id = test_dataset[idx]
        member_probs = [
            predict_with_tta(m["model"], image, m["transforms"], device) for m in members
        ]
        combined = ensemble.combine(member_probs)
        ids.append(image_id)
        labels.append(int(combined.argmax().item()))

    df = pd.DataFrame({"ID": ids, "Label": labels})
    df = df.sort_values("ID").reset_index(drop=True)
    df["ID"] = df["ID"].astype(str) + ".jpg"
    df.to_csv(args.out, index=False)
    logger.info(f"Submission saved to {args.out} ({len(df)} predictions)")


if __name__ == "__main__":
    main()
