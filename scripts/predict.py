"""Generate submission CSV from a trained checkpoint."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omegaconf import OmegaConf

from src.engine.predict import generate_submission, load_trained_model
from src.utils.checkpoint import load_checkpoint
from src.utils.device import get_device
from src.utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pth")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    parser.add_argument("--no-tta", action="store_true", help="Disable TTA")
    args = parser.parse_args()

    logger = setup_logger("predict")
    device = get_device()
    logger.info(f"Using device: {device}")

    ckpt = load_checkpoint(args.checkpoint, device="cpu")
    cfg = ckpt["config"]
    logger.info(f"Loaded checkpoint from epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f}")

    model = load_trained_model(ckpt)
    logger.info("Model loaded successfully")

    generate_submission(
        model=model,
        test_dir=cfg["paths"]["test_dir"],
        output_path=args.out,
        device=device,
        use_tta=not args.no_tta,
        image_size=cfg.get("image_size", 224),
        backbone_name=cfg["backbone"]["name"],
        model_name=cfg["backbone"].get("model_name"),
    )


if __name__ == "__main__":
    main()
