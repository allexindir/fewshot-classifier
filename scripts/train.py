"""Training script for few-shot image classification."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from src.data.dataset import FewShotImageDataset
from src.data.splits import stratified_split
from src.data.transforms import build_eval_transform_for, build_train_transform_for
from src.engine.train import FewShotClassifier, FewShotClassifierWithLoRA, train
from src.models.backbones import load_backbone
from src.models.heads import build_head
from src.utils.device import get_device
from src.utils.logging import setup_logger
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Load config: merge base defaults with experiment config
    cfg = OmegaConf.load(args.config)
    if "defaults" in cfg:
        base_path = os.path.join(os.path.dirname(args.config), f"{cfg.defaults[0]}.yaml")
        base_cfg = OmegaConf.load(base_path)
        cfg = OmegaConf.merge(base_cfg, cfg)
        del cfg["defaults"]

    if args.seed is not None:
        cfg.seed = args.seed

    run_name = cfg.get("run_name", "run")
    logger = setup_logger(run_name, cfg.paths.log_dir)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    set_seed(cfg.seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    backbone_name = cfg.backbone.name
    backbone_model_name = cfg.backbone.get("model_name")
    train_transform = build_train_transform_for(backbone_name, backbone_model_name, cfg.image_size)
    eval_transform = build_eval_transform_for(backbone_name, backbone_model_name, cfg.image_size)

    full_dataset = FewShotImageDataset(cfg.paths.train_dir, transform=None)
    train_indices, val_indices = stratified_split(full_dataset, cfg.val_per_class)

    train_dataset = _TransformSubset(full_dataset, train_indices, train_transform)
    val_dataset = _TransformSubset(full_dataset, val_indices, eval_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    backbone = load_backbone(
        cfg.backbone.name,
        cfg.backbone.get("pretrained", True),
        cfg.backbone.get("freeze", True),
        model_name=backbone_model_name,
    )

    head_kwargs = {}
    if cfg.head.type == "mlp":
        head_kwargs = {"hidden": cfg.head.get("hidden", 512), "dropout": cfg.head.get("dropout", 0.2)}
    head = build_head(cfg.head.type, backbone.embed_dim, cfg.num_classes, **head_kwargs)

    lora_params = None
    if cfg.get("lora", {}).get("enabled", False):
        from src.models.lora import apply_lora
        backbone, lora_params = apply_lora(
            backbone, r=cfg.lora.r, alpha=cfg.lora.alpha, target_blocks=cfg.lora.target_blocks
        )
        model = FewShotClassifierWithLoRA(backbone, head)
        logger.info(f"LoRA enabled: {sum(p.numel() for p in lora_params)} trainable backbone params")
    else:
        model = FewShotClassifier(backbone, head)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # Train
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    best_acc = train(model, train_loader, val_loader, config_dict, device, lora_params)
    logger.info(f"Done. Best val accuracy: {best_acc:.4f}")


class _TransformSubset:
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


if __name__ == "__main__":
    main()
