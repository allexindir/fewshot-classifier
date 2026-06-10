import json
import logging
import math
import os
from typing import Dict, List
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.engine.evaluate import evaluate
from src.utils.checkpoint import save_checkpoint

logger = logging.getLogger(__name__)


class FewShotClassifier(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone.forward_features(x)
        return self.head(features)


class FewShotClassifierWithLoRA(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        features = self.backbone.forward_features(x)
        return self.head(features)


def build_optimizer(
    model: nn.Module,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
    lora_params: List[torch.nn.Parameter] = None,
) -> AdamW:
    param_groups = [
        {"params": model.head.parameters(), "lr": head_lr},
    ]
    if lora_params:
        param_groups.append({"params": lora_params, "lr": backbone_lr})
    return AdamW(param_groups, weight_decay=weight_decay)


def build_scheduler(optimizer, epochs: int, warmup_epochs: int = 5):
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    device: str,
    lora_params: List[torch.nn.Parameter] = None,
):
    epochs = config.get("epochs", 50)
    label_smoothing = config.get("label_smoothing", 0.1)
    checkpoint_dir = config.get("paths", {}).get("checkpoint_dir", "outputs/checkpoints")
    metrics_dir = config.get("paths", {}).get("metrics_dir", "outputs/metrics")
    run_name = config.get("run_name", "run")
    seed = config.get("seed", 42)
    warmup_epochs = config.get("scheduler", {}).get("warmup_epochs", 5)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = build_optimizer(
        model,
        head_lr=config["optimizer"]["head_lr"],
        backbone_lr=config["optimizer"]["backbone_lr"],
        weight_decay=config["optimizer"]["weight_decay"],
        lora_params=lora_params,
    )
    scheduler = build_scheduler(optimizer, epochs, warmup_epochs)

    best_acc = 0.0
    history: List[Dict[str, float]] = []
    os.makedirs(metrics_dir, exist_ok=True)
    history_path = os.path.join(metrics_dir, f"{run_name}_history.json")

    use_amp = torch.cuda.is_available()

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        if hasattr(model, "backbone"):
            if not lora_params:
                model.backbone.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(images)
                    loss = criterion(logits, labels)
                if torch.isnan(loss):
                    logger.warning(f"NaN loss at epoch {epoch}")
                    continue
            else:
                logits = model(images)
                loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        if device == "mps":
            torch.mps.empty_cache()

        train_acc = correct / total if total > 0 else 0.0
        train_loss = running_loss / total if total > 0 else 0.0

        val_metrics = evaluate(model, val_loader, device)

        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        with open(history_path, "w") as f:
            json.dump({"run_name": run_name, "history": history}, f, indent=2)

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            save_checkpoint(
                path=os.path.join(checkpoint_dir, f"{run_name}_best.pth"),
                model=model,
                config=config,
                seed=seed,
                val_acc=best_acc,
                epoch=epoch,
            )
            logger.info(f"  -> New best val accuracy: {best_acc:.4f}")

    # save final checkpoint
    save_checkpoint(
        path=os.path.join(checkpoint_dir, f"{run_name}_final.pth"),
        model=model,
        config=config,
        seed=seed,
        val_acc=best_acc,
        epoch=epochs,
    )

    logger.info(f"Training complete. Best val accuracy: {best_acc:.4f}")
    return best_acc
