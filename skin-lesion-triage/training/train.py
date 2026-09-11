"""
Day 2 — fine-tune a pretrained CNN with class-imbalance-aware loss.

Usage:
    python training/train.py --backbone resnet18 --loss weighted_ce --epochs 15

Model selection during training is by *macro recall* on the validation set,
not accuracy — with nv at ~67% of the data, a model that always predicts
"nv" scores ~67% accuracy while catching zero melanomas. Macro recall
(average of each class's recall, unweighted) punishes that collapse.
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from sklearn.metrics import recall_score
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CHECKPOINT_DIR, SPLITS_DIR, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, BACKBONE, SEED  # noqa: E402
from data.dataset import HAM10000Dataset  # noqa: E402
from training.model import build_model  # noqa: E402
from training.losses import compute_class_weights, build_loss  # noqa: E402


def set_seed(seed: int):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    macro_recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default=BACKBONE, choices=["resnet18", "efficientnet_b0"])
    parser.add_argument("--loss", default="weighted_ce", choices=["weighted_ce", "focal"])
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()

    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ds = HAM10000Dataset(SPLITS_DIR / "train.csv", train=True)
    val_ds = HAM10000Dataset(SPLITS_DIR / "val.csv", train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    class_weights = compute_class_weights(train_ds.class_counts())
    print(f"Class weights (inverse frequency, normalized): {class_weights.tolist()}")

    model = build_model(args.backbone, freeze_backbone=args.freeze_backbone).to(device)
    criterion = build_loss(args.loss, class_weights, device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_macro_recall = 0.0
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CHECKPOINT_DIR / f"{args.backbone}_best.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_recall = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_recall = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_recall)

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"[{time.time() - t0:.0f}s] "
            f"train_loss={train_loss:.4f} train_macro_recall={train_recall:.4f} "
            f"val_loss={val_loss:.4f} val_macro_recall={val_recall:.4f}"
        )

        if val_recall > best_macro_recall:
            best_macro_recall = val_recall
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "val_macro_recall": val_recall,
                    "epoch": epoch,
                },
                best_path,
            )
            print(f"  -> new best (val_macro_recall={val_recall:.4f}), saved to {best_path}")

    print(f"\nTraining complete. Best val macro recall: {best_macro_recall:.4f}")
    print(f"Best checkpoint: {best_path}")
    print("Next: python evaluation/evaluate.py --checkpoint " + str(best_path))


if __name__ == "__main__":
    main()
