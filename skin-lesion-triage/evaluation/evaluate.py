"""
Day 3 — Evaluation: confusion matrix, per-class precision/recall, ROC-AUC.

Usage:
    python evaluation/evaluate.py --checkpoint checkpoints/resnet18_best.pt --split test

Writes:
    evaluation/reports/confusion_matrix.png
    evaluation/reports/roc_curves.png
    evaluation/reports/metrics.json   (used by the dashboard's model-card panel)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CLASS_CODES, CLASS_LABELS, SPLITS_DIR, MALIGNANT_CODES  # noqa: E402
from data.dataset import HAM10000Dataset  # noqa: E402
from training.model import build_model  # noqa: E402

REPORTS_DIR = Path(__file__).resolve().parent / "reports"


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_probs, all_targets = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_targets.extend(labels.tolist())
    return np.concatenate(all_probs, axis=0), np.array(all_targets)


def plot_confusion_matrix(y_true, y_pred, out_path):
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_CODES)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_CODES)))
    ax.set_yticks(range(len(CLASS_CODES)))
    ax.set_xticklabels(CLASS_CODES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_CODES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalized)")
    for i in range(len(CLASS_CODES)):
        for j in range(len(CLASS_CODES)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                     color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curves(y_true_onehot, y_probs, out_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, cls in enumerate(CLASS_CODES):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_probs[:, i])
        auc = roc_auc_score(y_true_onehot[:, i], y_probs[:, i])
        ax.plot(fpr, tpr, label=f"{cls} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Per-class ROC curves (one-vs-rest)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(ckpt["backbone"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    ds = HAM10000Dataset(SPLITS_DIR / f"{args.split}.csv", train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    probs, y_true = get_predictions(model, loader, device)
    y_pred = probs.argmax(axis=1)

    report = classification_report(
        y_true, y_pred, target_names=CLASS_CODES, output_dict=True, zero_division=0
    )
    print(classification_report(y_true, y_pred, target_names=CLASS_CODES, zero_division=0))

    # Malignant-vs-benign recall — the number that matters most for a triage tool:
    # of every case that was actually malignant, how many did we flag as such
    # (regardless of which malignant subtype we guessed)?
    malignant_idx = {i for i, c in enumerate(CLASS_CODES) if c in MALIGNANT_CODES}
    y_true_malignant = np.isin(y_true, list(malignant_idx))
    y_pred_malignant = np.isin(y_pred, list(malignant_idx))
    malignant_recall = (y_true_malignant & y_pred_malignant).sum() / max(y_true_malignant.sum(), 1)
    malignant_precision = (y_true_malignant & y_pred_malignant).sum() / max(y_pred_malignant.sum(), 1)
    print(f"\nMalignant-vs-benign recall (any-malignant-class caught): {malignant_recall:.4f}")
    print(f"Malignant-vs-benign precision: {malignant_precision:.4f}")

    plot_confusion_matrix(y_true, y_pred, REPORTS_DIR / "confusion_matrix.png")

    y_true_onehot = np.eye(len(CLASS_CODES))[y_true]
    plot_roc_curves(y_true_onehot, probs, REPORTS_DIR / "roc_curves.png")
    per_class_auc = {
        cls: roc_auc_score(y_true_onehot[:, i], probs[:, i]) for i, cls in enumerate(CLASS_CODES)
    }

    metrics_out = {
        "split": args.split,
        "n_samples": int(len(y_true)),
        "per_class": report,
        "per_class_auc": per_class_auc,
        "malignant_vs_benign_recall": float(malignant_recall),
        "malignant_vs_benign_precision": float(malignant_precision),
        "checkpoint": str(args.checkpoint),
        "class_labels": CLASS_LABELS,
    }
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"\nSaved confusion matrix, ROC curves, and metrics.json to {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
