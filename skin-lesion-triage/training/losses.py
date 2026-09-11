"""
Day 2 — class imbalance handling.

Two options, both wired into train.py via --loss:

  weighted_ce (default): standard cross-entropy with per-class weights
    inversely proportional to class frequency. Simple, well-understood,
    a strong baseline.

  focal: focal loss (Lin et al., 2017), which additionally down-weights
    easy/already-confident examples so training keeps paying attention to
    hard minority-class cases as accuracy climbs. Worth an ablation once
    the weighted_ce baseline is working — compare per-class recall, not
    just overall accuracy, when deciding between them.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(class_counts: "pd.Series") -> torch.Tensor:
    """Inverse-frequency weights, normalized so they average to 1.0."""
    counts = class_counts.values.astype(float)
    counts[counts == 0] = 1  # guard against div-by-zero on an empty split
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        return (focal_term * ce_loss).mean()


def build_loss(loss_name: str, class_weights: torch.Tensor, device: str):
    class_weights = class_weights.to(device)
    if loss_name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    if loss_name == "focal":
        return FocalLoss(alpha=class_weights, gamma=2.0)
    raise ValueError(f"Unknown loss: {loss_name!r}. Use 'weighted_ce' or 'focal'.")
