"""Day 2 — pretrained backbone factory (transfer learning)."""
import sys
from pathlib import Path

import torch.nn as nn
from torchvision import models

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import NUM_CLASSES  # noqa: E402


def build_model(backbone: str = "resnet18", num_classes: int = NUM_CLASSES, freeze_backbone: bool = False) -> nn.Module:
    """
    Returns a pretrained CNN with its final layer replaced for our 7 classes.

    freeze_backbone=True gives you a quick linear-probe baseline (fast, lower
    ceiling). Leave it False for full fine-tuning, which is what the 7-day
    plan assumes and what actually gets usable recall on the minority classes.
    """
    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown backbone: {backbone!r}. Use 'resnet18' or 'efficientnet_b0'.")

    return model


def get_target_layer(model: nn.Module, backbone: str):
    """Returns the conv layer Grad-CAM should hook into for a given backbone."""
    if backbone == "resnet18":
        return model.layer4[-1]
    if backbone == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"Unknown backbone: {backbone!r}")
