"""
Day 3 — Grad-CAM, so every prediction ships with a visual "why."

This is a from-scratch, dependency-light Grad-CAM (Selvaraju et al., 2017)
implementation — no extra package needed beyond torch/torchvision, which
keeps the deployment footprint small for the Streamlit dashboard.

Usage as a library (this is what dashboard/app.py and cloud/pipeline.py call):

    from evaluation.gradcam import GradCAM
    cam = GradCAM(model, target_layer)
    heatmap, pred_idx, probs = cam.generate(input_tensor)
    overlay = cam.overlay_on_image(heatmap, original_pil_image)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent.parent))


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        input_tensor: shape (1, C, H, W), already normalized/preprocessed.
        class_idx: which class to explain. If None, uses the model's own
                   top prediction (this is what the dashboard uses by default).

        Returns: (heatmap [H, W] in [0, 1], predicted_class_idx, softmax_probs)
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        # Global-average-pool the gradients -> per-channel importance weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, class_idx, probs.detach().cpu().numpy().squeeze()

    @staticmethod
    def overlay_on_image(heatmap: np.ndarray, original_image, alpha: float = 0.45) -> np.ndarray:
        """
        original_image: PIL.Image (any size) — heatmap is resized to match.
        Returns an RGB numpy array ready to display or save.
        """
        original_np = np.array(original_image.convert("RGB"))
        h, w = original_np.shape[:2]

        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        overlay = (alpha * heatmap_color + (1 - alpha) * original_np).astype(np.uint8)
        return overlay

    def close(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()
