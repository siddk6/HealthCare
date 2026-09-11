"""
Day 4 — The pipeline: uploaded image -> stored in bucket -> model scores it
-> prediction + confidence + risk tier written to a database record.

This is the single function the Streamlit dashboard calls on upload
(`process_new_case`), and it's also directly unit-testable / scriptable
without touching Streamlit at all (see scripts/seed_demo_data.py).
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CLASS_CODES, CLASS_LABELS, CHECKPOINT_DIR  # noqa: E402
from data.dataset import build_transforms  # noqa: E402
from training.model import build_model, get_target_layer  # noqa: E402
from evaluation.gradcam import GradCAM  # noqa: E402
from cloud.storage_backend import get_storage_backend  # noqa: E402
from cloud.db_backend import get_db_backend  # noqa: E402
from triage.risk_engine import assign_risk  # noqa: E402


class ScoringModel:
    """Loads a checkpoint once and reuses it — the dashboard should
    instantiate this a single time (e.g. via st.cache_resource), not per
    upload, since loading weights on every image would be painfully slow."""

    def __init__(self, checkpoint_path: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = checkpoint_path or self._find_default_checkpoint()

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.backbone_name = ckpt["backbone"]
        self.model = build_model(self.backbone_name).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.target_layer = get_target_layer(self.model, self.backbone_name)
        self.transform = build_transforms(train=False)

    @staticmethod
    def _find_default_checkpoint() -> str:
        candidates = sorted(CHECKPOINT_DIR.glob("*_best.pt"))
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint found in {CHECKPOINT_DIR}. Train a model first "
                "(training/train.py), or pass --checkpoint explicitly."
            )
        return str(candidates[0])

    def predict(self, pil_image: Image.Image) -> dict:
        """Returns predicted class, confidence, full class_probs dict, and a
        Grad-CAM overlay (as a numpy RGB array) explaining the top class."""
        tensor = self.transform(pil_image.convert("RGB")).unsqueeze(0).to(self.device)

        cam = GradCAM(self.model, self.target_layer)
        heatmap, pred_idx, probs = cam.generate(tensor)
        overlay = cam.overlay_on_image(heatmap, pil_image)
        cam.close()

        class_probs = {CLASS_CODES[i]: float(probs[i]) for i in range(len(CLASS_CODES))}
        predicted_class = CLASS_CODES[pred_idx]

        return {
            "predicted_class": predicted_class,
            "predicted_label": CLASS_LABELS[predicted_class],
            "confidence": float(probs[pred_idx]),
            "class_probs": class_probs,
            "gradcam_overlay": overlay,  # np.ndarray, HxWx3, uint8
        }


def process_new_case(local_image_path: str, model: ScoringModel, storage=None, db=None) -> dict:
    """
    The Day-4 pipeline end to end:
        1. upload the image to the storage backend (local disk by default)
        2. run the model + Grad-CAM
        3. run the risk engine to get a tier + priority score
        4. write a case record to the DB backend
    Returns the full case dict (including the case_id) so the caller
    (dashboard) can display it immediately without a re-fetch.
    """
    storage = storage or get_storage_backend()
    db = db or get_db_backend()

    image_ref = storage.upload(local_image_path)

    pil_image = Image.open(local_image_path)
    prediction = model.predict(pil_image)

    risk = assign_risk(
        prediction["predicted_class"], prediction["confidence"], prediction["class_probs"]
    )

    case_id = db.create_case(
        {
            "image_ref": image_ref,
            "image_filename": Path(local_image_path).name,
            "predicted_class": prediction["predicted_class"],
            "predicted_label": prediction["predicted_label"],
            "confidence": prediction["confidence"],
            "class_probs": prediction["class_probs"],
            "risk_tier": risk["risk_tier"],
            "priority_score": risk["priority_score"],
        }
    )

    case = db.get_case(case_id)
    case["rationale"] = risk["rationale"]
    case["gradcam_overlay"] = prediction["gradcam_overlay"]  # not persisted, for immediate display only
    return case
