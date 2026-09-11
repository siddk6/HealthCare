"""
Populates the local queue with a handful of realistic-looking cases so a
reviewer (or you, demoing this) can see the triage queue, the tier
color-coding, and the review/override flow working *before* HAM10000 is
downloaded or a model is trained.

This does NOT touch the real model or real dermatoscopic images — it writes
synthetic placeholder thumbnails and hand-crafted class-probability
distributions through the same risk_engine + db_backend code paths the real
pipeline uses, so what you see in the dashboard after running this is the
actual triage logic, just with fake inputs.

Usage:
    python scripts/seed_demo_data.py
Then:
    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CLASS_CODES, CLASS_LABELS, LOCAL_STORAGE_DIR  # noqa: E402
from cloud.storage_backend import get_storage_backend  # noqa: E402
from cloud.db_backend import get_db_backend  # noqa: E402
from triage.risk_engine import assign_risk  # noqa: E402

# Hand-crafted (predicted_class, confidence-ish distribution) pairs spanning
# all three tiers, so the seeded queue actually demonstrates the sorting.
SCENARIOS = [
    ("mel", {"mel": 0.88, "bcc": 0.05, "nv": 0.03, "akiec": 0.02, "bkl": 0.01, "df": 0.005, "vasc": 0.005}),
    ("bcc", {"bcc": 0.71, "akiec": 0.12, "nv": 0.08, "mel": 0.06, "bkl": 0.02, "df": 0.005, "vasc": 0.005}),
    ("akiec", {"akiec": 0.55, "nv": 0.20, "bkl": 0.15, "bcc": 0.06, "mel": 0.03, "df": 0.005, "vasc": 0.005}),
    ("mel", {"mel": 0.38, "nv": 0.33, "bkl": 0.20, "bcc": 0.05, "akiec": 0.02, "df": 0.01, "vasc": 0.01}),
    ("nv", {"nv": 0.52, "mel": 0.28, "bkl": 0.15, "bcc": 0.03, "akiec": 0.01, "df": 0.005, "vasc": 0.005}),
    ("bkl", {"bkl": 0.60, "nv": 0.25, "mel": 0.10, "akiec": 0.03, "bcc": 0.01, "df": 0.005, "vasc": 0.005}),
    ("nv", {"nv": 0.94, "bkl": 0.03, "mel": 0.01, "bcc": 0.01, "akiec": 0.005, "df": 0.0025, "vasc": 0.0025}),
    ("vasc", {"vasc": 0.91, "nv": 0.05, "bkl": 0.02, "mel": 0.01, "bcc": 0.005, "akiec": 0.0025, "df": 0.0025}),
    ("df", {"df": 0.85, "nv": 0.08, "bkl": 0.04, "mel": 0.02, "bcc": 0.005, "akiec": 0.0025, "vasc": 0.0025}),
    ("nv", {"nv": 0.99, "bkl": 0.005, "mel": 0.003, "bcc": 0.001, "akiec": 0.0005, "df": 0.0003, "vasc": 0.0002}),
]

COLORS = ["#8B5E3C", "#A9744F", "#C98E5E", "#7A4B2B", "#5C3A21", "#B57A56", "#946641", "#D9A066", "#6E4429", "#AD7A4C"]


def make_placeholder_thumbnail(idx: int, color: str) -> str:
    """A synthetic 'lesion-ish' blob thumbnail — clearly not a real medical
    image, just enough for the dashboard's image panel to have something to
    show during a demo."""
    img = Image.new("RGB", (300, 300), "#F2D6BD")
    draw = ImageDraw.Draw(img)
    rng = np.random.default_rng(idx)
    cx, cy = 150 + rng.integers(-20, 20), 150 + rng.integers(-20, 20)
    rx, ry = 60 + rng.integers(-10, 30), 60 + rng.integers(-10, 30)
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=color)
    out_path = LOCAL_STORAGE_DIR / f"_demo_seed_{idx}.png"
    img.save(out_path)
    return str(out_path)


def main():
    storage = get_storage_backend()
    db = get_db_backend()

    created = {"urgent": 0, "routine": 0, "low": 0}
    for i, (pred_class, probs) in enumerate(SCENARIOS):
        local_thumb = make_placeholder_thumbnail(i, COLORS[i % len(COLORS)])
        image_ref = storage.upload(local_thumb, dest_name=f"demo_seed_{i}.png")

        confidence = probs[pred_class]
        risk = assign_risk(pred_class, confidence, probs)

        db.create_case(
            {
                "image_ref": image_ref,
                "image_filename": f"demo_seed_{i}.png",
                "predicted_class": pred_class,
                "predicted_label": CLASS_LABELS[pred_class],
                "confidence": confidence,
                "class_probs": probs,
                "risk_tier": risk["risk_tier"],
                "priority_score": risk["priority_score"],
            }
        )
        created[risk["risk_tier"]] += 1

    print(f"Seeded {len(SCENARIOS)} demo cases: {created}")
    print("Run: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
