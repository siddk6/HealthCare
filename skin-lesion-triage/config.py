"""
Central configuration for the Skin Lesion Triage system.

Nothing in here talks to the network or the filesystem — it's pure constants —
so it's safe to import from training code, the Streamlit app, and tests alike.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data_raw"          # raw HAM10000 download lands here
PROCESSED_DIR = PROJECT_ROOT / "data_processed"
SPLITS_DIR = PROCESSED_DIR / "splits"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOCAL_STORAGE_DIR = PROJECT_ROOT / "local_cloud_storage"   # stand-in "bucket"
LOCAL_DB_PATH = PROJECT_ROOT / "local_cloud_db.sqlite3"    # stand-in Firestore/DynamoDB

for _p in (DATA_DIR, PROCESSED_DIR, SPLITS_DIR, CHECKPOINT_DIR, LOCAL_STORAGE_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# HAM10000 classes (dx column) -> human labels
# ---------------------------------------------------------------------------
CLASS_CODES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

CLASS_LABELS = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

# Malignant / needs-a-look classes vs. benign, per standard HAM10000 triage framing.
# This drives the risk engine — it is a *clinically-informed heuristic*, not a
# diagnostic claim. See docs/LIMITATIONS.md.
MALIGNANT_CODES = {"mel", "bcc", "akiec"}
BENIGN_CODES = {"nv", "bkl", "df", "vasc"}

NUM_CLASSES = len(CLASS_CODES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_CODES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# ---------------------------------------------------------------------------
# Model / training
# ---------------------------------------------------------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 3e-4
BACKBONE = "resnet18"          # or "efficientnet_b0"
SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Triage / risk tiers
# ---------------------------------------------------------------------------
# Tier assignment looks at (predicted class, confidence) together — see
# triage/risk_engine.py for the full logic and rationale.
RISK_TIERS = ["urgent", "routine", "low"]

TIER_COLORS = {
    "urgent": "#D7263D",
    "routine": "#F2A900",
    "low": "#2E7D32",
}

# Confidence thresholds used inside a malignant-class prediction to decide
# urgent vs. routine (see risk_engine.py for full docstring/rationale).
MALIGNANT_URGENT_CONF_THRESHOLD = 0.50
BENIGN_ESCALATE_MALIGNANT_PROB_THRESHOLD = 0.30
