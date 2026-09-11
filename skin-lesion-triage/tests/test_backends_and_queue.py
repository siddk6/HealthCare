import sys
import tempfile
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
from cloud.storage_backend import LocalStorageBackend  # noqa: E402
from cloud.db_backend import SQLiteDBBackend  # noqa: E402
from triage import queue_manager as qm  # noqa: E402


@pytest.fixture
def storage(tmp_path):
    return LocalStorageBackend(root=tmp_path / "bucket")


@pytest.fixture
def db(tmp_path):
    return SQLiteDBBackend(db_path=tmp_path / "test.sqlite3")


@pytest.fixture
def sample_image(tmp_path):
    from PIL import Image

    p = tmp_path / "sample.png"
    Image.new("RGB", (16, 16), "red").save(p)
    return str(p)


def test_local_storage_roundtrip(storage, sample_image, tmp_path):
    ref = storage.upload(sample_image)
    assert Path(ref).exists()
    dest = tmp_path / "downloaded.png"
    storage.download(ref, str(dest))
    assert dest.exists()


def test_create_and_fetch_case(db):
    case_id = db.create_case(
        {
            "image_ref": "/fake/path.jpg",
            "predicted_class": "mel",
            "predicted_label": "Melanoma",
            "confidence": 0.9,
            "class_probs": {"mel": 0.9, "nv": 0.1},
            "risk_tier": "urgent",
            "priority_score": 0.9,
        }
    )
    case = db.get_case(case_id)
    assert case["predicted_class"] == "mel"
    assert case["status"] == "pending"
    assert case["class_probs"]["mel"] == 0.9


def test_queue_sorted_by_tier_then_priority(db):
    db.create_case({"image_ref": "a", "predicted_class": "nv", "confidence": 0.9,
                     "class_probs": {}, "risk_tier": "low", "priority_score": 0.1})
    db.create_case({"image_ref": "b", "predicted_class": "mel", "confidence": 0.9,
                     "class_probs": {}, "risk_tier": "urgent", "priority_score": 0.9})
    db.create_case({"image_ref": "c", "predicted_class": "bcc", "confidence": 0.6,
                     "class_probs": {}, "risk_tier": "routine", "priority_score": 0.5})

    queue = qm.get_queue(db, status="pending")
    tiers_in_order = [c["risk_tier"] for c in queue]
    assert tiers_in_order == ["urgent", "routine", "low"]


def test_mark_reviewed_removes_case_from_pending_queue(db):
    case_id = db.create_case({"image_ref": "a", "predicted_class": "nv", "confidence": 0.9,
                               "class_probs": {}, "risk_tier": "low", "priority_score": 0.1})
    assert len(qm.get_queue(db)) == 1
    qm.mark_reviewed(db, case_id, reviewer_note="looks fine")
    assert len(qm.get_queue(db)) == 0
    reviewed = db.get_case(case_id)
    assert reviewed["status"] == "reviewed"
    assert reviewed["reviewer_note"] == "looks fine"


def test_override_preserves_original_prediction(db):
    case_id = db.create_case({"image_ref": "a", "predicted_class": "nv", "predicted_label": "Melanocytic nevi",
                               "confidence": 0.6, "class_probs": {}, "risk_tier": "low", "priority_score": 0.1})
    qm.override_prediction(db, case_id, override_class="mel", reviewer_note="looked malignant on inspection")
    case = db.get_case(case_id)
    # original model output is never overwritten -- only the override field is added
    assert case["predicted_class"] == "nv"
    assert case["override_class"] == "mel"
    assert case["status"] == "reviewed"


def test_dismiss_and_reopen(db):
    case_id = db.create_case({"image_ref": "a", "predicted_class": "nv", "confidence": 0.9,
                               "class_probs": {}, "risk_tier": "low", "priority_score": 0.1})
    qm.dismiss_case(db, case_id)
    assert db.get_case(case_id)["status"] == "dismissed"
    assert len(qm.get_queue(db)) == 0

    qm.reopen_case(db, case_id)
    assert db.get_case(case_id)["status"] == "pending"
    assert len(qm.get_queue(db)) == 1
