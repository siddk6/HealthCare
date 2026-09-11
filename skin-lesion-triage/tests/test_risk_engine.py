import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from triage.risk_engine import assign_risk, malignant_probability_mass  # noqa: E402


def _probs(**kwargs):
    base = {"mel": 0, "bcc": 0, "akiec": 0, "nv": 0, "bkl": 0, "df": 0, "vasc": 0}
    base.update(kwargs)
    return base


def test_confident_melanoma_is_urgent():
    probs = _probs(mel=0.85, nv=0.10, bcc=0.05)
    result = assign_risk("mel", 0.85, probs)
    assert result["risk_tier"] == "urgent"


def test_low_confidence_melanoma_is_routine_not_urgent():
    probs = _probs(mel=0.35, nv=0.35, bkl=0.30)
    result = assign_risk("mel", 0.35, probs)
    assert result["risk_tier"] == "routine"


def test_confident_benign_with_low_malignant_mass_is_low():
    probs = _probs(nv=0.95, bkl=0.03, mel=0.01, bcc=0.01)
    result = assign_risk("nv", 0.95, probs)
    assert result["risk_tier"] == "low"


def test_benign_prediction_with_high_malignant_mass_escalates():
    # predicted benign, but the model itself put a lot of mass on malignant classes
    probs = _probs(nv=0.45, mel=0.30, bcc=0.15, bkl=0.10)
    result = assign_risk("nv", 0.45, probs)
    assert result["risk_tier"] == "routine"


def test_malignant_probability_mass_sums_correctly():
    probs = _probs(mel=0.2, bcc=0.1, akiec=0.05, nv=0.5, bkl=0.15)
    assert abs(malignant_probability_mass(probs) - 0.35) < 1e-9


def test_urgent_tier_ranks_above_routine_and_low_by_priority_key():
    from triage.queue_manager import sort_queue

    cases = [
        {"case_id": "a", "risk_tier": "low", "priority_score": 0.9},
        {"case_id": "b", "risk_tier": "urgent", "priority_score": 0.4},
        {"case_id": "c", "risk_tier": "routine", "priority_score": 0.99},
    ]
    ordered = [c["case_id"] for c in sort_queue(cases)]
    assert ordered == ["b", "c", "a"]  # urgent always first, regardless of raw score
