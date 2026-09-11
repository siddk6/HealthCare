"""
Day 5 — Queue logic: sort tier-then-priority, plus the reviewed/dismissed/
override actions a clinician-role user takes in the dashboard.

This module owns sort order and status transitions. It knows nothing about
Streamlit or any particular DB backend — it takes a DBBackend instance and a
plain list of case dicts, so it's testable without spinning up the dashboard.
"""
import sys
import datetime
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RISK_TIERS  # noqa: E402
from cloud.db_backend import DBBackend  # noqa: E402

_TIER_RANK = {tier: i for i, tier in enumerate(RISK_TIERS)}  # urgent=0, routine=1, low=2


def sort_queue(cases: list) -> list:
    """Tier first (urgent, routine, low), then priority_score descending
    within urgent/routine, and priority_score ascending semantics are already
    baked into the stored value for 'low' by risk_engine.py — so a single
    descending sort on priority_score works for all three tiers once tier
    order is the primary key."""
    return sorted(
        cases,
        key=lambda c: (_TIER_RANK.get(c["risk_tier"], 99), -c.get("priority_score", 0)),
    )


def get_queue(db: DBBackend, status: Optional[str] = "pending") -> list:
    """The main queue view: pending cases, properly sorted."""
    cases = db.list_cases(status=status)
    return sort_queue(cases)


def mark_reviewed(db: DBBackend, case_id: str, reviewer_note: str = "") -> None:
    """Clinician confirms the model's call and clears the case from the queue."""
    db.update_case(
        case_id,
        {
            "status": "reviewed",
            "reviewer_note": reviewer_note,
            "reviewed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )


def dismiss_case(db: DBBackend, case_id: str, reviewer_note: str = "") -> None:
    """Clinician determines the case doesn't need action (e.g. clearly benign
    on manual inspection) and removes it from the active queue."""
    db.update_case(
        case_id,
        {
            "status": "dismissed",
            "reviewer_note": reviewer_note,
            "reviewed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )


def override_prediction(db: DBBackend, case_id: str, override_class: str, reviewer_note: str = "") -> None:
    """Clinician disagrees with the model's top class. We keep the model's
    original prediction on record (never overwritten) and store the
    clinician's call alongside it — this is what makes the override
    auditable instead of silently replacing the model's output."""
    db.update_case(
        case_id,
        {
            "status": "reviewed",
            "override_class": override_class,
            "reviewer_note": reviewer_note,
            "reviewed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )


def reopen_case(db: DBBackend, case_id: str) -> None:
    """Puts a reviewed/dismissed case back into the active queue."""
    db.update_case(case_id, {"status": "pending", "reviewed_at": None})


def queue_summary(db: DBBackend) -> dict:
    """Counts for the dashboard header (e.g. '3 urgent, 12 routine, 40 low')."""
    pending = db.list_cases(status="pending")
    summary = {tier: 0 for tier in RISK_TIERS}
    for case in pending:
        summary[case["risk_tier"]] = summary.get(case["risk_tier"], 0) + 1
    summary["total_pending"] = len(pending)
    summary["total_reviewed"] = len(db.list_cases(status="reviewed"))
    summary["total_dismissed"] = len(db.list_cases(status="dismissed"))
    return summary
