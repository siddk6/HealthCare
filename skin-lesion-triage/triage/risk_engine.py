"""
Day 5 — Confidence-based risk tiering. This is the heart of "triage aid, not
diagnosis": the model's job is not to say what a lesion is, it's to say how
urgently a clinician should look at it.

Tiering logic (documented here because it's a judgment call, not a formula
handed down from the dataset — a reviewer should be able to read this and
agree or disagree with the thresholds):

  1. If the top prediction is a malignant-class code (mel, bcc, akiec):
       - confidence >= MALIGNANT_URGENT_CONF_THRESHOLD  -> "urgent"
       - otherwise                                       -> "routine"
     (Even a low-confidence malignant guess still gets reviewed, just not
     jumped to the very top — a confident malignant call is the closest
     thing this system has to a "look now" signal.)

  2. If the top prediction is benign (nv, bkl, df, vasc), we don't just trust
     it and move on — we look at how much probability mass the model itself
     put on the *malignant* classes combined:
       - sum(malignant probs) >= BENIGN_ESCALATE_MALIGNANT_PROB_THRESHOLD
             -> "routine"  (model called it benign, but wasn't sure enough
                to rule out something serious)
       - otherwise -> "low"

  This means "low" only happens when the model is both confidently benign
  AND confidently not-malignant — the two checks catch different failure
  modes (a wrong benign call with high confidence vs. a wrong benign call
  that was actually a close race with a malignant class).

Within a tier, `priority_score` orders the queue:
  - urgent/routine: sorted by summed malignant probability, descending
    (closest to "actually malignant" floats to the top of its tier)
  - low: sorted by confidence ASCENDING (the least-confidently-benign cases
    surface first within "low," so borderline low-risk cases don't get
    buried at the very bottom behind lesions the model is very sure about)
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    CLASS_CODES,
    MALIGNANT_CODES,
    MALIGNANT_URGENT_CONF_THRESHOLD,
    BENIGN_ESCALATE_MALIGNANT_PROB_THRESHOLD,
)


def malignant_probability_mass(class_probs: dict) -> float:
    """Sum of predicted probability across all malignant-class codes."""
    return sum(prob for cls, prob in class_probs.items() if cls in MALIGNANT_CODES)


def assign_risk(predicted_class: str, confidence: float, class_probs: dict) -> dict:
    """
    Args:
        predicted_class: top-1 predicted class code, e.g. "mel"
        confidence: softmax probability of predicted_class
        class_probs: dict of {class_code: probability} for all 7 classes

    Returns:
        {"risk_tier": "urgent"|"routine"|"low", "priority_score": float,
         "malignant_prob_mass": float, "rationale": str}
    """
    mal_mass = malignant_probability_mass(class_probs)

    if predicted_class in MALIGNANT_CODES:
        if confidence >= MALIGNANT_URGENT_CONF_THRESHOLD:
            tier = "urgent"
            rationale = (
                f"Predicted {predicted_class} (malignant class) with "
                f"{confidence:.0%} confidence, above the {MALIGNANT_URGENT_CONF_THRESHOLD:.0%} urgent threshold."
            )
        else:
            tier = "routine"
            rationale = (
                f"Predicted {predicted_class} (malignant class) but only "
                f"{confidence:.0%} confidence — flagged for review, not top-urgent."
            )
        priority_score = mal_mass  # higher malignant mass = higher in the tier
    else:
        if mal_mass >= BENIGN_ESCALATE_MALIGNANT_PROB_THRESHOLD:
            tier = "routine"
            rationale = (
                f"Predicted benign ({predicted_class}) but combined malignant-class "
                f"probability is {mal_mass:.0%}, above the {BENIGN_ESCALATE_MALIGNANT_PROB_THRESHOLD:.0%} "
                "escalation threshold — not confidently ruled out."
            )
            priority_score = mal_mass
        else:
            tier = "low"
            rationale = (
                f"Predicted benign ({predicted_class}) with low combined malignant "
                f"probability ({mal_mass:.0%})."
            )
            # within "low", ascending confidence surfaces the shakier calls first —
            # we encode that by using (1 - confidence) as the sort key.
            priority_score = 1 - confidence

    return {
        "risk_tier": tier,
        "priority_score": round(float(priority_score), 6),
        "malignant_prob_mass": round(float(mal_mass), 6),
        "rationale": rationale,
    }


if __name__ == "__main__":
    # Quick sanity check with synthetic examples — no model or data needed.
    examples = [
        ("mel", 0.82, {"mel": 0.82, "nv": 0.10, "bcc": 0.03, "akiec": 0.02, "bkl": 0.02, "df": 0.005, "vasc": 0.005}),
        ("mel", 0.35, {"mel": 0.35, "nv": 0.30, "bkl": 0.20, "bcc": 0.10, "akiec": 0.03, "df": 0.01, "vasc": 0.01}),
        ("nv", 0.55, {"nv": 0.55, "mel": 0.25, "bkl": 0.15, "bcc": 0.03, "akiec": 0.01, "df": 0.005, "vasc": 0.005}),
        ("nv", 0.97, {"nv": 0.97, "bkl": 0.01, "mel": 0.01, "bcc": 0.005, "akiec": 0.005, "df": 0, "vasc": 0}),
    ]
    for cls, conf, probs in examples:
        result = assign_risk(cls, conf, probs)
        print(f"pred={cls:5s} conf={conf:.2f} -> {result['risk_tier']:8s} "
              f"(score={result['priority_score']:.3f})  {result['rationale']}")
