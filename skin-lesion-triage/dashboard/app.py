"""
Day 6 — Streamlit dashboard: upload interface, live queue view (sorted,
color-coded by risk), and a case detail view with image + Grad-CAM heatmap
+ confidence breakdown.

Run:
    streamlit run dashboard/app.py

Works immediately with zero setup (local SQLite + local-disk "bucket").
If no trained checkpoint exists yet in checkpoints/, the dashboard falls
back to a clearly-labeled DEMO MODE synthetic scorer so you can exercise
the whole queue/review/override workflow before Day 2's training finishes.
Point STORAGE_BACKEND / DB_BACKEND env vars at firebase/s3/firestore/dynamodb
for a real cloud deployment — see cloud/README in the project README.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CLASS_CODES, CLASS_LABELS, TIER_COLORS, CHECKPOINT_DIR  # noqa: E402
from cloud.storage_backend import get_storage_backend  # noqa: E402
from cloud.db_backend import get_db_backend  # noqa: E402
from cloud.pipeline import ScoringModel, process_new_case  # noqa: E402
from triage.risk_engine import assign_risk  # noqa: E402
from triage import queue_manager as qm  # noqa: E402

st.set_page_config(page_title="Skin Lesion Triage Queue", layout="wide")


# ---------------------------------------------------------------------------
# Demo-mode fallback: lets the dashboard run before a real model is trained.
# Clearly labeled everywhere it appears — never silently mixed with real
# predictions.
# ---------------------------------------------------------------------------
class DemoScoringModel:
    is_demo = True

    def predict(self, pil_image: Image.Image) -> dict:
        rng = np.random.default_rng(abs(hash(pil_image.tobytes()[:256])) % (2**32))
        raw = rng.dirichlet(np.ones(len(CLASS_CODES)) * 0.6)
        class_probs = {CLASS_CODES[i]: float(raw[i]) for i in range(len(CLASS_CODES))}
        predicted_class = max(class_probs, key=class_probs.get)

        overlay = np.array(pil_image.convert("RGB").resize((224, 224)))
        # cheap synthetic "heatmap" tint so demo mode is visually distinguishable
        # from a real Grad-CAM overlay, never mistaken for one.
        tint = np.zeros_like(overlay)
        tint[:, :, 0] = 255
        overlay = (0.75 * overlay + 0.25 * tint).astype(np.uint8)

        return {
            "predicted_class": predicted_class,
            "predicted_label": CLASS_LABELS[predicted_class],
            "confidence": class_probs[predicted_class],
            "class_probs": class_probs,
            "gradcam_overlay": overlay,
        }


@st.cache_resource
def load_model():
    checkpoints = list(CHECKPOINT_DIR.glob("*_best.pt"))
    if not checkpoints:
        return DemoScoringModel()
    try:
        model = ScoringModel(str(checkpoints[0]))
        model.is_demo = False
        return model
    except Exception as e:
        st.warning(f"Could not load checkpoint ({e}); falling back to demo mode.")
        return DemoScoringModel()


@st.cache_resource
def get_backends():
    return get_storage_backend(), get_db_backend()


model = load_model()
storage, db = get_backends()

if getattr(model, "is_demo", False):
    st.warning(
        "⚠️ DEMO MODE — no trained checkpoint found in `checkpoints/`. Predictions below are "
        "synthetic placeholders that exercise the queue/review/override workflow, not real "
        "model output. Run `training/train.py` to enable real predictions.",
        icon="⚠️",
    )

st.title("🔬 Skin Lesion Triage Queue")
st.caption(
    "A triage aid, not a diagnostic tool. It prioritizes which cases a clinician reviews "
    "first — it does not replace clinical judgment. See docs/LIMITATIONS.md."
)

tab_queue, tab_upload = st.tabs(["📋 Triage Queue", "⬆️ Upload New Case"])


# ---------------------------------------------------------------------------
# Upload tab
# ---------------------------------------------------------------------------
with tab_upload:
    st.subheader("Upload a dermatoscopic image")
    uploaded_file = st.file_uploader("Image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        pil_preview = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(pil_preview, caption="Uploaded image", use_container_width=True)

        if st.button("Score and add to queue", type="primary"):
            with st.spinner("Uploading, scoring, and filing into the queue..."):
                with tempfile.NamedTemporaryFile(suffix=Path(uploaded_file.name).suffix, delete=False) as tmp:
                    pil_preview.convert("RGB").save(tmp.name)
                    tmp_path = tmp.name

                case = process_new_case(tmp_path, model, storage=storage, db=db)

            tier = case["risk_tier"]
            st.success(f"Case `{case['case_id'][:8]}` filed — tier: **{tier.upper()}**")
            with col2:
                st.markdown(
                    f"<span style='background-color:{TIER_COLORS[tier]};color:white;"
                    f"padding:4px 10px;border-radius:6px;font-weight:bold'>{tier.upper()}</span>",
                    unsafe_allow_html=True,
                )
                st.write(f"**Predicted:** {case['predicted_label']} ({case['predicted_class']})")
                st.write(f"**Confidence:** {case['confidence']:.1%}")
                st.write(case.get("rationale", ""))
                st.image(case["gradcam_overlay"], caption="Grad-CAM: where the model looked", use_container_width=True)
            st.rerun()


# ---------------------------------------------------------------------------
# Queue tab
# ---------------------------------------------------------------------------
with tab_queue:
    summary = qm.queue_summary(db)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 Urgent", summary.get("urgent", 0))
    c2.metric("🟠 Routine", summary.get("routine", 0))
    c3.metric("🟢 Low", summary.get("low", 0))
    c4.metric("✅ Reviewed", summary.get("total_reviewed", 0))

    st.divider()
    queue = qm.get_queue(db, status="pending")

    if not queue:
        st.info("Queue is empty. Upload a case in the Upload tab to get started.")

    for case in queue:
        tier = case["risk_tier"]
        header = (
            f"● {tier.upper()}  —  {case.get('predicted_label', case['predicted_class'])}  "
            f"({case['confidence']:.0%} confidence)  —  case {case['case_id'][:8]}"
        )
        with st.expander(header):
            st.markdown(
                f"<div style='height:6px;background-color:{TIER_COLORS[tier]};border-radius:3px'></div>",
                unsafe_allow_html=True,
            )
            col1, col2 = st.columns([1, 1])

            with col1:
                try:
                    img_url = storage.get_url(case["image_ref"])
                    st.image(img_url, caption="Original image", use_container_width=True)
                except Exception:
                    st.caption("(image preview unavailable for this storage backend)")

            with col2:
                st.write("**Confidence breakdown**")
                probs = case.get("class_probs", {})
                for cls, p in sorted(probs.items(), key=lambda kv: -kv[1]):
                    st.write(f"{CLASS_LABELS.get(cls, cls)} (`{cls}`): {p:.1%}")
                    st.progress(min(max(p, 0.0), 1.0))

            st.divider()
            note = st.text_input("Reviewer note (optional)", key=f"note_{case['case_id']}")

            b1, b2, b3, b4 = st.columns(4)
            if b1.button("✅ Confirm & clear", key=f"review_{case['case_id']}"):
                qm.mark_reviewed(db, case["case_id"], reviewer_note=note)
                st.rerun()
            if b2.button("🙅 Dismiss", key=f"dismiss_{case['case_id']}"):
                qm.dismiss_case(db, case["case_id"], reviewer_note=note)
                st.rerun()
            with b3:
                override_choice = st.selectbox(
                    "Override diagnosis",
                    options=["(no override)"] + CLASS_CODES,
                    key=f"override_select_{case['case_id']}",
                    label_visibility="collapsed",
                )
            if b4.button("✏️ Apply override", key=f"override_btn_{case['case_id']}"):
                if override_choice != "(no override)":
                    qm.override_prediction(db, case["case_id"], override_choice, reviewer_note=note)
                    st.rerun()
                else:
                    st.warning("Pick a class to override to first.")

    st.divider()
    with st.expander("Reviewed / dismissed history"):
        for status in ["reviewed", "dismissed"]:
            cases = db.list_cases(status=status)
            if cases:
                st.write(f"**{status.capitalize()}** ({len(cases)})")
                for c in cases:
                    override_note = f" -> overridden to `{c['override_class']}`" if c.get("override_class") else ""
                    st.caption(
                        f"`{c['case_id'][:8]}` — predicted {c['predicted_class']}{override_note} "
                        f"— note: {c.get('reviewer_note') or '(none)'}"
                    )
