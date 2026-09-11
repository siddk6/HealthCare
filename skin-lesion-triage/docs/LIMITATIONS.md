# Limitations — read this before anyone mistakes this for a diagnostic tool

**This system is a triage aid. It is not a diagnostic tool, and it has not
been validated for clinical use.** It does not determine what a lesion is —
it prioritizes the order in which a clinician looks at a batch of images.
The answer is always "a person reviews this," never "the model decided."

## Why "triage aid, not diagnosis" is a real distinction, not a disclaimer

A diagnostic tool's mistakes are the whole story: if it says "benign" and
it's wrong, that's the failure. A triage tool's mistakes are softer *only if*
the tool is honest about its own uncertainty and a human is actually in the
loop — which is why every case here ends in a queue entry for a person to
confirm, dismiss, or override, never in an autonomous action. Remove the
human review step and this system's safety case falls apart entirely.

## Dataset limitations (these propagate directly into model behavior)

- **HAM10000 skews toward certain skin tones and imaging conditions.** It was
  collected primarily from Austrian and Australian clinics using
  dermatoscopes. A model trained on it will generalize worse — silently, not
  with a loud error — to lesions on darker skin, to phone-camera photos
  instead of dermatoscope images, and to populations/lighting not
  represented in the source data. This is a well-documented, general problem
  in dermatology AI, not specific to this implementation.
- **Class imbalance is severe** (melanocytic nevi are ~67% of the dataset;
  several classes are under 2%). Weighted loss and macro-recall-based model
  selection (see `training/losses.py`, `training/train.py`) reduce but do
  not eliminate the effect — always check per-class recall in
  `evaluation/reports/metrics.json` before trusting a headline accuracy
  number.
- **Multiple images per lesion.** The splits in this repo are grouped by
  `lesion_id` specifically to prevent the same lesion leaking across
  train/val/test (see `data/prepare_splits.py`) — a naive image-level split
  would inflate validation metrics and hide this.
- **Label noise.** Ground truth for a meaningful fraction of HAM10000 comes
  from expert consensus or follow-up rather than histopathology confirmation
  for every single image. The dataset documentation is explicit about this;
  treat reported metrics as an estimate, not a certified accuracy.

## Model / explainability limitations

- Grad-CAM shows *where* the model's attention concentrated for its top
  prediction — it does not prove the model is reasoning about clinically
  meaningful features (asymmetry, border irregularity, color, diameter,
  evolution — the "ABCDE" criteria clinicians use). A heatmap over the
  lesion is reassuring; a heatmap over background skin or an artifact is a
  sign the model may be latching onto something spurious, and should lower
  trust in that specific prediction, not just be treated as decoration.
- Confidence scores are softmax outputs, not calibrated probabilities. A
  reported "82% confidence" is not the same claim as "82% chance this is
  correct" unless the model has been explicitly calibrated (e.g. temperature
  scaling) and validated for calibration — this repo does not do that step,
  and the risk-tier thresholds in `triage/risk_engine.py` should be read as
  heuristics tuned on raw softmax output, not on true probability.
- The malignant/benign risk-tier split (`config.MALIGNANT_CODES` /
  `BENIGN_CODES`) is a simplification for triage-ordering purposes. It is
  not a clinical staging system, and several classes (e.g. `bkl`) contain
  a spectrum of lesions that isn't fully captured by a binary bucket.

## Measured weak spots of the current checkpoint (test split)

These are not hypothetical — they are the numbers the shipped
`checkpoints/resnet18_best.pt` actually produced on the held-out test split
(1,494 images), recorded in `evaluation/reports/metrics.json`. They will change
if the model is retrained, but for *this* checkpoint they are the honest bounds:

- **Melanoma (`mel`) is the weakest class where it matters most.** Recall is
  **73.3%** — meaning roughly **1 in 4 melanomas is not top-1 classified as
  melanoma** — precision is only 42.7%, and its ROC-AUC (**0.887**) is the
  **lowest of all seven classes**. The malignant-vs-benign grouping recovers
  some of this (80.8% recall on the combined malignant bucket, which is why the
  triage tiering looks at combined malignant probability mass rather than the
  top-1 label), but no one should read this model as reliably catching every
  melanoma. A missed melanoma is the exact failure mode this design exists to
  reduce, and it is a measured, non-trivial rate here — not something the
  numbers let us wave away.
- **`akiec` (actinic keratoses / intraepithelial carcinoma), also on the
  malignant spectrum, is under-caught:** recall **60.3%**, precision 58.5%.
- **`bkl` (benign keratosis-like lesions) is the most confused class overall:**
  recall **57.2%** and AUC **0.905**, the second-lowest AUC. It is benign, so
  the *safety* cost of confusing it is lower, but it drags down overall
  precision for the malignant classes it gets mixed up with.
- **`df` (dermatofibroma) has only 7 test images**, so its reported precision
  (25%) and recall (71.4%) are too small-sample to trust as stable estimates —
  treat that row as essentially unmeasured rather than as a real 25% precision.

The headline top-1 accuracy (74.4%) is therefore genuinely misleading if read
alone: it is propped up by the large, easy `nv` class (96.5% precision) while
the classes a clinician most needs caught are exactly the ones with the weakest
recall and AUC above.

## What would be needed before this could touch a real clinical workflow

This list is intentionally long — it's meant to show why "it prioritizes
review, it doesn't replace it" is a load-bearing design decision, not a
hedge:

1. Prospective validation on a dataset independent of HAM10000, ideally
   multi-site and including the skin tones/imaging conditions underrepresented
   in the training data.
2. Calibration analysis (are stated confidences actually reliable?) and
   recalibration if not.
3. A clinical-safety and regulatory review appropriate to the jurisdiction
   (e.g., in the US, this would likely implicate FDA software-as-a-medical-
   device pathways depending on intended use and marketing claims).
4. Human-factors testing of the actual triage workflow — does having a
   ranked queue change reviewer behavior in ways that help or hurt (e.g.
   automation bias, where reviewers start trusting the "low" tier without
   looking)?
5. A defined process for what happens when the model is wrong in a way that
   matters — false "low" tier on an actual malignancy is the failure mode
   this whole design exists to reduce, and it needs monitoring, not just a
   one-time evaluation number.

## Honest summary

If someone asks "would you trust this to diagnose someone?" — no, and
that's the point. It sorts a stack of images so a clinician looks at the
most concerning ones first. Every prediction ends at a human, every human
decision is logged, and every number in this repo's evaluation reports
should be read with the dataset limitations above in mind.
