# Skin Lesion Triage Queue

A cloud-backed **triage system** built on the HAM10000 dermatoscopic image
dataset. It doesn't just output "melanoma: 87%" — every upload becomes a
case in a queue: high-risk predictions jump to the top for a clinician to
review first, low-risk cases sit at the bottom, and every case ends in a
human decision (confirm / dismiss / override), logged.

**This is a triage aid, not a diagnostic tool.** See
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) before using, presenting, or
extending this — it's not a disclaimer bolted on at the end, it's load-bearing
for why the design looks the way it does.

## Try it right now (no dataset, no GPU, no cloud account)

```bash
python -m venv venv && source venv/bin/activate     # optional but recommended
pip install -r requirements.txt --break-system-packages
python scripts/seed_demo_data.py                    # populates a demo queue
streamlit run dashboard/app.py
```

This runs entirely on local disk (SQLite + a local folder standing in for a
cloud bucket) and, since there's no trained checkpoint yet, the dashboard
auto-falls-back to a clearly-labeled **DEMO MODE** scorer — so you can see
the queue, the tier color-coding, and the review/override flow before Day 2
of training even finishes. Everything under `triage/` and `cloud/` in demo
mode is the *real* logic — only the prediction itself is synthetic.

## What "cloud-based" means here, concretely

`cloud/storage_backend.py` and `cloud/db_backend.py` each define one
interface with a **local implementation that needs zero setup** (default)
and **real cloud implementations behind the same interface**:

| | Local (default) | Cloud option A | Cloud option B |
|---|---|---|---|
| Image storage | folder on disk | Firebase Storage | AWS S3 |
| Case database | SQLite file | Firestore | DynamoDB |

Switch by setting env vars — no code changes:

```bash
export STORAGE_BACKEND=firebase   # or s3
export DB_BACKEND=firestore       # or dynamodb
export FIREBASE_CREDENTIALS_JSON=/path/to/serviceAccountKey.json
export FIREBASE_STORAGE_BUCKET=your-project.appspot.com
# — or, for AWS —
export S3_BUCKET_NAME=your-bucket
export DYNAMODB_TABLE_NAME=your-table   # partition key: case_id (String)
```

This is what makes the architecture claim ("cloud-based triage queue") real
rather than aspirational: the interface is the same whether you're running
it on a laptop for a demo or pointed at production Firebase/AWS.

## Full pipeline, in order (matches the 7-day plan)

| Day | What | Where |
|---|---|---|
| 1 | Download HAM10000, explore class imbalance, leakage-safe lesion-level split | `data/download_data.py`, `data/prepare_splits.py` |
| 2 | Fine-tune ResNet18/EfficientNet-B0, weighted loss or focal loss for imbalance | `training/train.py`, `training/model.py`, `training/losses.py` |
| 3 | Confusion matrix, per-class precision/recall, ROC-AUC, Grad-CAM | `evaluation/evaluate.py`, `evaluation/gradcam.py` |
| 4 | Cloud pipeline: upload → store → score → risk-tier → DB record | `cloud/pipeline.py`, `cloud/storage_backend.py`, `cloud/db_backend.py` |
| 5 | Triage queue logic: tiering, sorting, review/dismiss/override | `triage/risk_engine.py`, `triage/queue_manager.py` |
| 6 | Streamlit dashboard: upload, live queue, case detail + Grad-CAM | `dashboard/app.py` |
| 7 | This README, limitations doc, tests, deployment | `docs/LIMITATIONS.md`, `tests/` |

### Running it end to end with the real dataset and a real model

```bash
# Day 1
python data/download_data.py --source kaggle      # needs ~/.kaggle/kaggle.json
python data/prepare_splits.py

# Day 2 (GPU strongly recommended; ~15 epochs, ResNet18, batch 32)
python training/train.py --backbone resnet18 --loss weighted_ce --epochs 15

# Day 3
python evaluation/evaluate.py --checkpoint checkpoints/resnet18_best.pt --split test

# Day 4-6: the dashboard picks up checkpoints/*_best.pt automatically
streamlit run dashboard/app.py
```

## Why recall matters more than accuracy here

`nv` (melanocytic nevi, benign) is ~67% of HAM10000. A model that always
predicts `nv` scores ~67% accuracy while catching **zero** melanomas. Every
part of this repo is built around not letting that happen quietly:

- **Training** selects the best checkpoint by validation **macro recall**,
  not accuracy (`training/train.py`), and supports class-weighted or focal
  loss (`training/losses.py`).
- **Evaluation** reports per-class precision/recall/ROC-AUC and a specific
  "malignant-vs-benign" recall/precision number
  (`evaluation/evaluate.py`) — the metric that most directly answers "how
  many actually-dangerous cases did we miss."
- **Splitting** is grouped by `lesion_id`, not image, so the same lesion
  never leaks across train/val/test and inflates validation numbers
  (`data/prepare_splits.py`).
- **Triage tiering** doesn't just trust a "benign" top-1 label — it checks
  the model's own combined malignant-class probability mass before letting
  a case drop to the "low" tier (`triage/risk_engine.py`).

## Explainability

Every prediction can be paired with a Grad-CAM heatmap
(`evaluation/gradcam.py`) showing which pixels drove the model's decision,
overlaid on the original image in the dashboard's case detail view. This is
a from-scratch implementation (register forward/backward hooks on the last
conv block, weight activations by pooled gradients) — no extra dependency
beyond torch/torchvision. See `docs/LIMITATIONS.md` for what a heatmap does
and doesn't tell you.

## Tests

```bash
pip install pytest --break-system-packages
pytest tests/ -v
```

`tests/` covers the risk-tiering logic and the full queue lifecycle
(create → sort → review / dismiss / override) against the local SQLite +
local-storage backends, with no GPU or dataset required — these are the
parts of the system that are pure logic and should never regress silently.

## Deploying the dashboard (Streamlit Community Cloud)

1. Push this repo to GitHub (exclude `data_raw/`, `checkpoints/`, and
   `local_cloud_storage/` — see `.gitignore`).
2. On [share.streamlit.io](https://share.streamlit.io), point at
   `dashboard/app.py`.
3. In the app's "Secrets," set `STORAGE_BACKEND` / `DB_BACKEND` and the
   corresponding cloud credentials if you want a persistent multi-user
   deployment (Streamlit Cloud's own filesystem is ephemeral, so the local
   backends won't persist across restarts there — that's exactly the case
   the Firebase/S3 backends exist for).
4. Either upload a trained checkpoint into the deployment, or ship it in
   demo mode with a clear banner — do not silently present demo-mode output
   as real predictions.

## Repository layout

```
config.py                   # class names, risk thresholds, paths — single source of truth
data/                        # Day 1: download + leakage-safe splitting + PyTorch Dataset
training/                    # Day 2: model factory, imbalance-aware losses, training loop
evaluation/                  # Day 3: metrics, confusion matrix, ROC-AUC, Grad-CAM
cloud/                       # Day 4: storage + DB backends (local default, Firebase/AWS pluggable), pipeline
triage/                      # Day 5: risk tiering, queue sort, review/dismiss/override
dashboard/                   # Day 6: Streamlit app
scripts/seed_demo_data.py    # populate a demo queue without the dataset/GPU
tests/                       # unit tests for the triage logic and backends
docs/LIMITATIONS.md          # Day 7: explicit, load-bearing "not diagnostic" framing
```
