# Demo guide — Skin Lesion Triage Queue

A tight script for a live presentation. Practice the flow once before the real
thing so nothing fumbles on the day.

## 0. Pre-flight (do this before you present)

From the project root, in **PowerShell**, with the venv already created:

```powershell
# point at the cloud DB + your service-account key (this shell session only)
$env:STORAGE_BACKEND = "local"
$env:DB_BACKEND      = "firestore"
$env:FIREBASE_CREDENTIALS_JSON = "C:\Users\siddk\Downloads\healthcare-analytics-7c05e-firebase-adminsdk-fbsvc-bc2287d9dd.json"
```

Then launch the dashboard:

```powershell
.\venv\Scripts\python.exe -m streamlit run dashboard\app.py --server.port 8501
```

It opens at http://localhost:8501. If the top of the page shows a yellow
**DEMO MODE** banner, the checkpoint wasn't found — stop and check
`checkpoints\resnet18_best.pt` exists before presenting.

> Two terminals is safest: one running the server, and keep the Firebase
> console open in a browser tab on **Firestore → `triage_cases`**.

## 1. The 90-second flow

1. **Open on the Triage Queue tab.** Point at the metric tiles
   (Urgent / Routine / Low / Reviewed) and the color-coded, sorted list.
   "Every upload becomes a case; the riskiest float to the top."
2. **Upload a real image.** Go to **⬆️ Upload New Case**, pick a file from
   `data_raw\images\` (e.g. `ISIC_0033444.jpg`), click **Score and add to
   queue**. Show the predicted class, confidence, risk tier, and the Grad-CAM
   overlay ("this is where the model looked").
3. **Show it landed in the cloud.** Switch to the Firebase console tab, refresh
   Firestore → `triage_cases` → the new document is there with its
   prediction, probabilities, and tier. "The record is persisted to a cloud
   database, not just held in memory."
4. **Close the loop.** Back in the dashboard, expand the case and click
   **Confirm & clear** or **Apply override** — show that the Firestore doc's
   `status` updates too. "Every case ends in a logged human decision."

## 2. Talking points that land

- **It's triage, not diagnosis.** The system prioritizes *which case a
  clinician sees first* — it never replaces the clinician. Point to
  `docs/LIMITATIONS.md`.
- **The clever bit:** the risk engine escalates on *combined malignant
  probability mass*, not just the top-1 label — so a case the model calls
  "benign nevi" can still be flagged **routine/urgent** if it carries
  meaningful melanoma probability. This is the difference between a classifier
  and a triage tool.
- **Cloud, honestly stated:** "The case *database* is live Firebase Firestore;
  images stay on local disk in the free configuration." Precise and true.

## 3. Q&A cheat sheet

- **"How accurate is it?"** — 74% overall, but accuracy is the wrong lens
  here. Macro recall is 74.5%; the number that matters is malignant-vs-benign
  recall at **80.8%**. (Numbers: `evaluation/reports/metrics.json`.)
- **"Does it catch melanoma?"** — Melanoma recall is **73%** with the lowest
  per-class AUC (0.887). That's exactly why it's a *triage aid* that
  over-refers rather than a diagnostic tool — a missed melanoma is a real,
  measured failure mode, documented in `docs/LIMITATIONS.md`.
- **"Is it really cloud-based?"** — Yes for the database (Firestore, verified
  end-to-end). Images are local in the free tier; S3/Firebase Storage are
  pluggable behind the same interface if you enable billing.
- **"Could a clinic use this?"** — No, not as-is: no prospective validation,
  no calibration, no regulatory review. The limitations doc lists what would
  be needed. That honesty is deliberate.

## 4. If something breaks mid-demo

- **Server won't start / port busy:** use a different port
  (`--server.port 8502`) or free 8501.
- **Firestore auth error:** re-check `$env:FIREBASE_CREDENTIALS_JSON` points at
  the real key path; env vars only last for the current PowerShell session.
- **Total fallback:** unset the cloud vars and run local — it still works fully
  on SQLite + local storage, and `python scripts/seed_demo_data.py` repopulates
  a tier-spanning queue instantly:
  ```powershell
  Remove-Item Env:DB_BACKEND, Env:STORAGE_BACKEND, Env:FIREBASE_CREDENTIALS_JSON
  .\venv\Scripts\python.exe scripts\seed_demo_data.py
  .\venv\Scripts\python.exe -m streamlit run dashboard\app.py
  ```
