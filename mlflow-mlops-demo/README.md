# NorthPeak Goods — MLflow-Logged CTR Optimization

## The Story

| | |
|---|---|
| **Company** | NorthPeak Goods — mid-market e-commerce brand (home & outdoor) |
| **Hero** | Maya Patel, Growth Analytics Lead — owns paid acquisition efficiency + website conversion targets |
| **Problem** | A campaign mix shift 3 weeks ago pushed CTR prediction accuracy off track — **$240K in wasted spend** and missed conversions as performance decays toward baseline |
| **Journey** | Maya standardizes a **batch training workflow** for regression-based CTR/conversion prediction, compares XGBoost candidates with **MLflow logging + tracing**, tunes hyperparameters, and reviews validation lift, feature importance, and drift trends to pick the best model with clean lineage |
| **Resolution** | She promotes the winning model with proper **versioning** (UC model registry + `@champion` alias), keeps the standardized train + batch-score pipeline for future retrains, and watches dashboards show segment-level CTR lift, model drift, and conversion trends |
| **Impact** | $242K wasted spend recovered, prediction gap on the mis-bid segments cut ~80%, MLOps best practices established for every retrain |

---

## Overview

Maya opens her Monday dashboard and sees the model powering NorthPeak's bid decisions has drifted: predicted CTR no longer tracks actuals, and the gap opened ~3 weeks ago when marketing shifted budget from Search toward Social + Display. The stale model over-bid on the new campaign mix, burning **$240K** in three weeks.

Instead of hand-patching a notebook, Maya runs a **standardized batch training workflow**. The workflow engineers features from ad-impression logs — new vs. returning visitor, channel, geography, device, campaign, hour-of-day — and trains several **XGBoost regression candidates** to predict click-through rate. Every run is captured in **MLflow**: parameters, the exact feature set, validation metrics (RMSE, MAE, R²), feature importance, and the model artifact. Hyperparameter tuning trials all log to one experiment, so the leaderboard ranks candidates and the best run is obvious.

She reviews validation lift, feature importance (the campaign-mix features now dominate), and the drift trend, then **promotes the winning model version** to the UC registry with a `@champion` alias. The same notebook **batch-scores** every segment into a Gold predictions table. Dashboards then show segment-level CTR lift (predicted vs. actual), the drift that triggered the retrain collapsing back to zero, and conversion-value recovery.

**Duration:** 6–8 minutes

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Ad impressions generated | ~2.0M rows over 90 days |
| Segments scored | ~430 (campaign × device × geo × visitor-type) |
| Wasted spend during drift window | **~$242K** over 3 weeks |
| Baseline blended CTR | ~2.9% (weekly), ~2.8% overall |
| CTR on mis-bid Social/Display mix | drops from ~1.0% toward ~0.5% while the stale model still predicts the pre-drift rate |
| Drift window | began ~3 weeks ago, still elevated through today (closes on retrain) |
| Champion model error improvement | ~18% lower RMSE on the mis-bid cohort; prediction gap cut ~80% (0.49pp → 0.09pp) |
| XGBoost candidates compared | 3 feature-set variants × Optuna hyperparameter trials, all MLflow-logged |

---

## Demo Walkthrough

**Frame:** Monday morning. Maya's spend-efficiency alert fired — predicted CTR and actual CTR have diverged, and finance flagged $240K of over-spend. She opens Databricks to retrain, properly this time.

### Act 1 — See the drift (1–2 min) — *AI/BI Dashboard + Genie*
- Open the **CTR Optimization dashboard**. The predicted-vs-actual CTR line diverges ~3 weeks ago; a drift KPI spikes; the wasted-spend counter reads ~$240K, concentrated in the Social + Display channels that got the new budget.
- Ask **Genie**: *"Which channels have the biggest gap between predicted and actual CTR in the last 3 weeks?"* — Social and Display surface immediately, with the mis-bid segments.

### Act 2 — Standardize the retrain (2–3 min) — *MLflow + Genie Code*
- Open the **standardized batch training notebook**. *"This is the same workflow every retrain runs — no bespoke scripts."* (**Genie Code** can generate it from a plain-English prompt.)
- Run it: it engineers the feature sets, trains **XGBoost regression** candidates, and tunes hyperparameters — every trial auto-logged to one **MLflow experiment**.
- Open the **MLflow experiment UI**: the run leaderboard ranks candidates by validation RMSE, feature importance shows the campaign-mix features now dominating, and each run carries its exact feature set, params, metrics, and artifact. **MLflow tracing** captures every step so a retrain is reproducible and debuggable.

### Act 3 — Version + promote (1–2 min) — *Unity Catalog + MLflow Registry*
- The winning run is **registered as a new model version** in Unity Catalog and promoted with a `@champion` alias — proper versioning, clean lineage, one governed artifact. *"Which version is live, trained on what, who promoted it — all answerable."*
- The same notebook **batch-scores** every segment into the Gold predictions table (no serving endpoint — predictions are a governed Delta table).

### Act 4 — Watch the recovery (1 min) — *AI/BI Dashboard*
- Refresh the dashboard: the drift line collapses back toward zero, segment-level **CTR lift** (new champion vs. stale model) shows the biggest gains exactly on the Social/Display mix, and projected recovered spend climbs. The standardized pipeline is ready for the next scheduled retrain.

### Closing
> Every retrain is now the same governed workflow: features → XGBoost candidates → MLflow-logged runs → a versioned, promoted model → batch predictions that dashboards and Genie read. **MLflow gives Maya the observability — every run, metric, feature set, and model version — and Unity Catalog gives it the same governance as her data.** Days of ad-hoc notebook archaeology become a reproducible, auditable pipeline she can trust on every retrain.

---

## Products Showcased

| Product | Mode | What it does in this demo |
|---------|------|---------------------------|
| **Lakeflow Connect** | Talk track | Ingests ad-impression logs + web analytics (the ad platforms, GA-style events) into the lakehouse — the raw feed the whole workflow trains on |
| **Synthetic Data Gen** | Build | Generates ~2M ad impressions across channels/devices/geos with the engineered campaign-mix drift that broke the old model |
| **ML Training (MLflow + UC)** | Build | Standardized batch workflow: engineer feature sets, train XGBoost CTR-regression candidates, tune with Optuna, **log every run/param/metric/feature-importance/artifact to MLflow**, register + version the champion in UC, batch-score segments to a Gold predictions table |
| **MLflow (tracking + tracing + registry)** | Build | The observability spine — experiment leaderboard, feature importance, run comparison, model versioning + `@champion` alias, tracing for reproducible/debuggable retrains |
| **AI/BI Dashboard** | Build | The drift at a glance — predicted-vs-actual CTR divergence, $240K wasted-spend counter, segment CTR-lift, and the post-retrain recovery |
| **AI/BI Genie** | Build | Natural-language Q&A over the CTR data + predictions — *"which channels drifted most?"*, *"what's the CTR lift by segment?"* |
| **Unity Catalog** | Talk track | One governance model over the impression tables, the registered model versions, and the predictions — lineage + audit end to end |
| **Genie Code** | Talk track | Generates the standardized training notebook from a plain-English prompt — anyone can produce the governed workflow |
| **Genie One** | Talk track | The business-user front door — marketing asks the CTR/conversion questions in natural language without touching a notebook |
