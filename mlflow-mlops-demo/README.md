# NorthPeak Goods — MLflow-Logged CTR Optimization

## The Story

| | |
|---|---|
| **Company** | NorthPeak Goods — mid-market e-commerce brand (home & outdoor) |
| **Hero** | Maya Patel, Growth Analytics Lead — owns paid acquisition efficiency + website conversion targets |
| **Problem** | A campaign mix shift 3 weeks ago pushed CTR prediction accuracy off track — **$240K in wasted spend** and missed conversions as performance decays toward baseline |
| **Journey** | Maya standardizes a **batch training workflow** for regression-based CTR/conversion prediction, compares MLflow-logged XGBoost candidates, tunes hyperparameters, and reviews validation lift, feature importance, and drift trends to pick the best model with clean lineage |
| **Resolution** | She promotes the winning model with proper **versioning** (UC model registry + `@champion` alias), keeps the standardized train + batch-score pipeline for future retrains, and watches dashboards show segment-level CTR lift, model drift, and conversion trends |
| **Impact** | $242K wasted spend recovered, prediction gap on the mis-bid segments cut ~80%, MLOps best practices established for every retrain |

---

## Overview

Maya opens her Monday dashboard and sees the model powering NorthPeak's bid decisions has drifted: predicted CTR no longer tracks actuals, and the gap opened ~3 weeks ago when marketing shifted budget from Search toward Social + Display. The stale model over-bid on the new campaign mix, burning **$240K** in three weeks.

## Repository layout

The bundle configuration, executable jobs, demo notebooks, and reusable library code are intentionally separated:

```text
resources/                 Databricks schemas, dashboards, and setup/retrain/monitor job definitions
src/jobs/                  Plain Python job entrypoints
src/notebooks/             Interactive, educational workflow notebooks
src/northpeak_mlops/       Reusable training, registry, deployment, and scoring helpers
src/northpeak_mlops/training_config.yml  Feature sets and XGBoost search space
src/dashboard/             AI/BI dashboard source
src/genie/                 Genie space source
tests/                     Local unit tests
```

There are three complementary jobs:

- **NorthPeak CTR Setup** uses notebooks to make the MLflow primitives visible during the demo. It generates data, trains and scores the initial models, and deploys Genie.
- **NorthPeak CTR Retrain** uses a plain `.py` task. Its entrypoint imports `train_model`, `register_model`, `deploy_model`, and `batch_score_model`; direct MLflow calls remain encapsulated in the reusable library. A quality gate protects the current `@champion` alias from weaker candidates.
- **NorthPeak CTR Model Monitor** runs daily at 7:00 AM Eastern. It refreshes granular accuracy metrics, evaluates persistent drift, and conditionally invokes the existing retraining job. Setup runs it once after initial scoring so the monitoring tables and Model Health dashboard are immediately populated.

### Declarative training configuration

Feature sets and the XGBoost hyperparameter search space live in
[`src/northpeak_mlops/training_config.yml`](src/northpeak_mlops/training_config.yml). The helper library loads this file at runtime and uses each parameter's `type`, `low`, `high`, and optional `log` fields to construct the Optuna search space.

```yaml
destination:
  catalog: solution_builder
  schema: demo_mlflow_logged_ctr_optimization
  tables:
    training_features: gold_segment_ctr_daily
    predictions: gold_ctr_predictions
    monitoring_metrics: gold_model_monitoring_metrics
    retrain_decisions: model_retrain_decisions
  registered_model: {name: ctr_regressor, alias: champion}
  experiments:
    initial_training: {id: "2232515377525630"}
    automated_retraining: {id: "3770970266054053"}

categorical_features:
  - channel
  - device
  - country
  - visitor_type
  - campaign_id
  - objective

feature_sets:
  fs_a_core:
    - channel
    - device
    - country
    - visitor_type
  fs_b_campaign:
    - channel
    - device
    - country
    - visitor_type
    - campaign_id
    - objective
  fs_c_full:
    - channel
    - device
    - country
    - visitor_type
    - campaign_id
    - objective
    - day_of_week
    - impressions
    - is_drift_window

xgboost_search_space:
  n_estimators: {type: int, low: 150, high: 500}
  max_depth: {type: int, low: 3, high: 9}
  learning_rate: {type: float, low: 0.02, high: 0.25, log: true}
  subsample: {type: float, low: 0.7, high: 1.0}
  colsample_bytree: {type: float, low: 0.7, high: 1.0}

monitoring:
  windows_days: [1, 7, 28]
  dimensions: [channel, device, country, visitor_type, campaign_name, objective]
  warning_rmse_increase_pct: 10.0
  critical_rmse_increase_pct: 20.0
  consecutive_critical_periods: 2
  minimum_impressions: 50000
  retrain_cooldown_days: 7
```

The retraining job runs four Optuna trials for each of the three feature sets, so a default run evaluates 12 candidates. Change the YAML to alter the candidate features or tuning ranges without changing training code.

### Drift monitoring and automated retraining

The monitor writes two governed Unity Catalog tables:

- `gold_model_monitoring_metrics` contains daily and rolling 7/28-day metrics at overall, channel, device, country, visitor type, campaign, and objective grain.
- `model_retrain_decisions` records every threshold evaluation, whether retraining was triggered, and the decision reason.

Metrics include impression-weighted RMSE and MAE, signed bias, actual and predicted CTR, absolute CTR gap, change from the pre-drift baseline, volume, and a `healthy`/`warning`/`critical` status. MAPE is intentionally excluded because CTR values near zero make it unstable.

The default policy is declared in `training_config.yml`: warning at 10% RMSE deterioration, critical at 20%, two consecutive critical 7-day periods, at least 50,000 impressions, and a seven-day cooldown. The job uses a Databricks condition task to start `northpeak_ctr_retrain` only when every safeguard passes. The retraining job's independent promotion gate still prevents a weaker candidate from replacing `@champion`.

The dashboard's **Model Health** page uses the same metrics table as the trigger and shows overall trends, accuracy/drift KPIs, filterable slice-level hotspots, and retraining decision history.

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
- Open the **MLflow experiment UI**: the run leaderboard ranks candidates by validation RMSE, feature importance shows the campaign-mix features now dominating, and each run carries its exact feature set, params, metrics, and artifact. Seeded tuning and logged configuration make each retrain reproducible and debuggable.

### Act 3 — Version + promote (1–2 min) — *Unity Catalog + MLflow Registry*
- The winning run is **registered as a new model version** in Unity Catalog and promoted with a `@champion` alias — proper versioning, clean lineage, one governed artifact. *"Which version is live, trained on what, who promoted it — all answerable."*
- The same notebook **batch-scores** every segment into the Gold predictions table (no serving endpoint — predictions are a governed Delta table).

### Act 4 — Watch the recovery (1 min) — *AI/BI Dashboard*
- Refresh the dashboard: the drift line collapses back toward zero, segment-level **CTR lift** (new champion vs. stale model) shows the biggest gains exactly on the Social/Display mix, and projected recovered spend climbs. The standardized pipeline is ready for the next retrain; the demo job is triggered manually with **Run now** or `databricks bundle run northpeak_ctr_retrain`.

### Closing
> Every retrain is now the same governed workflow: features → XGBoost candidates → MLflow-logged runs → a versioned, promoted model → batch predictions that dashboards and Genie read. **MLflow gives Maya the observability — every run, metric, feature set, and model version — and Unity Catalog gives it the same governance as her data.** Days of ad-hoc notebook archaeology become a reproducible, auditable pipeline she can trust on every retrain.

---

## Products Showcased

| Product | Mode | What it does in this demo |
|---------|------|---------------------------|
| **Lakeflow Connect** | Talk track | Ingests ad-impression logs + web analytics (the ad platforms, GA-style events) into the lakehouse — the raw feed the whole workflow trains on |
| **Synthetic Data Gen** | Build | Generates ~2M ad impressions across channels/devices/geos with the engineered campaign-mix drift that broke the old model |
| **ML Training (MLflow + UC)** | Build | Standardized batch workflow: engineer feature sets, train XGBoost CTR-regression candidates, tune with Optuna, **log every run/param/metric/feature-importance/artifact to MLflow**, register + version the champion in UC, batch-score segments to a Gold predictions table |
| **MLflow (tracking + registry)** | Build | The observability spine — experiment leaderboard, feature importance, run comparison, model versioning, and the guarded `@champion` alias |
| **AI/BI Dashboard** | Build | The drift at a glance — predicted-vs-actual CTR divergence, $240K wasted-spend counter, segment CTR-lift, and the post-retrain recovery |
| **AI/BI Genie** | Build | Natural-language Q&A over the CTR data + predictions — *"which channels drifted most?"*, *"what's the CTR lift by segment?"* |
| **Unity Catalog** | Talk track | One governance model over the impression tables, the registered model versions, and the predictions — lineage + audit end to end |
| **Genie Code** | Talk track | Generates the standardized training notebook from a plain-English prompt — anyone can produce the governed workflow |
| **Genie One** | Talk track | The business-user front door — marketing asks the CTR/conversion questions in natural language without touching a notebook |
