# ML — CTR Regression Model (MLflow-logged, versioned, batch-scored)

**Skill**: `databricks-ml-training` / `databricks-model-serving` (owns the *how* — UC registry URI, experiment parent-folder trap, alias mechanics, Optuna + MLflow autolog, `spark_udf` env_manager rules, serverless-job `--no-wait` + TASK-run_id pattern, gotchas). **This spec is *what*.** This is the centerpiece of the demo — MLflow observability, feature engineering, standardized batch workflow, versioning + best practices.

Reads `gold_segment_ctr_daily` from `01-lakeflow.md`. Writes `gold_ctr_predictions`; the monitor derives `gold_model_monitoring_metrics` and `model_retrain_decisions`.

## The story this model tells

The **stale production model** was trained on the pre-`DRIFT_START` world (Search-heavy, warm Social/Display). It still predicts ~2.4% CTR for the newly-scaled Social/Display prospecting segments whose true CTR collapsed to ~1.2%, causing the bidding system to over-bid → **$240K wasted spend** over 3 weeks. Maya runs the **standardized batch retrain workflow**: it engineers feature sets, trains several **XGBoost regression** candidates predicting segment CTR, tunes hyperparameters, logs **every run** to one MLflow experiment, and the leaderboard makes the winner obvious. She promotes the champion (proper versioning + `@champion` alias), and the same notebook batch-scores every segment — the new predictions track actuals again, the drift collapses, and the dashboard shows the recovery.

The model is doing something a static rule can't: it **learns the CTR surface from the current data**, so when the campaign mix shifts, a retrain re-fits it and the predictions follow reality instead of a stale assumption.

## What to train

**Regression** target `actual_ctr` (continuous, 0–0.15). Train on `gold_segment_ctr_daily`. **XGBoost regressor**, **4 Optuna trials per feature set**, **MLflow autolog** — every trial is a child run under one experiment (12 candidates by default).

**Two model generations, both logged, to make the story concrete:**
1. **Stale/baseline model** — train on **rows before `DRIFT_START` only** (the pre-drift world). Register as version 1, this is what "production" was running. Its predictions on drift-window segments are systematically too high.
2. **Champion model (the retrain)** — train on the **full 90-day window** (includes the drift period), so it learns the new Social/Display reality. Register as a new version, promote `@champion`. Compare candidates:
   - **3 feature-set variants** (the "compare candidates" beat): (a) channel+device+country+visitor_type only; (b) + campaign + objective; (c) + day_of_week + impressions + `is_drift_window` (full). Each variant × Optuna trials, all in the same experiment.

MLflow logs per run: params (feature set name, XGB hyperparams), metrics (`rmse`, `mae`, `r2` on a held-out validation split), **feature importance** artifact, the model artifact, and the exact feature list. Seeded Optuna sampling plus logged configuration makes the retrain reproducible and debuggable. The automated path registers every candidate winner, but moves `@champion` and overwrites predictions only when the candidate passes the incumbent comparison gate.

## Features (engineered from `gold_segment_ctr_daily`)

Categorical (one-hot encoded): `channel`, `device`, `country`, `visitor_type`, `campaign_id`, `objective`. Numeric: `day_of_week`, `impressions` (volume context), `is_drift_window` (the feature that lets the champion separate the two regimes). Target: `actual_ctr`. Weight rows by `impressions` so high-volume segments matter more. Feature sets and XGBoost search ranges are declared in `src/northpeak_mlops/training_config.yml`.

## Inference shape

Same notebook trains AND scores. After promoting `@champion`, batch-score **every segment-day** (or the latest snapshot per segment) with `spark_udf(models:/{catalog}.{schema}.ctr_regressor@champion)` and overwrite `gold_ctr_predictions`:

| Column | |
|---|---|
| `event_date` | segment-day date |
| `channel`, `device`, `country`, `visitor_type`, `campaign_id`, `campaign_name`, `objective` | segment keys (for dashboard/Genie slicing) |
| `impressions`, `spend_usd` | pass-through context |
| `actual_ctr` | pass-through from gold (the truth) |
| `predicted_ctr_champion` | champion model output, 0–0.15 |
| `predicted_ctr_stale` | stale/baseline model output (so the dashboard can show predicted-vs-actual for BOTH and the gap collapsing) |
| `ctr_gap_stale` | `predicted_ctr_stale − actual_ctr` (the divergence that opened during drift) |
| `ctr_gap_champion` | `predicted_ctr_champion − actual_ctr` (~0 — the recovery) |
| `is_drift_window`, `is_drift_segment` | pass-through flags |
| `model_version_champion` | the UC version string used |
| `predicted_at` | now() |

**Batch only — no serving endpoint.** Every downstream consumer reads `gold_ctr_predictions`; serving adds cost/quota for zero narrative gain.

## Execution paths

The setup job uses `src/notebooks/ctr_train_score.py` as the educational path: load features → engineer 3 feature sets → train stale model → tune champion candidates → log with MLflow → register/promote → batch-score → write `gold_ctr_predictions`.

The separate serverless `northpeak_ctr_retrain` job uses the plain-Python `src/jobs/retrain_model.py` entrypoint. It imports reusable functions from `src/northpeak_mlops/` for feature loading, training, registration, guarded promotion, and scoring; the entrypoint contains no direct MLflow primitives.

**Notebook-source format required** (`# Databricks notebook source`, `# MAGIC %md` headers, `# COMMAND ----------` cell separators) so cells render in the workspace and the MLflow experiment UI is demo-able.

## Monitoring and trigger

`northpeak_ctr_monitor` runs `src/jobs/monitor_model.py` daily at 7:00 AM Eastern and once during setup. It calculates impression-weighted RMSE, MAE, signed bias, actual/predicted CTR, absolute CTR gap, baseline change, volume, and drift status for 1/7/28-day windows at overall, channel, device, country, visitor type, campaign, and objective grain.

The default policy is warning at +10% RMSE versus the pre-drift baseline and critical at +20%. A Databricks condition task calls the existing retraining job only when two consecutive 7-day periods are critical, the latest window has at least 50,000 impressions, and no retrain was triggered in the prior seven days. Every evaluation is appended to `model_retrain_decisions`.

## Who consumes the predictions

1. **AI/BI dashboard** (`04-ai-bi.md`) — predicted-vs-actual CTR line (both models), the gap collapsing after retrain, segment-level CTR-lift, wasted-spend recovery. Reads `gold_ctr_predictions` + `gold_campaign_daily`.
2. **Genie** — answers *"which segments have the biggest predicted-vs-actual gap?"*, *"what's the CTR lift of the champion over the stale model by channel?"*, *"how much wasted spend did the drift cause?"* from `gold_ctr_predictions`.

## Functional validation

- `predicted_ctr_stale − actual_ctr` (i.e. `ctr_gap_stale`) is **large and positive on drift segments** in the drift window (stale over-predicts by ~1.0–1.2pp), ~0 on non-drift segments.
- `ctr_gap_champion` ≈ 0 across the board (champion tracks actuals, including drift segments).
- `rmse_improvement_pct` ≈ 40% (champion vs stale on drift-window validation rows). If far off, adjust Optuna trial count / feature set, don't iterate endlessly — the story needs a clearly better champion, exact % is flexible.
- Two model versions exist in UC registry; `@champion` alias points at the retrain (higher version).
- MLflow experiment shows ≥12 candidate child runs (3 feature sets × 4 Optuna trials) with rmse/mae/r2 + feature-importance artifacts.
- Monitoring tables contain 1/7/28-day metrics for all configured dimensions, and every scheduled evaluation records a reason even when retraining is skipped.

## resources.json

- `ml_model_name`: `solution_builder.demo_mlflow_logged_ctr_optimization.ctr_regressor`
- `mlflow_experiment_path`: `/Workspace/Users/<your-user>/mlflow_ctr_optimization/experiments/ctr_regressor`
