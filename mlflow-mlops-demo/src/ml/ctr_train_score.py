# Databricks notebook source
# MAGIC %md
# MAGIC # NorthPeak Goods — CTR Regression: MLflow-logged, versioned, batch-scored
# MAGIC
# MAGIC The standardized batch retrain workflow for NorthPeak's paid-acquisition CTR model.
# MAGIC A campaign mix shift 3 weeks ago moved budget from Search to Social + Display; those
# MAGIC prospecting/awareness segments' true CTR collapsed, but the **stale production model**
# MAGIC (trained on the pre-drift world) still predicts the old rate → over-bidding → ~$242K
# MAGIC wasted spend. This notebook:
# MAGIC
# MAGIC 1. Engineers feature sets from `gold_segment_ctr_daily`.
# MAGIC 2. Trains the **stale/baseline** model on pre-drift rows only → registers **v1**.
# MAGIC 3. Trains **champion candidates** on the full window across **3 feature-set variants**
# MAGIC    with **Optuna** hyperparameter tuning — every trial logged to one MLflow experiment.
# MAGIC 4. Picks the best by validation RMSE → registers a new version → promotes **`@champion`**.
# MAGIC 5. Batch-scores every segment with BOTH models → writes `gold_ctr_predictions`.
# MAGIC
# MAGIC Everything (params, metrics, feature importance, artifacts, the exact feature set) is
# MAGIC captured in MLflow so every retrain is reproducible, versioned, and debuggable.

# COMMAND ----------

dbutils.widgets.text("catalog", "solution_builder", "Catalog")
dbutils.widgets.text("schema", "demo_mlflow_logged_ctr_optimization", "Schema")
dbutils.widgets.text("experiment_path", "", "MLflow experiment path")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
EXPERIMENT_PATH = dbutils.widgets.get("experiment_path")

MODEL_NAME = f"{CATALOG}.{SCHEMA}.ctr_regressor"
print(f"Model: {MODEL_NAME}")
print(f"Experiment: {EXPERIMENT_PATH}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Load features and engineer the 3 feature sets
# MAGIC The prediction target is `actual_ctr` (continuous, 0–0.15). Rows are weighted by
# MAGIC `impressions` so high-volume segments dominate the fit. Three feature-set variants
# MAGIC (the "compare candidates" beat) let the leaderboard show which features matter.

# COMMAND ----------

import json
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import optuna
from mlflow.models.signature import infer_signature

optuna.logging.set_verbosity(optuna.logging.WARNING)

mlflow.set_registry_uri("databricks-uc")
if EXPERIMENT_PATH:
    # set_experiment does NOT create the parent workspace folder — pre-create it.
    try:
        import os as _os
        from databricks.sdk import WorkspaceClient
        WorkspaceClient().workspace.mkdirs(_os.path.dirname(EXPERIMENT_PATH))
    except Exception as e:
        print(f"(experiment folder pre-create skipped: {e})")
    mlflow.set_experiment(EXPERIMENT_PATH)

# Pull the segment-day actuals to pandas (small: ~40K rows).
sdf = spark.table(f"{CATALOG}.{SCHEMA}.gold_segment_ctr_daily")
pdf = sdf.toPandas()
print(f"Segment-day rows: {len(pdf):,}")

CATEGORICALS = ["channel", "device", "country", "visitor_type", "campaign_id", "objective"]
for c in CATEGORICALS:
    pdf[c] = pdf[c].astype("category")

# One-hot encode categoricals (consistent columns across train/score via a saved template).
def encode(df, template_cols=None):
    X = pd.get_dummies(df, columns=CATEGORICALS, dtype=float)
    if template_cols is not None:
        X = X.reindex(columns=template_cols, fill_value=0.0)
    return X

# Feature-set variants (each is a subset of engineered columns).
FEATURE_SETS = {
    "fs_a_core":      ["channel", "device", "country", "visitor_type"],
    "fs_b_campaign":  ["channel", "device", "country", "visitor_type", "campaign_id", "objective"],
    "fs_c_full":      ["channel", "device", "country", "visitor_type", "campaign_id", "objective",
                       "day_of_week", "impressions", "is_drift_window"],
}

# gold_segment_ctr_daily is aggregated to segment-day grain (hour_of_day is gone);
# derive day_of_week from event_date for the richest feature set.
pdf["event_date"] = pd.to_datetime(pdf["event_date"])
pdf["day_of_week"] = pdf["event_date"].dt.dayofweek.astype(int)
pdf["is_drift_window"] = pdf["is_drift_window"].astype(int)
TARGET = "actual_ctr"
WEIGHT = "impressions"

def build_xy(df, feats):
    num = [f for f in feats if f not in CATEGORICALS]
    cat = [f for f in feats if f in CATEGORICALS]
    base = df[cat + num].copy()
    X = pd.get_dummies(base, columns=cat, dtype=float)
    return X

# COMMAND ----------
# MAGIC %md
# MAGIC ## Train the stale / baseline model (pre-drift rows only) → register v1
# MAGIC This is what "production" was running: it learned the CTR surface from the Search-heavy,
# MAGIC warm-Social/Display world and never saw the mix shift.

# COMMAND ----------

mlflow.xgboost.autolog(log_input_examples=False, log_models=False, silent=True)

pre = pdf[pdf["is_drift_window"] == 0].copy()
feats_stale = FEATURE_SETS["fs_b_campaign"]

X_pre = build_xy(pre, feats_stale)
STALE_COLS = list(X_pre.columns)
y_pre = pre[TARGET].values
w_pre = pre[WEIGHT].values

Xtr, Xte, ytr, yte, wtr, wte = train_test_split(X_pre, y_pre, w_pre, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="stale_baseline_v1") as run:
    mlflow.set_tag("model_generation", "stale_baseline")
    mlflow.log_param("feature_set", "fs_b_campaign")
    mlflow.log_param("trained_on", "pre_drift_only")
    stale_model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08,
                               subsample=0.9, colsample_bytree=0.9, random_state=42)
    stale_model.fit(Xtr, ytr, sample_weight=wtr)
    pred = stale_model.predict(Xte)
    stale_rmse_selftest = float(np.sqrt(mean_squared_error(yte, pred, sample_weight=wte)))
    mlflow.log_metric("rmse", stale_rmse_selftest)
    sig_stale = infer_signature(Xte, pred)
    info_stale = mlflow.xgboost.log_model(stale_model, name="model",
                                          signature=sig_stale,
                                          input_example=Xte.head(3),
                                          registered_model_name=MODEL_NAME)
STALE_VERSION = info_stale.registered_model_version
print(f"Stale model registered as v{STALE_VERSION}, self-test RMSE={stale_rmse_selftest:.5f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Train champion candidates — 3 feature sets × Optuna trials
# MAGIC All trials log to the same experiment. We hold out a validation split that INCLUDES
# MAGIC drift-window rows, so the champion is rewarded for learning the new Social/Display regime.
# MAGIC The winner is the (feature-set, hyperparams) combo with the lowest validation RMSE.

# COMMAND ----------

# Full-window split (stratify-ish by drift window via random split).
# Row training weight = impressions, but drift-window rows are upweighted so the
# retrain fits the CURRENT regime tightly (that's the point of retraining).
full = pdf.copy()
full["train_weight"] = full[WEIGHT] * np.where(full["is_drift_window"] == 1, 4.0, 1.0)
train_idx, val_idx = train_test_split(full.index, test_size=0.2, random_state=7)
# Selection metric = validation RMSE on the MIS-BID COHORT (drift segments in the
# drift window) — the regime that actually changed and where the stale model is
# wrong. Judging on all rows lets a champion "win" by matching the healthy
# segments both models already agree on; judging on the mis-bid cohort rewards the
# feature set (fs_c_full, with is_drift_window) that can predict the new reality.
val_cohort_mask = ((full.loc[val_idx, "is_drift_segment"] == True) &
                   (full.loc[val_idx, "is_drift_window"] == 1)).values

def evaluate_feature_set(fs_name, feats, n_trials=8):
    X_all = build_xy(full, feats)
    cols = list(X_all.columns)
    Xtr, Xval = X_all.loc[train_idx], X_all.loc[val_idx]
    ytr, yval = full.loc[train_idx, TARGET].values, full.loc[val_idx, TARGET].values
    wtr = full.loc[train_idx, "train_weight"].values
    wval = full.loc[val_idx, WEIGHT].values

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        }
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_generation", "champion_candidate")
            mlflow.log_param("feature_set", fs_name)
            m = XGBRegressor(**params, random_state=42)
            m.fit(Xtr, ytr, sample_weight=wtr)
            p = m.predict(Xval)
            rmse_all = float(np.sqrt(mean_squared_error(yval, p, sample_weight=wval)))
            rmse_cohort = float(np.sqrt(mean_squared_error(
                yval[val_cohort_mask], p[val_cohort_mask], sample_weight=wval[val_cohort_mask])))
            mlflow.log_metric("rmse", rmse_all)
            mlflow.log_metric("rmse_misbid_cohort", rmse_cohort)
            mlflow.log_metric("mae", float(mean_absolute_error(yval, p, sample_weight=wval)))
            mlflow.log_metric("r2", float(r2_score(yval, p, sample_weight=wval)))
            return rmse_cohort

    with mlflow.start_run(run_name=f"hpo_{fs_name}"):
        mlflow.set_tag("feature_set", fs_name)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
    return {"fs_name": fs_name, "feats": feats, "cols": cols,
            "best_params": study.best_params, "best_rmse": study.best_value}

results = [evaluate_feature_set(name, feats) for name, feats in FEATURE_SETS.items()]
for r in results:
    print(f"  {r['fs_name']}: best RMSE={r['best_rmse']:.5f}")

best = min(results, key=lambda r: r["best_rmse"])
print(f"\nWinning feature set: {best['fs_name']}  RMSE={best['best_rmse']:.5f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Retrain the winner on the full window, register a new version, promote `@champion`

# COMMAND ----------

X_champ = build_xy(full, best["feats"])
CHAMP_COLS = list(X_champ.columns)
y_champ = full[TARGET].values
w_champ = full[WEIGHT].values
w_champ_train = full["train_weight"].values
Xtr, Xval, ytr, yval, wtr, wval, wtr_train, _ = train_test_split(
    X_champ, y_champ, w_champ, w_champ_train, test_size=0.2, random_state=7)

with mlflow.start_run(run_name="champion_best") as run:
    mlflow.set_tag("model_generation", "champion")
    mlflow.log_param("feature_set", best["fs_name"])
    mlflow.log_param("trained_on", "full_window")
    for k, v in best["best_params"].items():
        mlflow.log_param(k, v)
    champ = XGBRegressor(**best["best_params"], random_state=42)
    champ.fit(Xtr, ytr, sample_weight=wtr_train)
    p = champ.predict(Xval)
    champ_rmse = float(np.sqrt(mean_squared_error(yval, p, sample_weight=wval)))
    mlflow.log_metric("rmse", champ_rmse)
    mlflow.log_metric("mae", float(mean_absolute_error(yval, p, sample_weight=wval)))
    mlflow.log_metric("r2", float(r2_score(yval, p, sample_weight=wval)))
    # Feature importance artifact
    fi = pd.DataFrame({"feature": CHAMP_COLS, "importance": champ.feature_importances_}) \
        .sort_values("importance", ascending=False)
    fi.to_csv("/tmp/feature_importance.csv", index=False)
    mlflow.log_artifact("/tmp/feature_importance.csv")
    top_features = fi.head(10)["feature"].tolist()
    sig_champ = infer_signature(Xval, p)
    info_champ = mlflow.xgboost.log_model(champ, name="model",
                                          signature=sig_champ,
                                          input_example=Xval.head(3),
                                          registered_model_name=MODEL_NAME)
CHAMP_VERSION = info_champ.registered_model_version

client = MlflowClient(registry_uri="databricks-uc")
client.set_registered_model_alias(MODEL_NAME, "champion", CHAMP_VERSION)
client.set_registered_model_alias(MODEL_NAME, "stale", STALE_VERSION)
print(f"Champion v{CHAMP_VERSION} promoted @champion (RMSE={champ_rmse:.5f})")
print(f"Top features: {top_features}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Compare champion vs stale on the mis-bid cohort (the headline improvement)
# MAGIC The over-bidding lives in the Social/Display prospecting segments during the drift
# MAGIC window. That's where the stale model is wrong and where the retrain earns its keep.

# COMMAND ----------

cohort_mask = (full["is_drift_segment"] == True) & (full["is_drift_window"] == 1)
Xc_stale = build_xy(full[cohort_mask], feats_stale).reindex(columns=STALE_COLS, fill_value=0.0)
Xc_champ = build_xy(full[cohort_mask], best["feats"]).reindex(columns=CHAMP_COLS, fill_value=0.0)
yc = full.loc[cohort_mask, TARGET].values
wc = full.loc[cohort_mask, WEIGHT].values

stale_rmse_drift = float(np.sqrt(mean_squared_error(yc, stale_model.predict(Xc_stale), sample_weight=wc)))
champ_rmse_drift = float(np.sqrt(mean_squared_error(yc, champ.predict(Xc_champ), sample_weight=wc)))
improvement_pct = round(100.0 * (stale_rmse_drift - champ_rmse_drift) / stale_rmse_drift, 1)
print(f"Mis-bid cohort RMSE — stale={stale_rmse_drift:.5f}  champion={champ_rmse_drift:.5f}  improvement={improvement_pct}%")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Batch-score every segment-day with BOTH models → `gold_ctr_predictions`
# MAGIC No serving endpoint — predictions are a governed Delta table that dashboards and Genie read.
# MAGIC Scored in-notebook with pandas (small data), then written back via Spark.

# COMMAND ----------

score = pdf.copy()
X_all_stale = build_xy(score, feats_stale).reindex(columns=STALE_COLS, fill_value=0.0)
X_all_champ = build_xy(score, best["feats"]).reindex(columns=CHAMP_COLS, fill_value=0.0)

score["predicted_ctr_stale"] = np.clip(stale_model.predict(X_all_stale), 0.0, 0.15)
score["predicted_ctr_champion"] = np.clip(champ.predict(X_all_champ), 0.0, 0.15)
score["ctr_gap_stale"] = score["predicted_ctr_stale"] - score["actual_ctr"]
score["ctr_gap_champion"] = score["predicted_ctr_champion"] - score["actual_ctr"]
score["model_version_champion"] = str(CHAMP_VERSION)

out_cols = ["event_date", "channel", "device", "country", "visitor_type", "campaign_id",
            "campaign_name", "objective", "impressions", "spend_usd", "actual_ctr",
            "predicted_ctr_stale", "predicted_ctr_champion", "ctr_gap_stale",
            "ctr_gap_champion", "is_drift_window", "is_drift_segment", "model_version_champion"]
out = score[out_cols].copy()
out["is_drift_window"] = out["is_drift_window"].astype(bool)
out["is_drift_segment"] = out["is_drift_segment"].astype(bool)

sdf_out = spark.createDataFrame(out)
sdf_out = sdf_out.withColumn("predicted_at", __import__("pyspark").sql.functions.current_timestamp())
(sdf_out.write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.gold_ctr_predictions"))

spark.sql(f"""COMMENT ON TABLE {CATALOG}.{SCHEMA}.gold_ctr_predictions IS
  'Batch CTR predictions for every segment-day from BOTH the stale (v{STALE_VERSION}) and champion (v{CHAMP_VERSION} @champion) models. ctr_gap_stale is large+positive on drift segments (the over-bidding); ctr_gap_champion ~0 (the retrain fix). Dashboards + Genie read this.'""")
print(f"Wrote gold_ctr_predictions: {sdf_out.count():,} rows")

# COMMAND ----------

result = {
    "champion_version": str(CHAMP_VERSION),
    "stale_version": str(STALE_VERSION),
    "champion_feature_set": best["fs_name"],
    "champion_rmse_val": round(champ_rmse, 5),
    "stale_rmse_drift": round(stale_rmse_drift, 5),
    "champion_rmse_drift": round(champ_rmse_drift, 5),
    "rmse_improvement_pct": improvement_pct,
    "segments_scored": int(len(out)),
    "top_features": top_features,
}
dbutils.notebook.exit(json.dumps(result))
