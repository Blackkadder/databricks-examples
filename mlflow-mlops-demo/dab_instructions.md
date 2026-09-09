# Deploy — NorthPeak Goods MLflow CTR Optimization

Workflow-only bundle (no App / Lakebase / KA / MAS). One deploy creates the schema, dashboard, and all three jobs; one setup run initializes the complete demo.

Bundle resource definitions live under `resources/`. Executable Python entrypoints live under `src/jobs/`, interactive notebooks under `src/notebooks/`, and reusable MLOps helpers under `src/northpeak_mlops/`.

**Prerequisites:** Databricks CLI **v0.283.0+** (dashboard `dataset_catalog`/`dataset_schema` rebinding). Auth via a configured CLI profile (`--profile <name>` if not default). A running SQL warehouse for the dashboard + Genie space.

```bash
# 1. Create the schema, dashboard, setup job, retraining job, and monitor job.
databricks bundle deploy \
  --var catalog=solution_builder \
  --var schema=demo_mlflow_logged_ctr_optimization \
  --var warehouse_id=<your_warehouse_id>

# 2. Run the setup job — generate ad-impression data → MLflow CTR retrain
#    (train stale + champion, register @champion in UC, batch-score every
#    segment into gold_ctr_predictions) → initialize model monitoring →
#    deploy the Genie space.
databricks bundle run northpeak_ctr_setup \
  --var catalog=solution_builder \
  --var schema=demo_mlflow_logged_ctr_optimization \
  --var warehouse_id=<your_warehouse_id>
```

After the setup job has created the source tables and initial champion, run the separate plain-Python retraining job with:

```bash
databricks bundle run northpeak_ctr_retrain \
  --var catalog=solution_builder \
  --var schema=demo_mlflow_logged_ctr_optimization \
  --var warehouse_id=<your_warehouse_id>
```

The retraining entrypoint imports helper functions for training, registration, guarded `@champion` promotion, and batch scoring. It does not call MLflow primitives directly.

The **NorthPeak CTR Model Monitor** runs automatically every day at 7:00 AM in `America/New_York`. It refreshes `gold_model_monitoring_metrics`, appends its decision to `model_retrain_decisions`, and conditionally invokes `northpeak_ctr_retrain` only after two consecutive critical 7-day periods, sufficient impression volume, and the seven-day cooldown check. Run it manually with:

```bash
databricks bundle run northpeak_ctr_monitor \
  --var catalog=solution_builder \
  --var schema=demo_mlflow_logged_ctr_optimization \
  --var warehouse_id=<your_warehouse_id>
```

Thresholds, windows, dimensions, and cooldown policy are configured in `src/northpeak_mlops/training_config.yml`. The dashboard's **Model Health** page reads the exact tables used by the trigger.

`dev` (default) prefixes the schema with `dev_<user>_` and resource names with `[dev <user>]`. For a shared/prod deploy add `-t prod` to both commands.

After a code change, re-run step 1; after a data or model change, re-run steps 1 + 2. The setup tasks are idempotent (`CREATE OR REPLACE` tables, model re-register, monitoring refresh, Genie create-or-update by title).

## Teardown
```bash
databricks bundle destroy --auto-approve \
  --var catalog=solution_builder \
  --var schema=demo_mlflow_logged_ctr_optimization \
  --var warehouse_id=<your_warehouse_id>
```
Removes the schema, dashboard, and bundle-managed jobs. Does **not** drop the registered UC model (`<catalog>.<schema>.ctr_regressor`), the MLflow experiment, or the Genie space — delete those manually if needed.
