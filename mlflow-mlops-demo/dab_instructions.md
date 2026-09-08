# Deploy — NorthPeak Goods MLflow CTR Optimization

Workflow-only bundle (no App / Lakebase / KA / MAS), so the deploy is two commands.

**Prerequisites:** Databricks CLI **v0.283.0+** (dashboard `dataset_catalog`/`dataset_schema` rebinding). Auth via a configured CLI profile (`--profile <name>` if not default). A running SQL warehouse for the dashboard + Genie space.

```bash
# 1. Create the resource shells (schema + dashboard) and the setup job.
databricks bundle deploy \
  --var catalog=solution_builder \
  --var schema=demo_mlflow_logged_ctr_optimization \
  --var warehouse_id=<your_warehouse_id>

# 2. Run the setup job — generate ad-impression data → MLflow CTR retrain
#    (train stale + champion, register @champion in UC, batch-score every
#    segment into gold_ctr_predictions) → deploy the Genie space.
databricks bundle run northpeak_ctr_setup \
  --var catalog=solution_builder \
  --var schema=demo_mlflow_logged_ctr_optimization \
  --var warehouse_id=<your_warehouse_id>
```

`dev` (default) prefixes the schema with `dev_<user>_` and resource names with `[dev <user>]`. For a shared/prod deploy add `-t prod` to both commands.

After a code change, re-run step 1 (redeploys shells + job); after a data or model change, re-run steps 1 + 2. The setup tasks are idempotent (`CREATE OR REPLACE` tables, model re-register, Genie create-or-update by title).

## Teardown
```bash
databricks bundle destroy --auto-approve \
  --var catalog=solution_builder \
  --var schema=demo_mlflow_logged_ctr_optimization \
  --var warehouse_id=<your_warehouse_id>
```
Removes the schema, dashboard, and job. Does **not** drop the registered UC model (`<catalog>.<schema>.ctr_regressor`), the MLflow experiment, or the Genie space — delete those manually if needed.
