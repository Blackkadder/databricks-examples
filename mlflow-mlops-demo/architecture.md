```json
[
  {
    "name": "MLflow CTR Optimization",
    "story": "Synthetic ad-impression, campaign, and web-analytics data represents production feeds in a governed lakehouse. A standardized batch workflow trains XGBoost CTR-regression candidates with MLflow, registers every winner in Unity Catalog, promotes only candidates that beat @champion, and batch-scores segments. A daily model monitor computes granular accuracy and drift metrics, then conditionally triggers retraining when persistent drift passes volume and cooldown safeguards. AI/BI dashboards and Genie consume the governed gold tables.",
    "options": {
      "trademarkLogos": true
    },
    "columns": [
      "sources",
      "pipeline",
      "ml",
      "compute",
      "work",
      "entry"
    ],
    "nodes": [
      {
        "id": "batch-score",
        "type": "lakeflow-jobs",
        "col": "ml",
        "row": 3,
        "label": "Batch Scoring",
        "desc": "Loads @champion, writes gold CTR predictions",
        "ai_reasoning": "batch consumption (no serving endpoint) — loads the @champion model and scores every segment into the gold predictions table the dashboard + Genie read"
      },
      {
        "id": "dashboard",
        "type": "ai-bi-dashboard",
        "col": "work",
        "row": 1,
        "desc": "Drift, wasted spend, champion recovery",
        "ai_reasoning": "the 3-page CTR Optimization dashboard — predicted-vs-actual divergence, $242K wasted-spend KPI, champion recovery, and granular model health"
      },
      {
        "id": "db-platform",
        "type": "db-platform",
        "pin": {
          "at": "top-left",
          "to": "platform-box"
        },
        "ai_reasoning": "platform banner pinned to the box corner; reserves a top band"
      },
      {
        "id": "genie",
        "type": "genie",
        "col": "work",
        "row": 2,
        "desc": "Ask about CTR drift, mix shift, lift",
        "ai_reasoning": "Genie Agent over the gold + predictions tables"
      },
      {
        "id": "genie-one",
        "type": "genie-one",
        "col": "entry",
        "rot": 90,
        "ai_reasoning": "business-user front door (Maya's marketing team) onto the dashboard + Genie; persona built in; edges auto-arrow"
      },
      {
        "id": "governance",
        "type": "governance-block",
        "pin": {
          "at": "top-right",
          "to": "platform-box"
        },
        "params": {
          "access_control": true,
          "ai_gateway": false,
          "genie_ontology": true
        },
        "ai_reasoning": "governance spans everything — Unity Catalog governs the impression tables, the registered model versions, and the predictions; Genie Ontology grounds Genie. No AI Gateway surface (no frontier-model calls / serving endpoint in this batch-only demo). Spanning bar, no per-tile edges."
      },
      {
        "id": "lakeflow",
        "type": "lakeflow-genie-block",
        "col": "pipeline",
        "params": {
          "bronze_desc": "Raw impressions + campaigns",
          "silver_desc": "Cleaned impression fact + drift flags",
          "gold_desc": "Segment-day CTR actuals + predictions"
        },
        "ai_reasoning": "the demo generates the raw→silver→gold layers inline; Lakeflow Connect and SDP remain the production talking track"
      },
      {
        "id": "model-training",
        "type": "model-training",
        "col": "ml",
        "row": 1,
        "desc": "MLflow: 3 feature sets × XGBoost + Optuna, every run logged",
        "ai_reasoning": "the standardized batch retrain — MLflow tracking of every candidate run/param/metric/feature-importance/artifact; trains on gold segment features from the block"
      },
      {
        "id": "model-monitoring",
        "type": "lakeflow-jobs",
        "col": "compute",
        "row": 2,
        "label": "Model Monitoring",
        "desc": "Daily accuracy slices + guarded drift trigger",
        "ai_reasoning": "computes 1/7/28-day RMSE, MAE, bias and CTR gaps by business dimension; triggers retraining only after persistent critical drift, minimum volume and cooldown checks"
      },
      {
        "id": "note-batch",
        "type": "note",
        "at": [
          746,
          262
        ],
        "ai_reasoning": "explains why there's no model-serving tile — deliberate batch pattern",
        "text": "Batch only — no serving endpoint. Predictions are a governed Delta table; dashboards + Genie read the table, never the model."
      },
      {
        "id": "note-mlflow",
        "type": "note",
        "at": [
          506,
          246
        ],
        "ai_reasoning": "MLflow observability is the core of the brief — call it out as a note beside the training + registry lane",
        "text": "Standardized retrain: same MLflow-logged workflow every time — the run leaderboard, feature importance, and model version give clean lineage and reproducible, debuggable retrains."
      },
      {
        "id": "platform-box",
        "type": "box",
        "ai_reasoning": "one white box wrapping the whole flow = 'all of this is the Databricks platform'; raw sources sit outside; auto-renders behind children",
        "wraps": [
          "lakeflow",
          "model-training",
          "uc-registry",
          "batch-score",
          "model-monitoring",
          "sql-lakehouse",
          "dashboard",
          "genie",
          "genie-one"
        ]
      },
      {
        "id": "sql-lakehouse",
        "type": "sql-lakehouse",
        "col": "compute",
        "desc": "Serves CTR actuals + predictions",
        "ai_reasoning": "governed serving copy; the consumption lane (dashboard + Genie) reads gold actuals AND the predictions table from here"
      },
      {
        "id": "src-adplatform",
        "type": "source",
        "col": "sources",
        "row": 1,
        "label": "Ad platforms",
        "icon": "file:vendor/google-ads",
        "desc": "Campaigns, spend, bids",
        "ai_reasoning": "SaaS marketing feed (campaign metadata + spend) → lands on @in-lakeflow-connect"
      },
      {
        "id": "src-impressions",
        "type": "source",
        "col": "sources",
        "row": 3,
        "label": "Ad impression logs",
        "icon": "parquet",
        "desc": "~2M impressions/90d (clicks, conversions)",
        "ai_reasoning": "high-volume impression event files → land directly on @in-direct"
      },
      {
        "id": "src-webanalytics",
        "type": "source",
        "col": "sources",
        "row": 2,
        "label": "Web analytics",
        "icon": "file:vendor/google-analytics",
        "desc": "On-site conversion events",
        "ai_reasoning": "web/GA-style events → managed connector, @in-lakeflow-connect"
      },
      {
        "id": "uc-registry",
        "type": "uc-model-registry",
        "col": "ml",
        "row": 2,
        "desc": "Versioned models + @champion alias",
        "ai_reasoning": "proper versioning + governance — every candidate winner is registered, but only a candidate that improves on the incumbent moves @champion"
      }
    ],
    "edges": [
      {
        "id": "e1",
        "from": "src-adplatform@r",
        "to": "lakeflow@in-lakeflow-connect",
        "flow": true
      },
      {
        "id": "e10",
        "from": "sql-lakehouse@r",
        "to": "genie@l",
        "flow": true
      },
      {
        "id": "e11",
        "from": "genie-one@l",
        "to": "dashboard@r"
      },
      {
        "id": "e12",
        "from": "genie-one@l",
        "to": "genie@r"
      },
      {
        "id": "e2",
        "from": "src-webanalytics@r",
        "to": "lakeflow@in-lakeflow-connect",
        "flow": true
      },
      {
        "id": "e3",
        "from": "src-impressions@r",
        "to": "lakeflow@in-direct",
        "flow": true
      },
      {
        "id": "e4",
        "from": "lakeflow@r",
        "to": "model-training@l",
        "flow": true,
        "label": "Gold segment features"
      },
      {
        "id": "e5",
        "from": "model-training@b",
        "to": "uc-registry@t",
        "flow": true,
        "label": "Register + @champion"
      },
      {
        "id": "e6",
        "from": "uc-registry@b",
        "to": "batch-score@t",
        "flow": true,
        "label": "Load @champion"
      },
      {
        "id": "e7",
        "from": "batch-score@r",
        "to": "sql-lakehouse@l",
        "flow": true,
        "label": "Gold CTR predictions"
      },
      {
        "id": "e8",
        "from": "lakeflow@r",
        "to": "sql-lakehouse@l",
        "flow": true
      },
      {
        "id": "e9",
        "from": "sql-lakehouse@r",
        "to": "dashboard@l",
        "flow": true
      },
      {
        "id": "e13",
        "from": "sql-lakehouse@b",
        "to": "model-monitoring@t",
        "flow": true,
        "label": "Predictions + actuals"
      },
      {
        "id": "e14",
        "from": "model-monitoring@l",
        "to": "model-training@r",
        "flow": true,
        "label": "Conditional retrain"
      }
    ]
  }
]
```
