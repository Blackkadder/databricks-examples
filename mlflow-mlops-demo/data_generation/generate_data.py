# Databricks notebook source
"""
NorthPeak Goods — MLflow-logged CTR optimization synthetic data generator.

Mirrors specifications/01-lakeflow.md. No SDP in this build, so THIS script
does the full raw -> silver -> gold layering itself, all in Spark via
databricks-connect. Story: a campaign mix shift ~21 days ago moved budget from
Search to Social + Display; those prospecting/awareness segments' true CTR
collapsed from ~2.4% to ~1.2%, but the stale production model still predicts
~2.4% for them -> over-bidding -> ~$240K wasted spend over 3 weeks.

Layers:
  Phase 1 - RAW:    raw_campaigns, raw_impressions (~2.0M impression events)
  Phase 2 - SILVER: silver_impressions (cleaned + derived drift flags)
  Phase 3 - GOLD:   gold_segment_ctr_daily (segment-day actuals, ML source),
                    gold_campaign_daily (campaign/channel daily rollup)
  gold_ctr_predictions is created by the ML notebook, not here.

Runtime: pre-provisioned databricks-connect venv. Python 3.12. Spark-native
(spark.range + F.when + broadcast joins). No driver loops / no .collect() on
big tables.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from databricks.connect import DatabricksSession
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

IN_NOTEBOOK = "dbutils" in dir()
if IN_NOTEBOOK:
    dbutils.widgets.text("catalog", "solution_builder", "Catalog")
    dbutils.widgets.text("schema", "demo_mlflow_logged_ctr_optimization", "Schema")
    CATALOG = dbutils.widgets.get("catalog")
    SCHEMA = dbutils.widgets.get("schema")
else:
    CATALOG = os.environ.get("DEMO_CATALOG", "solution_builder")
    SCHEMA = os.environ.get("DEMO_SCHEMA", "demo_mlflow_logged_ctr_optimization")
assert CATALOG and SCHEMA

# Time anchors — rolling by default; NPK_PIN_TIME=1 freezes for reproducibility.
STORY_PINNED_NOW = datetime(2026, 9, 8)
NOW = STORY_PINNED_NOW if os.environ.get("NPK_PIN_TIME") == "1" else datetime.now()
STORY_START = NOW - timedelta(days=90)
DRIFT_START = NOW - timedelta(days=21)
DRIFT_PEAK = NOW - timedelta(days=14)
NOW_STR = NOW.strftime("%Y-%m-%d")
START_STR = STORY_START.strftime("%Y-%m-%d")
DRIFT_START_STR = DRIFT_START.strftime("%Y-%m-%d")

N_IMPRESSIONS = 2_000_000

print(f"Target:      {CATALOG}.{SCHEMA}")
print(f"Window:      {START_STR} .. {NOW_STR}")
print(f"DRIFT_START: {DRIFT_START_STR}   DRIFT_PEAK: {DRIFT_PEAK.date()}")

try:
    spark  # noqa: F821
except NameError:
    spark = DatabricksSession.builder.profile(
        os.environ.get("DATABRICKS_CONFIG_PROFILE", "DEFAULT")
    ).serverless(True).getOrCreate()
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")


def _save(df: DataFrame, table: str) -> None:
    fqn = f"{CATALOG}.{SCHEMA}.{table}"
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(fqn)
    print(f"  ok {table:26s} rows={spark.table(fqn).count():>10,}")


# ── Reference data ──────────────────────────────────────────────────────────
# (campaign_id, campaign_name, channel, objective, base_bid_usd)
CAMPAIGNS = [
    ("CMP-001", "SEA-Brand-Core",           "Search",    "conversion",  2.20),
    ("CMP-002", "SEA-NonBrand-Generic",     "Search",    "conversion",  1.80),
    ("CMP-003", "SEA-Shopping-PLA",         "Search",    "conversion",  1.60),
    ("CMP-004", "SOC-Prospecting-Lookalike","Social",    "awareness",   0.90),
    ("CMP-005", "SOC-Retargeting-Cart",     "Social",    "retargeting", 1.10),
    ("CMP-006", "SOC-Video-Awareness",      "Social",    "awareness",   0.70),
    ("CMP-007", "DIS-Programmatic-RON",     "Display",   "awareness",   0.45),
    ("CMP-008", "DIS-Contextual-Home",      "Display",   "awareness",   0.55),
    ("CMP-009", "EML-Newsletter",           "Email",     "conversion",  0.30),
    ("CMP-010", "EML-Winback",              "Email",     "retargeting", 0.35),
    ("CMP-011", "AFF-Coupon-Sites",         "Affiliate", "conversion",  0.80),
    ("CMP-012", "AFF-Content-Blogs",        "Affiliate", "conversion",  0.75),
]
# Drift segments: Social + Display awareness/prospecting campaigns whose CTR collapses.
DRIFT_CAMPAIGNS = ["CMP-004", "CMP-006", "CMP-007", "CMP-008"]

# Channel base CTR (pre-drift warm audiences).
CHANNEL_BASE_CTR = {"Search": 0.032, "Email": 0.028, "Affiliate": 0.022,
                    "Social": 0.020, "Display": 0.012}
CHANNEL_BASE_CVR = {"Search": 0.16, "Email": 0.15, "Affiliate": 0.13,
                    "Social": 0.11, "Display": 0.08}

# Pre-drift budget weights (impression share). Post-drift shift applied inline.
PRE_WEIGHTS = {"CMP-001": 0.16, "CMP-002": 0.16, "CMP-003": 0.13,   # Search 45%
               "CMP-009": 0.12, "CMP-010": 0.08,                    # Email 20%
               "CMP-011": 0.08, "CMP-012": 0.07,                    # Affiliate 15%
               "CMP-004": 0.05, "CMP-005": 0.04, "CMP-006": 0.03,   # Social 12%
               "CMP-007": 0.05, "CMP-008": 0.03}                    # Display 8%

COUNTRIES = [("US", 0.32), ("CA", 0.10), ("GB", 0.16), ("DE", 0.14), ("FR", 0.14), ("AU", 0.14)]
DEVICES = [("Desktop", 0.42), ("Mobile", 0.48), ("Tablet", 0.10)]
VISITORS = [("returning", 0.45), ("new", 0.55)]


def _bands(pairs):
    total = sum(p[-1] for p in pairs)
    out, cum = [], 0.0
    for p in pairs:
        w = p[-1] / total
        out.append((*p[:-1], cum, cum + w))
        cum += w
    return out


# ── Phase 1 — RAW ───────────────────────────────────────────────────────────
campaigns_df = (
    spark.createDataFrame(
        CAMPAIGNS,
        "campaign_id string, campaign_name string, channel string, objective string, base_bid_usd double",
    )
    .withColumn("launch_date", F.lit((NOW - timedelta(days=200)).strftime("%Y-%m-%d")))
)
_save(campaigns_df, "raw_campaigns")

print("Generating impressions…")
# Broadcast dimension band tables for weighted selection via range joins.
camp_band_pre = F.broadcast(spark.createDataFrame(
    _bands([(cid, w) for cid, w in PRE_WEIGHTS.items()]),
    "campaign_id string, cp_low double, cp_high double"))
# Post-drift weights: Search 25 / Email 15 / Affiliate 10 / Social 30 / Display 20.
POST_WEIGHTS = {"CMP-001": 0.09, "CMP-002": 0.09, "CMP-003": 0.07,
                "CMP-009": 0.09, "CMP-010": 0.06,
                "CMP-011": 0.05, "CMP-012": 0.05,
                "CMP-004": 0.13, "CMP-005": 0.04, "CMP-006": 0.13,
                "CMP-007": 0.11, "CMP-008": 0.09}
camp_band_post = F.broadcast(spark.createDataFrame(
    _bands([(cid, w) for cid, w in POST_WEIGHTS.items()]),
    "campaign_id string, cp_low double, cp_high double"))
country_band = F.broadcast(spark.createDataFrame(
    _bands(COUNTRIES), "country string, co_low double, co_high double"))
device_band = F.broadcast(spark.createDataFrame(
    _bands(DEVICES), "device string, de_low double, de_high double"))
visitor_band = F.broadcast(spark.createDataFrame(
    _bands(VISITORS), "visitor_type string, vi_low double, vi_high double"))
camp_lk = F.broadcast(spark.createDataFrame(
    [(c[0], c[1], c[2], c[3], c[4]) for c in CAMPAIGNS],
    "campaign_id string, campaign_name string, channel string, objective string, base_bid_usd double"))

# Multiplier lookup tables (as SQL CASE via F.when for compactness).
def channel_base(col):
    e = F.lit(0.02)
    for ch, v in CHANNEL_BASE_CTR.items():
        e = F.when(col == ch, F.lit(v)).otherwise(e)
    return e

def channel_cvr(col):
    e = F.lit(0.12)
    for ch, v in CHANNEL_BASE_CVR.items():
        e = F.when(col == ch, F.lit(v)).otherwise(e)
    return e

base = (
    spark.range(0, N_IMPRESSIONS, numPartitions=64)
    .withColumn("impression_id", F.format_string("IMP-%09d", F.col("id")))
    .withColumn("_r_day", F.rand(seed=11))
    .withColumn("event_date", F.date_sub(F.lit(NOW_STR), (F.col("_r_day") * F.lit(89)).cast("int")))
    .withColumn("is_drift_window", F.col("event_date") >= F.lit(DRIFT_START_STR))
    .withColumn("hour_of_day", (F.rand(seed=12) * F.lit(24)).cast("int"))
    .withColumn("day_of_week", (F.dayofweek(F.col("event_date")) - F.lit(1)))
    .withColumn("_r_camp", F.rand(seed=13))
    .withColumn("_r_co", F.rand(seed=14))
    .withColumn("_r_de", F.rand(seed=15))
    .withColumn("_r_vi", F.rand(seed=16))
    .withColumn("_r_click", F.rand(seed=17))
    .withColumn("_r_conv", F.rand(seed=18))
    .withColumn("_r_jit", (F.randn(seed=19) * F.lit(0.0015)))  # ±0.15pp daily jitter
    .withColumn("_r_aov", F.rand(seed=20))
    .withColumn("event_ts",
                F.to_timestamp(F.concat_ws(" ", F.col("event_date").cast("string"),
                               F.format_string("%02d:00:00", F.col("hour_of_day")))))
)
# Campaign pick differs pre/post drift (the budget shift). Split, join, union.
pre = (base.filter(~F.col("is_drift_window")).alias("b")
       .join(camp_band_pre.alias("cb"),
             (F.col("b._r_camp") >= F.col("cb.cp_low")) & (F.col("b._r_camp") < F.col("cb.cp_high")), "inner"))
post = (base.filter(F.col("is_drift_window")).alias("b")
        .join(camp_band_post.alias("cb"),
              (F.col("b._r_camp") >= F.col("cb.cp_low")) & (F.col("b._r_camp") < F.col("cb.cp_high")), "inner"))
imp = pre.unionByName(post)

imp = (
    imp.join(camp_lk.alias("cl"), "campaign_id", "inner")
    .join(country_band.alias("co"),
          (F.col("_r_co") >= F.col("co.co_low")) & (F.col("_r_co") < F.col("co.co_high")), "inner")
    .join(device_band.alias("de"),
          (F.col("_r_de") >= F.col("de.de_low")) & (F.col("_r_de") < F.col("de.de_high")), "inner")
    .join(visitor_band.alias("vi"),
          (F.col("_r_vi") >= F.col("vi.vi_low")) & (F.col("_r_vi") < F.col("vi.vi_high")), "inner")
)

# CTR multipliers
visitor_mult = F.when(F.col("visitor_type") == "returning", F.lit(1.4)).otherwise(F.lit(0.8))
device_mult = (F.when(F.col("device") == "Desktop", F.lit(1.15))
               .when(F.col("device") == "Mobile", F.lit(0.95)).otherwise(F.lit(0.90)))
hour_mult = (F.when(F.col("hour_of_day").between(18, 22), F.lit(1.2))
             .when(F.col("hour_of_day").between(2, 6), F.lit(0.7)).otherwise(F.lit(1.0)))
country_mult = (F.when(F.col("country").isin("US", "CA", "GB"), F.lit(1.05))
                .when(F.col("country") == "AU", F.lit(0.95)).otherwise(F.lit(1.0)))
objective_mult = (F.when(F.col("objective") == "retargeting", F.lit(1.5))
                  .when(F.col("objective") == "awareness", F.lit(0.6)).otherwise(F.lit(1.0)))

# Drift override: for drift campaigns in the drift window, ramp CTR down to ×0.55
# at DRIFT_PEAK and hold. days_from_start ∈ [0,21]; ramp reaches trough by ~day 7.
days_into_drift = F.datediff(F.col("event_date"), F.lit(DRIFT_START_STR))
drift_ramp = F.when(
    F.col("campaign_id").isin(DRIFT_CAMPAIGNS) & F.col("is_drift_window"),
    F.lit(1.0) - F.least(days_into_drift.cast("double") / F.lit(7.0), F.lit(1.0)) * F.lit(0.60)
).otherwise(F.lit(1.0))

imp = (
    imp
    .withColumn("base_ctr", channel_base(F.col("channel")))
    .withColumn("true_ctr",
                F.greatest(F.least(
                    F.col("base_ctr") * visitor_mult * device_mult * hour_mult
                    * country_mult * objective_mult * drift_ramp + F.col("_r_jit"),
                    F.lit(0.15)), F.lit(0.001)))
    .withColumn("clicked", (F.col("_r_click") < F.col("true_ctr")).cast("int"))
    .withColumn("true_cvr", channel_cvr(F.col("channel"))
                * F.when(F.col("visitor_type") == "returning", F.lit(1.3)).otherwise(F.lit(0.85))
                * F.when(F.col("objective") == "awareness", F.lit(0.6)).otherwise(F.lit(1.0)))
    .withColumn("converted",
                (F.col("clicked") == 1) & (F.col("_r_conv") < F.col("true_cvr")))
    .withColumn("converted", F.col("converted").cast("int"))
    # CPM billing (cost per impression served) — this is how awareness / display /
    # social inventory is actually bought, so spend concentrates on the high-volume
    # Social+Display awareness campaigns the mix shift tripled. Each synthetic row
    # represents a sampled block of real impressions, so the per-row cost is a scaled
    # CPM (channel-dependent; awareness inventory is expensive relative to its CTR).
    .withColumn("cpm_row",
                F.when(F.col("channel") == "Search", F.lit(0.80))
                 .when(F.col("channel") == "Social", F.lit(1.55))
                 .when(F.col("channel") == "Display", F.lit(1.70))
                 .when(F.col("channel") == "Email", F.lit(0.20))
                 .otherwise(F.lit(0.50)))
    .withColumn("cost_usd", F.round(
        F.col("cpm_row") * (F.lit(0.9) + F.rand(seed=21) * F.lit(0.2)), 5))
    .withColumn("conversion_value_usd", F.round(
        F.col("converted") * (F.lit(45.0) + F.col("_r_aov") * F.lit(75.0)), 2))
    .select(
        "impression_id", "event_ts", "event_date", "campaign_id", "channel",
        "campaign_name", "device", "country", "visitor_type", "objective",
        "hour_of_day", "day_of_week", "clicked", "converted", "cost_usd",
        "conversion_value_usd",
    )
)
_save(imp, "raw_impressions")

# ── Phase 2 — SILVER ─────────────────────────────────────────────────────────
print("Building silver_impressions …")
drift_camp_sql = ",".join(f"'{c}'" for c in DRIFT_CAMPAIGNS)
spark.sql(f"""
  CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA}.silver_impressions
  COMMENT 'Cleaned impression fact (1 row per ad impression) with derived week + drift flags. is_drift_segment marks the Social/Display prospecting cohort the campaign mix shift over-scaled.'
  AS SELECT
    impression_id, event_ts, event_date,
    date_trunc('week', event_date) AS event_week,
    campaign_id, campaign_name, channel, objective,
    device, country, visitor_type, hour_of_day, day_of_week,
    clicked, converted, cost_usd,
    conversion_value_usd,
    conversion_value_usd AS revenue_usd,
    (event_date >= DATE '{DRIFT_START_STR}') AS is_drift_window,
    (campaign_id IN ({drift_camp_sql})) AS is_drift_segment
  FROM {CATALOG}.{SCHEMA}.raw_impressions
""")

# ── Phase 3 — GOLD ───────────────────────────────────────────────────────────
print("Building gold_segment_ctr_daily …")
spark.sql(f"""
  CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA}.gold_segment_ctr_daily
  COMMENT 'Segment-day CTR actuals — the ML training source AND the actuals dashboards/predictions join against. One row per (event_date, channel, device, country, visitor_type, campaign_id).'
  AS SELECT
    event_date,
    date_trunc('week', event_date) AS event_week,
    channel, campaign_id, campaign_name, device, country, visitor_type, objective,
    COUNT(*)                                    AS impressions,
    SUM(clicked)                                AS clicks,
    SUM(converted)                              AS conversions,
    SUM(clicked) / COUNT(*)                     AS actual_ctr,
    SUM(converted) / NULLIF(SUM(clicked), 0)    AS actual_cvr,
    SUM(cost_usd)                               AS spend_usd,
    SUM(conversion_value_usd)                   AS conversion_value_usd,
    MAX(is_drift_window)                        AS is_drift_window,
    MAX(is_drift_segment)                        AS is_drift_segment
  FROM {CATALOG}.{SCHEMA}.silver_impressions
  GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9
""")

print("Building gold_campaign_daily …")
spark.sql(f"""
  CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA}.gold_campaign_daily
  COMMENT 'Daily rollup per campaign/channel — powers KPI counters, channel-mix, and blended-CTR trend widgets.'
  AS SELECT
    event_date,
    date_trunc('week', event_date) AS event_week,
    channel, campaign_id, campaign_name, objective,
    COUNT(*)                        AS impressions,
    SUM(clicked)                    AS clicks,
    SUM(converted)                  AS conversions,
    SUM(clicked) / COUNT(*)         AS blended_ctr,
    SUM(cost_usd)                   AS spend_usd,
    SUM(conversion_value_usd)       AS conversion_value_usd,
    MAX(is_drift_window)            AS is_drift_window,
    MAX(is_drift_segment)           AS is_drift_segment
  FROM {CATALOG}.{SCHEMA}.silver_impressions
  GROUP BY 1, 2, 3, 4, 5, 6
""")

# ── Validation ───────────────────────────────────────────────────────────────
print("\n── Validation ──")
seg = spark.table(f"{CATALOG}.{SCHEMA}.gold_segment_ctr_daily")
n_seg = seg.select("channel", "device", "country", "visitor_type", "campaign_id").distinct().count()
print(f"distinct segments: {n_seg:,}")

drift_ctr = seg.filter(F.col("is_drift_segment") & F.col("is_drift_window")).agg(
    (F.sum("clicks") / F.sum("impressions")).alias("ctr")).collect()[0]["ctr"]
predrift_ctr = seg.filter(F.col("is_drift_segment") & ~F.col("is_drift_window")).agg(
    (F.sum("clicks") / F.sum("impressions")).alias("ctr")).collect()[0]["ctr"]
print(f"drift-segment CTR  pre-drift={predrift_ctr:.4f}  drift-window={drift_ctr:.4f}")

# Channel mix shift
mix = spark.sql(f"""
  SELECT is_drift_window,
    SUM(CASE WHEN channel IN ('Social','Display') THEN impressions ELSE 0 END)/SUM(impressions) AS soc_disp_share
  FROM {CATALOG}.{SCHEMA}.gold_campaign_daily GROUP BY 1 ORDER BY 1
""").collect()
for r in mix:
    print(f"  drift_window={r['is_drift_window']}  Social+Display impression share={r['soc_disp_share']:.3f}")

print(f"\nDone. DRIFT_START={DRIFT_START_STR}  DRIFT_PEAK={DRIFT_PEAK.date()}")
