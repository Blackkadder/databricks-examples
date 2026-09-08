# Lakeflow — Data Generation + Small Transformation Chain

> **Simple-demo contract (no SDP).** One self-contained, idempotent data-generation script produces the full **raw → silver → gold** layering inline: the raw impression/campaign source tables, a cleaned/enriched silver impression fact, and the gold tables the ML notebook, dashboard, and Genie read. There is **no SDP** in this build — the script does the layering with `spark.range` generation + `spark.sql` CTAS. **No metric view.** Talking track: *"in production Lakeflow Connect pulls the ad-platform + web-analytics feeds and SDP shapes them — here the data-gen does the equivalent layering inline so the demo lands in minutes."*

**Skill**: `databricks-synthetic-data-gen` — read `SKILLS/databricks-synthetic-data-gen/SKILL.md` first.
**Runtime**: pre-provisioned databricks-connect venv (path in system prompt). Do NOT create a new venv.
**Location**: `solution_builder.demo_mlflow_logged_ctr_optimization` (catalog.schema from `resources.json`).
**Script path**: `PROJECT/src/notebooks/generate_data.py`.

---

## Shared Context (defined once — every later spec references these)

**Company / persona**: NorthPeak Goods (mid-market e-commerce, home & outdoor). Maya Patel, Growth Analytics Lead.

**The metric**: **CTR** (click-through rate = clicks / impressions) is the primary regression target. **CVR** (conversion rate = conversions / clicks) is the secondary metric shown on dashboards.

**Channels** (5): `Search`, `Social`, `Display`, `Email`, `Affiliate`.
**Devices** (3): `Desktop`, `Mobile`, `Tablet`.
**Visitor type** (2): `new`, `returning`.
**Countries** (6): `US`, `CA`, `GB`, `DE`, `FR`, `AU`.
**Campaigns** (~12): named per channel, e.g. `SEA-Brand-Core`, `SEA-NonBrand-Generic`, `SOC-Prospecting-Lookalike`, `SOC-Retargeting-Cart`, `DIS-Programmatic-RON`, `DIS-Contextual-Home`, `EML-Newsletter`, `EML-Winback`, `AFF-Coupon-Sites`, `AFF-Content-Blogs`, `SEA-Shopping-PLA`, `SOC-Video-Awareness`. Each campaign belongs to exactly one channel.

**Segment grain** (the unit of prediction): `(channel, device, country, visitor_type, campaign)`. ~2,400 distinct segments exist across the data.

**Time anchors** — `NOW = datetime.now()` by default (rolling window; dashboard right edge is always ~yesterday). Set `NPK_PIN_TIME=1` to freeze `NOW` for reproducible runs.
- `STORY_END_DATE = NOW`
- `STORY_START_DATE = NOW − 90 days` (~2.0M impressions over the window)
- `DRIFT_START = NOW − 21 days` (the campaign mix shift — marketing rebudgets Search → Social + Display)
- `DRIFT_PEAK = NOW − 14 days` (worst prediction gap)
- `DECAY_START = NOW − 10 days` (nothing recovers on its own until the model is retrained — the gap stays elevated through NOW; the *retrain* closes it, which the demo does live)

**The catalyst — the campaign mix shift (load-bearing block).**
- **Before `DRIFT_START`** (first ~69 days): stable, healthy mix. Budget weighting ≈ Search 45% / Email 20% / Affiliate 15% / Social 12% / Display 8%. Blended CTR ≈ **2.4%**. Each segment's true CTR is a stable function of its features (see § CTR generative model). This is the period the **stale production model** was trained on.
- **At `DRIFT_START`** (last 21 days): marketing shifts budget → Search 25% / Email 15% / Affiliate 10% / **Social 30% / Display 20%**. The newly-scaled Social + Display prospecting campaigns (`SOC-Prospecting-Lookalike`, `SOC-Video-Awareness`, `DIS-Programmatic-RON`, `DIS-Contextual-RON`) reach colder, lower-intent audiences → their **true CTR collapses from ~2.4% toward ~1.3%**. Because impression volume there tripled, the **blended CTR** drops visibly too.
- **The waste**: the stale model — trained on the pre-drift world where Social/Display ran small, warm retargeting only — still predicts ~2.4% CTR for these segments, so the bidding system over-bids on them. Over-spend accrues across the 21-day window and totals **≈ $240K** (see § Wasted-spend math). This must be a clearly visible divergence, not noise.

**Signal-to-noise commitment**: the predicted-vs-actual CTR gap on Social/Display must **dominate** normal day-to-day CTR variance. Keep per-segment daily CTR noise tight (±0.15pp gaussian) so the ~1.1pp drop reads instantly on the trend line. Peak gap sits ~2 weeks in the past with a build-up from `DRIFT_START`; it does NOT sit at the chart's right edge as a cliff — it plateaus into `NOW` (unresolved until retrain).

---

## A. Data Generation

Generate impression-level events, then roll up. Use `spark.range` + `F.when`/`F.rand` for volume; broadcast small dimension frames.

### Raw tables

- **`raw_campaigns`** ~12 rows — hand-curated (NOT generated): `campaign_id` (`CMP-NNN`), `campaign_name` (from list above), `channel`, `objective` (`conversion`/`awareness`/`retargeting`), `launch_date`, `base_bid_usd` (avg CPC bid, $0.30–$2.20 by channel; Search highest, Display lowest). Fixed so downstream specs validate campaigns by name.
- **`raw_impressions`** ~2.0M — one row per ad impression: `impression_id` (PK, `IMP-<n>`), `event_ts` (TIMESTAMP within the 90-day window), `event_date` (DATE), `campaign_id` (FK), `channel`, `campaign_name`, `device`, `country`, `visitor_type`, `hour_of_day` (0–23), `day_of_week` (0–6), `clicked` (INT 0/1), `converted` (INT 0/1; can only be 1 if `clicked=1`), `cost_usd` (DOUBLE — what was paid to serve/bid this impression; ≈ `base_bid_usd × clicked` + small serving cost, so cost concentrates on clicks), `conversion_value_usd` (DOUBLE — order value if `converted=1`, else 0; ~$45–$120 AOV).

### CTR generative model (how `clicked` is drawn)

Each impression's click probability = a **base CTR by segment features**, so the model has real signal to learn:
- **Channel base**: Search 3.2% · Email 2.8% · Affiliate 2.2% · Social 2.0% · Display 1.2% (pre-drift warm audiences).
- **Visitor**: `returning` ×1.4, `new` ×0.8.
- **Device**: Desktop ×1.15, Mobile ×0.95, Tablet ×0.90.
- **Hour-of-day**: mild peak 18:00–22:00 (×1.2), trough 02:00–06:00 (×0.7).
- **Country**: US/CA/GB ×1.05, DE/FR ×1.0, AU ×0.95.
- **Campaign objective**: `retargeting` ×1.5, `conversion` ×1.0, `awareness` ×0.6.
- Draw `clicked ~ Bernoulli(clip(base_ctr × multipliers, 0.001, 0.15))` with ±0.15pp gaussian jitter on the daily segment mean.
- **CVR** (given click): ~14% blended; higher for `returning` + `retargeting`, lower for `awareness`. `converted = clicked & Bernoulli(cvr)`.

**Drift override (last 21 days, Social + Display prospecting/awareness campaigns only)**: multiply those impressions' base CTR by a ramp that reaches **×0.55** at `DRIFT_PEAK` and holds through `NOW` (true CTR ≈1.1–1.3% vs ~2.4% the stale model expects). Simultaneously **triple the impression volume** on those campaigns from `DRIFT_START` onward (the budget shift). Non-drift segments keep their stable CTR.

### Silver table

- **`silver_impressions`** ~2.0M — cleaned impression fact, one row per impression with all raw columns plus derived: `event_week` (Monday of week), `is_drift_window` (BOOL = `event_date >= DRIFT_START`), `is_drift_segment` (BOOL = channel in (Social,Display) AND objective in (awareness,retargeting-prospecting) — the mis-bid cohort), `revenue_usd` alias of `conversion_value_usd`. COMMENT every column.

### Gold tables (what ML + dashboard + Genie read)

- **`gold_segment_ctr_daily`** ~ (2,400 segments × active days, ~150K rows) — **one row per `(event_date, channel, device, country, visitor_type, campaign_id)`**. Columns: `event_date`, `event_week`, `channel`, `campaign_id`, `campaign_name`, `device`, `country`, `visitor_type`, `objective`, `impressions` (COUNT), `clicks` (SUM clicked), `conversions` (SUM converted), `actual_ctr` (clicks/impressions), `actual_cvr` (conversions/NULLIF(clicks,0)), `spend_usd` (SUM cost_usd), `conversion_value_usd` (SUM), `is_drift_window`, `is_drift_segment`. This is the **training source** (pre-drift rows) AND the actuals the dashboard/predictions join against. COMMENT every column.
- **`gold_campaign_daily`** ~ (12 campaigns × 90 days × channels, small) — daily rollup per `(event_date, event_week, channel, campaign_id, campaign_name, objective)`: `impressions`, `clicks`, `conversions`, `blended_ctr`, `spend_usd`, `conversion_value_usd`. Powers the trend/KPI widgets and the channel-mix view. COMMENT every column.

> `gold_ctr_predictions` is **created by the ML notebook** (`03-ml-ctr.md`), not the data-gen script — it depends on the trained model. The data-gen script must leave the gold actuals tables in place for the notebook to read.

---

## B. Wasted-spend math (the $240K)

Compute a `wasted_spend_usd` figure so the dashboard KPI and README agree. Definition, per **drift-segment** row in the drift window (the mis-bid Social/Display cohort):
`wasted_spend ≈ spend_usd × (1 − actual_ctr / baseline_ctr)` clipped ≥ 0, where `baseline_ctr` is that campaign's pre-drift CTR (`SUM(clicks)/SUM(impressions)` over `is_drift_window = FALSE`). This is the spend that bought clicks at the pre-drift rate the bidding system still assumed. Scoped to `is_drift_segment = TRUE AND is_drift_window = TRUE`, the SUM ≈ **$242K** (story target $240K; band $200K–$280K). The dashboard KPI and the "wasted spend by channel" bar both use this exact definition so they agree.

---

## C. Validation (one-line queries; fix synth before writing 03/04)

**Load-bearing (gate the story):**
- **Drift visible** — weekly `SUM(clicks)/SUM(impressions)` on drift segments from `gold_segment_ctr_daily`: ~2.4% before `DRIFT_START`, dropping to ~1.2–1.3% by `DRIFT_PEAK`, staying low through `NOW`. Non-drift segments hold ~2.4% flat.
- **Blended CTR dip** — `gold_campaign_daily` weekly blended CTR shows a visible dip over the last 3 weeks (Social/Display volume triples and drags the blend down).
- **Channel-mix shift** — `gold_campaign_daily`: Social+Display share of impressions jumps from ~20% (pre-drift) to ~50% (drift window).
- **Wasted spend** — the § B computation over the drift window totals **$200K–$280K**.
- **Segment count** — `COUNT(DISTINCT (channel,device,country,visitor_type,campaign_id))` ≈ 2,400.
- **Row integrity** — `converted=1` implies `clicked=1` for all rows; `actual_ctr` in [0,0.15].

**Smoke checks (LLM derives):** ~2.0M impression rows · `gold_segment_ctr_daily` non-empty for every day in window · every campaign maps to exactly one channel · countries cover all 6 · no NULL in join keys.

Surface the resolved `DRIFT_START` / `DRIFT_PEAK` dates (notebook exit JSON or `resources.json`) so `03-ml-ctr.md` and `04-ai-bi.md` reference the same anchors.
