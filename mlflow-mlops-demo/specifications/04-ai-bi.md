# AI/BI — Dashboard + Genie

Tables/columns referenced here are defined in `01-lakeflow.md` (`gold_segment_ctr_daily`, `gold_campaign_daily`) and `03-ml-ctr.md` (`gold_ctr_predictions`). All FQNs under `solution_builder.demo_mlflow_logged_ctr_optimization`.

> **Talking-track-only products in the README** — do NOT build resources for these: **Genie One** (business-user surface — appears once the Genie space exists), **Genie Code** (authoring assist inside the notebook/SQL editor), **Unity Catalog** (workspace governance, grants applied in build), **Lakeflow Connect** (ingest narrative).

---

## A. Genie Space

**Skill**: `databricks-genie` — read `SKILLS/databricks-genie/SKILL.md` first.

Create **`NorthPeak CTR Optimization Analytics`** Genie Space.

### Tables
`gold_segment_ctr_daily` (segment-level actuals + drift flags), `gold_campaign_daily` (channel/campaign trends + KPIs), `gold_ctr_predictions` (predicted-vs-actual for both models, gaps, CTR lift), `raw_campaigns` (campaign catalog / objectives).

### Self-sufficient room
- **Space `description`** (via `PATCH /api/2.0/genie/spaces/<id>`): 1–3 sentences naming the event (campaign mix shift 3 weeks ago → Social/Display CTR collapsed → stale model over-bid → $240K wasted spend → retrained champion recovers it) and pointing to the suggested questions in order.
- **Story-context `text_instruction`** at TOP of `instructions.text_instructions[]`: WHAT HAPPENED · WHO (Maya, Growth Analytics Lead) · WHAT TO HELP HER DO · TONE.
- **`sample_questions`** chips + matching `example_question_sqls` walk the arc.

### Instructions (lift into text_instruction)
```
You analyze NorthPeak Goods paid-acquisition CTR data for Maya (Growth Analytics Lead).

BASELINES: blended CTR ~2.4%. A campaign mix shift 21 days ago moved budget from Search to Social + Display; those prospecting segments' true CTR fell to ~1.2%. The stale production model still predicted ~2.4% for them → over-bidding → ~$240K wasted spend. A retrained champion model tracks actuals again.

KEY METRICS:
- CTR = clicks / impressions (from gold_segment_ctr_daily or gold_campaign_daily)
- Predicted vs actual gap = predicted_ctr_stale - actual_ctr (gold_ctr_predictions) — large+positive on drift segments
- CTR lift of champion = the champion now matches actuals where the stale model didn't
- Wasted spend ~ spend on drift segments the stale model over-valued (~$240K over the drift window)

INVESTIGATION FLOW for "why did CTR accuracy drift?":
1. gold_campaign_daily → weekly impression share by channel → Social+Display jump from ~20% to ~50% (the mix shift)
2. gold_segment_ctr_daily → weekly actual_ctr for Social/Display segments → collapses from ~2.4% to ~1.2%
3. gold_ctr_predictions → ctr_gap_stale by channel/segment → stale model over-predicts most on Social/Display
4. gold_ctr_predictions → ctr_gap_champion ≈ 0 → the retrained champion fixed it
```

### Sample Questions (chips, arc order)
1. **Drift** — "Which channels have the biggest gap between predicted and actual CTR in the last 3 weeks?"
2. **Mix shift** — "How did the impression mix by channel change over the last 90 days?"
3. **CTR collapse** — "Show weekly actual CTR for Social and Display segments."
4. **Wasted spend** — "How much wasted spend did the CTR drift cause?"
5. **Champion lift** — "By channel, how does the champion model's prediction gap compare to the stale model's?"
6. **Recovery** — "Is the champion model's predicted CTR now tracking actuals?"

Curated `example_question_sqls` (3 load-bearing): **Drift** (ctr_gap_stale by channel, drift window), **Mix shift** (weekly impression share by channel), **Champion lift** (avg ABS(ctr_gap_stale) vs ABS(ctr_gap_champion) by channel).

### Validation
- Q1 → Social + Display surface with the largest `ctr_gap_stale`.
- Q4 → returns a value in the $200K–$280K band.
- Q6 → champion gap ≈ 0 while stale gap is large on drift segments.

Add `genie_space_id` to `resources.json`.

---

## B. Dashboard

**Skill**: `databricks-aibi-dashboards` — read `SKILLS/databricks-aibi-dashboards/SKILL.md` first. It owns JSON shape, encoding, grid math; this spec is WHAT.

Create **`NorthPeak CTR Optimization`** dashboard. Save locally as `PROJECT/dashboard.json`. Link the Genie space. Set `--dataset-catalog solution_builder --dataset-schema demo_mlflow_logged_ctr_optimization` on `lakeview create` AND `update`.

### Design principles
- **Two pages**: Page 1 **Spend & Drift** (the glance — "accuracy drifted, $240K wasted, here's where"); Page 2 **Model & Recovery** (the MLflow payoff — predicted-vs-actual for both models, CTR lift by segment, recovery).
- **5-second test**: predicted-vs-actual CTR divergence on Page 1 is unmissable; the champion line snapping back onto actuals on Page 2 is the wow.
- Row 1 of each page = a markdown `text` widget naming the event + touring the page.

### Theme
```
canvasBackgroundColor: #F5F7FB / #0F1419
widgetBackgroundColor: #FFFFFF / #161B22
widgetBorderColor:     same as widgetBackgroundColor (no visible border)
fontColor:             #1F2530 / #E8ECF0
selectionColor:        #4F7CE3 / #8ACAFF
visualizationColors:   ["#094074","#3C6997","#5ADBFF","#FFDD4A","#FE9000"]
widgetHeaderAlignment: LEFT
```
**Semantic pins (literal-hex, never `themeColorType`):** Actual CTR → `#094074` navy · Predicted (stale) → `#FE9000` orange (the "wrong" line) · Predicted (champion) → `#5ADBFF` cyan (the "fixed" line). **Channel pins** (consistent across widgets): Search `#094074`, Email `#3C6997`, Affiliate `#5ADBFF`, Social `#FFDD4A`, Display `#FE9000`.

### Datasets
| Name | Source | Powers |
|---|---|---|
| `ds_kpi` | aggregates from `gold_campaign_daily` + a wasted-spend rollup | KPI counters (blended CTR, wasted spend, impressions, CVR) |
| `ds_channel_mix` | weekly impression share + CTR by channel from `gold_campaign_daily` | channel-mix area/bar + weekly CTR-by-channel line |
| `ds_pred_actual` | weekly avg `actual_ctr`, `predicted_ctr_stale`, `predicted_ctr_champion` from `gold_ctr_predictions` (optionally joined to gold for weeks) | predicted-vs-actual trend lines (both pages) |
| `ds_segment_gap` | segment-level `ctr_gap_stale`, `ctr_gap_champion`, `spend_usd`, wasted-spend from `gold_ctr_predictions`, drift window | gap-by-channel bars, CTR-lift bars, top-wasted-segments table |

Global filters (left panel): **Date Range** (`event_date`/`event_week`), **Channel**, **Device**, **Country** — bound to the datasets above.

### Page 1 — Spend & Drift (the glance)
12-col grid; `(x,y,w,h)`:
| y | x | w | h | Widget |
|---|---|---|---|---|
| 0 | 0 | 12 | 3 | `title` (markdown — event + tour) |
| 3 | 0 | 3 | 3 | `kpi_wasted_spend` (`SUM(wasted_spend_usd)` ≈ $240K, currency compact, color `#FE9000`) |
| 3 | 3 | 3 | 3 | `kpi_blended_ctr` (blended CTR %, color `#094074`) |
| 3 | 6 | 3 | 3 | `kpi_impressions` (SUM impressions, compact) |
| 3 | 9 | 3 | 3 | `kpi_cvr` (blended CVR %) |
| 6 | 0 | 12 | 5 | `pred_actual_trend` (line: weekly `actual_ctr` navy vs `predicted_ctr_stale` orange — the divergence opening ~3 weeks ago; vertical annotation at `DRIFT_START` = "Campaign mix shift") |
| 11 | 0 | 7 | 6 | `channel_mix_area` (area: weekly impression share by channel — Social+Display swelling; channel pins) |
| 11 | 7 | 5 | 6 | `ctr_by_channel_bar` (bar: avg CTR by channel, drift window vs prior — Social/Display collapse; channel pins) |
| 17 | 0 | 12 | 5 | `wasted_by_channel_bar` (horizontal bar: `SUM(wasted_spend_usd)` by channel — Social + Display dominate) |

**`title`** ~5 lines: what happened (mix shift 3w ago) · effect (Social/Display CTR fell to ~1.2%, stale model over-bid) · impact ($240K wasted) · what to see (top line = predicted-vs-actual divergence; area = mix swelling; bars = where the waste is).
**`pred_actual_trend`** frame description: *"The stale model's prediction (orange) keeps predicting ~2.4% while actual CTR (navy) falls — the gap is the wasted spend. Vertical bar = the campaign mix shift."*

### Page 2 — Model & Recovery (the MLflow payoff)
| y | x | w | h | Widget |
|---|---|---|---|---|
| 0 | 0 | 12 | 3 | `title2` (markdown) |
| 3 | 0 | 12 | 5 | `both_models_trend` (line: weekly `actual_ctr` navy + `predicted_ctr_stale` orange + `predicted_ctr_champion` cyan — champion snaps onto actuals; annotation at retrain point) |
| 8 | 0 | 6 | 6 | `gap_by_channel` (grouped bar: avg ABS gap by channel — stale (orange) tall vs champion (cyan) near zero) |
| 8 | 6 | 6 | 6 | `ctr_lift_bar` (bar: CTR-accuracy lift = ABS(stale gap) − ABS(champion gap) by channel — biggest on Social/Display) |
| 14 | 0 | 12 | 6 | `top_segments_table` (table: worst-mis-bid segments — channel/campaign/device/country, actual_ctr, predicted_ctr_stale, predicted_ctr_champion, ctr_gap_stale, wasted_spend_usd; sort ctr_gap_stale DESC) |

**`title2`** ~4 lines: the retrained champion (registered in UC, `@champion` alias) now tracks actuals · gap-by-channel shows stale vs champion · lift bars quantify the fix · table lists the exact mis-bid segments the retrain corrected. Talking track: MLflow experiment leaderboard + feature importance shown live in the notebook/experiment UI.
**`both_models_trend`** frame description: *"Champion (cyan) tracks actual CTR (navy) even through the drift window, where the stale model (orange) stayed wrong. That's the retrain closing the $240K gap."*

### Validation
Published dashboard reads at a glance: predicted-vs-actual divergence obvious on Page 1, wasted-spend KPI ≈ $240K, champion line snapping onto actuals on Page 2, CTR-lift bars biggest on Social/Display, filters update every widget. Add `dashboard_id` to `resources.json`.
