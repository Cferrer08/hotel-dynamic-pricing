# Power BI Dashboard — Build Guide

Same four pages as the Streamlit app, built in Power BI Desktop.

**Prerequisite:** run the export first.

```powershell
python src\export_powerbi.py
```

That writes eight CSVs to `data\powerbi\`. Verify the reconciliation block matches
your Streamlit figures before continuing — if they disagree, the dashboard is
already wrong and no amount of formatting will fix it.

```
Open nights       : 885
Baseline revenue  : $10,483,387
Optimized revenue : $11,508,395
Uplift            : $1,025,007 (+9.78%)
```

---

## Step 1 — Load the data

1. Open **Power BI Desktop** → **Get Data** → **Text/CSV**
2. Load all eight files from `data\powerbi\`:
  `dim_hotel`, `dim_room_type`, `dim_date`, `fact_recommendation`,
   `fact_scenario`, `fact_history`, `model_metrics`, `strategy_comparison`
3. For each, click **Transform Data** rather than Load, and check the column types
  in Power Query. Power BI guesses, and it guesses wrong on dates roughly half the
   time.
4. Confirm `stay_date` and `date` are **Date**, not Text. Fix via right-click →
  **Change Type** → **Date**.
5. **Close & Apply**

> **Why not connect straight to PostgreSQL?** You can — Get Data → PostgreSQL,
> server `localhost:5433`. But the CSVs are what the models actually produced.
> Reading them guarantees the dashboard agrees with your notebooks. A live database
> connection is the better choice once the pipeline is scheduled; for a portfolio
> piece, reproducibility wins.

---



## Step 2 — Build the model

Go to **Model view** (left sidebar, third icon).

Create these relationships by dragging field to field:


| From                          | To                                  | Cardinality | Direction |
| ----------------------------- | ----------------------------------- | ----------- | --------- |
| `dim_hotel[hotel_id]`         | `fact_recommendation[hotel_id]`     | One to many | Single    |
| `dim_hotel[hotel_id]`         | `fact_history[hotel_id]`            | One to many | Single    |
| `dim_hotel[hotel_id]`         | `fact_scenario[hotel_id]`           | One to many | Single    |
| `dim_room_type[room_type_id]` | `fact_recommendation[room_type_id]` | One to many | Single    |
| `dim_room_type[room_type_id]` | `fact_history[room_type_id]`        | One to many | Single    |
| `dim_room_type[room_type_id]` | `fact_scenario[room_type_id]`       | One to many | Single    |
| `dim_date[date]`              | `fact_recommendation[stay_date]`    | One to many | Single    |
| `dim_date[date]`              | `fact_history[stay_date]`           | One to many | Single    |
| `dim_date[date]`              | `fact_scenario[stay_date]`          | One to many | Single    |


**Then mark the date table.** Select `dim_date` → **Table tools** → **Mark as date
table** → choose `date`. Without this, time intelligence functions silently return
wrong results.

**Set the sort order for room types.** Select `dim_room_type[room_type_name]` →
**Column tools** → **Sort by column** → `sort_order`. Otherwise Power BI sorts
alphabetically and you get *Deluxe, Standard, Suite*, which is nobody's mental model.

Do the same for `dim_date[day_name]` sorted by `day_sort`.

### Why a star schema rather than one flat table

Power BI's engine is built for one-to-many relationships from dimensions to facts.
A single wide table technically works, but slicers stop propagating predictably
across visuals and cross-filtering breaks in ways that are hard to debug. The
schema from Phase 2 maps onto this directly — that groundwork pays off here.

---



## Step 3 — Add the measures

1. **Home** → **Enter Data** → name the table `_Measures` → **Load**
  (delete the placeholder column; an empty table is fine)
2. Open `dashboard\powerbi_measures.dax`
3. For each measure: **Modeling** → **New Measure**, paste, press Enter
4. In the field list, drag each new measure into `_Measures`

Housing measures in their own table keeps them together instead of scattered across
fact tables.

### The one rule that matters most

**Never average a stored ratio.** Always divide a summed numerator by a summed
denominator.

```dax
Occupancy On Books = DIVIDE ( [Rooms On Books], [Capacity] )   -- correct
Occupancy WRONG    = AVERAGE ( fact_recommendation[baseline_occupancy] )
```

In this dataset the wrong version is off by up to **4.3 percentage points** at hotel
level, because it weights a 13-room suite the same as a 154-room standard block.
That is a visibly wrong number on a slide, and it is the single most common Power BI
error.

This is also why the export ships `rooms_on_books_now` and `capacity` rather than a
precomputed `occupancy_pct` column. Exporting the ratio invites the mistake.

---



## Step 4 — Page 1: Executive overview

**Cards** (Insert → Card), across the top:


| Card                     | Measure             | Format                         |
| ------------------------ | ------------------- | ------------------------------ |
| Nights priced            | `Nights Priced`     | whole number                   |
| Revenue at current rates | `Baseline Revenue`  | currency, 2 decimals, millions |
| Revenue if optimized     | `Optimized Revenue` | currency                       |
| Opportunity              | `Revenue Uplift`    | currency                       |
| Uplift %                 | `Revenue Uplift %`  | percentage, 1 decimal          |


**Line chart** — opportunity over time:

- X axis: `dim_date[date]`
- Y axis: `Baseline Revenue` and `Optimized Revenue`
- Title: "Revenue at stake by stay date"

**Clustered bar** — where the opportunity sits:

- Y axis: `fact_recommendation[occupancy_band]`
- X axis: `Revenue Uplift`

**Table** — portfolio performance:

- Rows: `dim_hotel[hotel_name]`
- Values: `ADR`, `Historical Occupancy`, `RevPAR`
- Sort by RevPAR descending

**Table** — strategy comparison from `strategy_comparison`.

**Text box** at the bottom — do not skip this:

> All figures are simulated under the project's demand model. Uplift is a model
> estimate, not a measured result. A live A/B test on held-out stay dates is
> required to validate.

**Slicers** down the left: `dim_hotel[hotel_name]`, `dim_room_type[room_type_name]`,
`dim_date[date]` (set to Between).

---



## Step 5 — Page 2: Recommendations

**Table** — the main grid:


| Column      | Source                           |
| ----------- | -------------------------------- |
| Stay date   | `fact_recommendation[stay_date]` |
| Hotel       | `dim_hotel[hotel_name]`          |
| Room type   | `dim_room_type[room_type_name]`  |
| Current     | `Current Rate`                   |
| Recommended | `Recommended Rate`               |
| Change %    | `Rate Change %`                  |
| Occupancy   | `Occupancy On Books`             |
| Uplift      | `Revenue Uplift`                 |


**Round rates to whole dollars.** Select the column → **Format** → 0 decimal places.
Phase 7 found the revenue optimum is flat — on 68% of nights, three or more candidate
rates land within 1% of best. Showing $478.31 implies precision the model does not
have.

**Conditional formatting** on Change %:
Format → Cell elements → Background color → Format style: **Field value** →
`Action Colour`

**Cards** for `Constrained by Policy %` and `Flat Optimum %`.

**Slicer** on `fact_recommendation[action]` (Increase / Decrease / Hold).

**Detail panel** — select a row and these update:

- Cards: `Mean Elasticity`, `Occupancy On Books`, `Cost of Inaction`
- Line chart from `fact_scenario`: X = `rate`, Y = `expected_revenue`, filtered to
the selected night

To make the chart respond to table selection, select the table →
**Format** → **Edit interactions** → set the chart to **Filter**.

---



## Step 6 — Page 3: Rate simulator

1. **Modeling** → **New Parameter** → **Numeric range**
  - Name: `Rate Adjustment`
  - Minimum −20, Maximum 25, Increment 2.5, Default 0
  - Tick "Add slicer to this page"
2. Add cards: `Scenario Revenue`, `Scenario Pickup`, `Scenario vs Current`
3. Add slicers for hotel, room type, and stay date so a single night is selected
4. Line chart from `fact_scenario`: X = `rate`, Y = `expected_revenue`

The parameter creates a **disconnected table** — no relationship to the facts, so the
slicer filters nothing directly. The `Scenario Revenue` measure reads its value with
`SELECTEDVALUE()` and finds the nearest precomputed scenario. This is the standard
what-if pattern in Power BI and worth understanding: the slicer is an *input*, not a
filter.

---



## Step 7 — Page 4: Model performance

All visuals read `model_metrics`, filtered by its `category` column.

**Demand models** — bar chart:

- Filter: `category = "Demand model"`, `metric = "poisson_deviance"`
- Axis: `item`, Value: `Metric Value`

**Elasticity recovery** — scatter:

- Filter: `category = "Elasticity"`
- This needs a small reshape; simplest is a matrix with `item` on rows and `metric`
on columns, values `Metric Value`

**Robustness** — bar chart:

- Filter: `category = "Robustness"`, `metric = "mean_elasticity"`
- Add a constant line at −1.471 (the true value)

Add a text box explaining the placebo row: applying the identical estimator to
non-randomized rates labelled as an experiment returns the **wrong sign** — which is
what proves the result comes from the experimental design rather than the estimator.

**SHAP drivers** — horizontal bar:

- Filter: `category = "SHAP"`
- Axis: `item`, Value: `Metric Value`, Legend: `metric`

---



## Step 8 — Polish

- **View** → **Themes** → pick one, or build a custom theme with `#2E5E4E` accent
- Give every page a title text box
- **View** → **Page view** → **Fit to page**
- **File** → **Options** → **Report settings** → tick "Enable persistent filters"
- Save as `dashboard\hotel_pricing.pbix`



### Publishing (optional)

**Publish** → sign in with a Microsoft work account → publish to *My workspace* →
share the public link in your README. Requires a Power BI account; the free tier is
enough for a portfolio.

If you would rather not, a few screenshots in the README work fine. The `.pbix` file
in the repo is the artefact that proves you built it.

---



## Reconciliation check

Before you call it finished, confirm these match the Streamlit dashboard exactly:


| Measure              | Expected    |
| -------------------- | ----------- |
| Nights Priced        | 885         |
| Baseline Revenue     | $10,483,387 |
| Optimized Revenue    | $11,508,395 |
| Revenue Uplift %     | +9.78%      |
| Historical Occupancy | 82.1%       |
| ADR                  | $290.30     |
| RevPAR               | $238.30     |


If occupancy comes out near 1.4%, you have summed `capacity` across booking-date rows
— the fan-out bug from Phase 3. The export already aggregates to the correct grain,
so this should not happen, but it is the first thing to check if a ratio looks absurd.

---



## Why build both dashboards

Worth a line in your README, because "I built the same thing twice" needs a reason:

- **Streamlit** — Python-native, version-controlled, runs the models directly.
What a data scientist ships.
- **Power BI** — what most organisations actually have deployed, and what business
users can self-serve from.

Both read the same outputs and produce identical numbers. That reconciliation is
itself the point: the analysis lives in the pipeline, not in the presentation layer.