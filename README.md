# Dynamic Pricing Intelligence Platform

**Hotel revenue optimization built on causal inference, not just prediction.**

A production-shaped pricing system for a five-hotel portfolio: it forecasts demand,
measures how guests actually respond to price, and recommends a rate for every unsold
night — with an explanation a revenue manager can argue with.

---

## The finding that drove the project

The obvious way to learn how price affects demand is to regress bookings on rate.
Here is what that gives you, using two years of real booking history:

| Method | Estimated elasticity | Error vs truth |
|---|---|---|
| Naive regression on all data | **+0.56** | 2.03 |
| Add lead-time controls | +0.13 | 1.60 |
| Randomized rate tests only | −0.87 | 0.60 |
| Drop censored (sold-out) rows | −1.32 | 0.25 |
| **Add competitor-rate control** | **−1.49** | **0.13** |
| *True value* | *−1.47* | |

The naive model says **raising prices increases bookings**. Not a small error — the
wrong sign, and a system built on it would raise rates forever.

The cause is not a modelling failure. It is the data. The hotel's legacy pricing rule
raised rates on nights it expected to be busy, so high prices and high demand always
arrived together. That upward confounding cancels the true downward demand response,
and no amount of model capacity recovers it.

The fix is design, not algorithms: roughly 12% of stay dates had their rate set
**randomly**, ignoring demand. On those nights price is independent of demand by
construction, and the association *is* the causal effect.

**The perverse implication, worth stating plainly: the better the old pricing system
was, the less its data can teach you.** Good rules destroy the price variation you
need to learn from.

---

## Results

| | |
|---|---|
| Causal elasticity recovered to within | **0.13** of ground truth |
| 95% confidence-interval coverage | **93%** (14/15 segments) |
| Demand model (XGBoost) night-level MAE | **5.4 rooms** |
| Simulated revenue uplift on 885 open nights | **+9.8%** ($1.03M) |
| Recommendations sitting on a policy cap | **56%** |

> **On that uplift figure.** It is *simulated* under this project's own demand model,
> and the baseline it is compared against is also a model output. If the demand model
> is optimistic about how many rooms a discount attracts, both numbers move together.
> The only way to know the real figure is a live A/B test on held-out stay dates.
> Quoted as simulated throughout; never as achieved.

---

## Why hotel pricing is not retail pricing

Most dynamic-pricing projects model `date, product, price, units_sold`. That grain is
wrong for hotels, and getting it right shapes everything downstream.

A hotel does not sell a room on one day. It sells **a future night**, repeatedly, over
a booking window. On 30 July you might sell four rooms for the night of 12 September;
tomorrow you sell three more for that same night.

So one row here is a **(stay_date, booking_date) pair** — the *booking curve*. This
makes lead time a feature, makes **pace** (how fast a night is filling versus normal)
the central signal, and leaves the last 59 stay dates genuinely unsold. The final
pricing problem is live, not a backtest.

Three consequences the system has to respect:

- **Fixed capacity.** You cannot restock, and an empty room tonight is revenue that no
  longer exists.
- **Censoring.** Once a room type sells out, later rows show zero bookings — not
  because nobody wanted the room, but because nothing was on sale. Those are not
  demand observations. They are 1.2% of rows and including them moves the elasticity
  estimate from −1.50 to −0.93.
- **Segment-relative pace.** Business hotels take 28% of bookings in the final week;
  leisure hotels take 13%. A resort at 40% occupancy 30 days out is in trouble; a
  business hotel at the same position is normal.

---

## Architecture

```
  Synthetic generator          PostgreSQL star schema         Feature engineering
  682,650 booking rows   ──►   6 dims / 2 facts        ──►    36 predictive
  embedded rate tests          role-playing dim_date          21 causal
                                                                    │
                    ┌───────────────────────────────────────────────┤
                    ▼                                               ▼
        DEMAND MODEL (5a)                              ELASTICITY MODEL (5b)
        XGBoost, Poisson objective                     Poisson GLM, clustered SE
        all 36 features                                21 features, no post-treatment
        "how many rooms will sell?"                    "what if we change price?"
                    │                                               │
                    └───────────────────┬───────────────────────────┘
                                        ▼
                            PRICE OPTIMIZER (Phase 6)
                            forward simulation to arrival
                            + capacity, margin, ±15% caps
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
              SHAP explainer      Streamlit app        FastAPI service
```

### The design decision at the centre

Two models, because they answer different questions and the same feature is valid for
one and poison for the other.

| | Demand model | Elasticity model |
|---|---|---|
| Question | how many rooms will sell? | what happens **if** we change price? |
| Pace features | **used** — 70% of importance | **excluded** — post-treatment |
| Data | everything (682k rows) | rate tests, rooms on sale (12%) |
| Judged by | forecast error | recovery of true elasticity |

`occupancy_on_books` is the strongest predictor available *and* an invalid causal
control: rooms already sold are caused by the rates already charged, so conditioning
on it blocks the very effect being measured. Measured cost of getting this wrong:

| Specification | Elasticity error |
|---|---|
| Causal set (correct) | 0.13 |
| \+ `occupancy_on_books` | 0.16 |
| \+ `pace_gap` | **0.41** |

`pace_gap` does the most damage precisely because it is the *best* feature — it strips
out the exogenous lead-time pattern, leaving nearly pure price-induced variation.
**The better a feature captures what price did, the more damage it does to measuring
what price does.**

The reverse cost is small: dropping pace features costs the forecaster ~4% of
deviance. Same columns, opposite verdicts.

---

## The optimizer

```
bookings(rate) = demand_forecast × (rate / current_rate) ^ elasticity
                 └── Phase 5a ──┘   └────── Phase 5b ──────┘
```

The demand model supplies the **level**; the causal elasticity supplies the
**response**. Neither can do both jobs — the forecaster's own implied elasticity is
**−0.50** against a true **−1.50**, understated threefold, because it learned that
high rates accompany high demand.

Each night is **simulated forward day by day to arrival**, not predicted once: today's
pickup changes tomorrow's pace features, and once inventory runs out further pickup is
impossible.

That last constraint produces behaviour nobody programmed:

| Occupancy on books | Nights | Mean rate change |
|---|---|---|
| <50% | 12 | −12.5% |
| 50–70% | 157 | −10.3% |
| 70–85% | 407 | −11.5% |
| **>85%** | **309** | **−3.3%** |

Every elasticity is below −1, so a naive `revenue = price × quantity` calculation says
*always discount*. The simulation says *discount where there is room to fill* — on a
night already at 90%, a price cut sells a handful of extra rooms and marks down all
the rest.

### Two findings worth escalating

**56% of recommendations sit on a policy cap, not an economic optimum.** 46% hit the
−15% floor, 10% the +15% ceiling. The optimizer wants to move further than policy
allows — which says the ±15% band, not the model, is the binding constraint on
revenue.

**Revenue maximization over-discounts.**

| Strategy | Mean rate change | Revenue | Profit |
|---|---|---|---|
| Revenue max | −8.4% | $11.51M | $9.28M |
| **Profit max** | **−4.3%** | $11.47M | **$9.29M** |
| Conservative (CI upper bound) | +0.2% | $11.37M | $9.24M |
| Do nothing | 0% | $10.48M | $8.62M |

Profit maximization halves the discounting and still returns slightly more profit.
The gap is small here (~0.1%) because variable cost is only 22% — but revenue
maximization is the default objective in most pricing tutorials, and it is the wrong
one whenever marginal costs exist.

**The honest summary: most of the gain comes from differentiating rates across nights,
not from discounting on average.**

---

## Explainability

Two layers, because "explain the recommendation" is two questions:

- **Why this forecast?** → SHAP. A 48-feature gradient booster is genuinely opaque.
- **Why this rate?** → direct decomposition. The optimizer is code we wrote. Running
  SHAP over it would be theatre.

Sample output:

```
HTL003 / RT_STD / 2026-09-10
  Recommendation: reduce rate $562 -> $478 (-15.0%)

  Why:
    - Measured elasticity -2.07, so a 10% rate change moves bookings ~21%.
    - Night is 69% sold, with 31% of inventory still to fill.
    - Capped by policy floor (-15%): the unconstrained optimum is further out.
    - Expected: 170 more rooms, $17,843 additional revenue (+28.1%).
```

Two things this layer surfaced:

**SHAP values here are multiplicative, not additive in rooms.** The model predicts in
log space, so `exp(shap)` is the demand multiplier — +0.7 means demand roughly
doubles. Reporting raw values as room counts is a quiet and common error.

**The revenue optimum is flat.** On 68% of nights, three or more candidate rates land
within 1% of the best. So recommendations are rounded to the dollar and reported as a
**range** — $470–$490, not $478.31. Spurious precision costs trust.

---

## Interfaces

### Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

Four pages: executive overview, recommendation explorer with per-night explanations,
interactive rate simulator, and model performance. Reads only the CSVs written by
earlier phases — nothing is retrained on load, so it starts in about a second and
cannot disagree with the analysis that produced it.

### FastAPI service

```bash
uvicorn api.main:app --reload      # docs at http://localhost:8000/docs
pytest api/test_api.py -v          # 23 tests
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | confirms models loaded, not just that the process runs |
| `GET /segments` | 15 hotel/room-type pairs with elasticities and CIs |
| `GET /recommendations` | precomputed, filterable — what a nightly rate push calls |
| `POST /optimize` | live pricing for arbitrary state, returns the full revenue curve |

Models load **once** at startup via the `lifespan` handler. Tests assert economics,
not status codes — a 200 carrying a nonsensical price is worse than a 500, because
nothing alerts on it.

---

## Quick start

**Prerequisites:** Python 3.11+, Docker.

```bash
git clone <repo> && cd hotel-dynamic-pricing
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env          # then set POSTGRES_PASSWORD
docker compose up -d          # PostgreSQL on port 5433
```

Run the pipeline in order — each step writes what the next one reads:

```bash
python src/generate_dimensions.py   # calendar, hotels, events
python src/generate_facts.py        # 682,650 booking rows       (~90s)
python src/load_to_postgres.py      # star schema + ETL          (~55s)
python src/features.py              # 36 predictive, 21 causal   (~60s)
python src/elasticity.py            # causal estimation          (~60s)
python src/demand_model.py          # 4-model comparison         (~7 min)
python src/optimizer.py             # 885 rate recommendations   (~3 min)
python src/explain.py               # SHAP + decompositions      (~90s)

streamlit run dashboard/app.py
```

`demand_model.py --sample 0.2` trains on a fraction of rows if Random Forest is slow
on your machine.

The whole pipeline is deterministic from `seed=42` — rebuilt from scratch, it
reproduces every figure in this README exactly.

---

## Repository

```
├── src/
│   ├── generate_dimensions.py   hotels, room types, calendar, city events
│   ├── generate_facts.py        booking curves with embedded rate tests
│   ├── load_to_postgres.py      ETL: SQLAlchemy for dims, COPY for facts
│   ├── features.py              feature engineering + leakage assertions
│   ├── elasticity.py            causal estimation + robustness + placebo
│   ├── demand_model.py          Poisson / RF / XGBoost / CatBoost
│   ├── optimizer.py             forward simulation + constraints
│   └── explain.py               SHAP layer + decision decomposition
├── notebooks/                   03 EDA · 04 features · 05a/05b models
│                                06 optimizer · 07 explainability
├── database/schema.sql          star schema DDL
├── dashboard/app.py             Streamlit
├── api/                         FastAPI service + tests
└── reports/                     charts and exported findings
```

Notebooks are the narrative — every phase explains *why* before the code, and
interprets results immediately after. Scripts are the reproducible version: a reviewer
cloning this repo should not have to execute cells in the right order.

---

## Bugs worth reading about

Kept because the debugging is the interesting part.

**Foreign key caught a modelling error.** The ETL failed with
`booking_date_id=20240731 not present in dim_date`. `dim_date` started at the first
stay date, but bookings for that night were taken 60 days earlier. The underlying
mistake was conceptual: a calendar day is neither a stay date nor a booking date until
a fact joins to it. That is the whole point of a **role-playing dimension**, and I had
violated it in the generator while implementing it correctly in SQL. Note the
*constraint* found this, not a test — loading into a schema without constraints would
have "worked", and the EDA would have quietly analysed a broken join.

**Occupancy came out at 1.4%.** Classic **fan-out**: `rooms_available` repeats across
~60 booking-date rows per night, so summing it multiplied capacity by 60. The fix is
to aggregate to the fact's true grain first. The EDA notebook demonstrates the bug
before fixing it — seeing 1.4% next to 82.1% is how the lesson sticks.

**A leak demo that didn't work.** I built the standard missing-`.shift(1)` example on
hotel-level momentum. Correct: 0.136. Leaky: 0.139. Indistinguishable — the leak was
real but *diluted*, because hotel-level aggregation buries one cell's contribution.
Rebuilt at stay-date level it inflates 0.688 → 0.855. The real lesson turned out to be
about **detection**: a leak in an aggregated feature will not show up in a correlation
check.

**Two business rules that contradict each other.** At a current rate of $60 the margin
floor ($73) sits above the +15% ceiling ($69) — no rate satisfies both. The API first
returned a 422 the caller could not act on. The right answer needs an explicit
precedence rule: the margin floor wins, because whiplash is a worse experience but
selling below cost is a worse outcome. The response now breaches the cap and sets
`requires_signoff: true`.

**A confidence note that contradicted itself.** The API reported "3 of 19 rates within
1% of best" alongside a range whose low and high were identical. The optimizer clips
candidates to the policy band, so several multipliers collapse onto the same rate.
Fixed by counting distinct rates; now a regression test.

---

## Limitations

Being explicit about these is worth more than another decimal place of uplift.

- **The uplift is simulated.** Both baseline and optimized figures come from the same
  demand model. A live A/B test on held-out stay dates is the only real validation.
- **Competitor rates are held fixed.** If we cut 10%, competitors may respond and the
  volume gain would not materialise. Modelling that needs a game-theoretic setup
  beyond this project's scope — but pretending it does not exist would be worse.
- **Room types are optimized independently.** A cheaper standard room cannibalises
  deluxe demand. Cross-price elasticity is real and unestimated here, so the portfolio
  uplift is likely overstated.
- **Constant elasticity across the rate range.** A single elasticity is extrapolated
  across ±15%; real demand curves bend, and the rate tests only spanned 0.70–1.40 of
  the reference rate.
- **No length-of-stay or cancellation modelling.** Real revenue management prices
  multi-night stays and forecasts cancellations. This prices single nights and assumes
  every booking materialises.
- **Forecast error peaks near arrival**, exactly when pricing decisions are most
  urgent.
- **Synthetic data.** Elasticities are recoverable partly because the generator put
  them there. The methodology transfers; the specific numbers do not.

### If this were going to production

1. Run the A/B test — hold out stay dates, apply recommended rates to half
2. Estimate cross-price elasticity between room types
3. Model competitor response
4. Monitor for elasticity drift; re-estimate on a rolling window
5. Keep randomized rate tests running — they are the only thing that keeps the model
   identifiable

---

## Stack

Python 3.12 · pandas · NumPy · PostgreSQL 16 (Docker) · SQLAlchemy · psycopg2 ·
scikit-learn · XGBoost · CatBoost · statsmodels · SHAP · Streamlit · Plotly · FastAPI ·
pytest

---

## What this project demonstrates

Anyone can fit a gradient booster. What is harder, and what this is built to show:

- Knowing when prediction is the wrong tool, and being able to prove it with numbers
- Designing a data model whose grain matches the business process
- Recognising post-treatment bias and quantifying its cost
- Building constraints that make an optimizer's output usable rather than merely optimal
- Explaining a recommendation in language a revenue manager will act on
- Stating limitations clearly enough that someone could act on the work responsibly
