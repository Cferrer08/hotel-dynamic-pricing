"""
Phase 8b - Export a Power BI-ready star schema.

Power BI is not pandas. Two rules drive the shape of these files:

  1. STAR SCHEMA, NOT A FLAT TABLE. Power BI's engine (VertiPaq) is built
     for one-to-many relationships from dimensions to facts. A single wide
     table works but kills filter propagation and makes slicers behave
     unpredictably.

  2. MEASURES, NOT PRECOMPUTED COLUMNS. Never export a percentage or an
     average - export the numerator and denominator and let DAX divide.
     Averaging an average is the single most common Power BI error, and
     exporting `occupancy_pct` invites it. Export rooms_sold and capacity
     instead; SUM(rooms)/SUM(capacity) is then correct at every level.

Output (data/powerbi/):
    dim_hotel.csv        one row per hotel
    dim_room_type.csv    one row per room type
    dim_date.csv         one row per stay date, with a proper date column
    fact_recommendation.csv   one row per open night  (885)
    fact_scenario.csv         one row per night x candidate rate (16,815)
    fact_history.csv          daily historical performance, for trends
    model_metrics.csv         flat table for the model performance page

Run from the project root:
    python src/export_powerbi.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from db import get_engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
MODELS = PROJECT_ROOT / "models"
OUT = PROJECT_ROOT / "data" / "powerbi"

TODAY = pd.Timestamp("2026-07-30")


# ==========================================================================
# Dimensions
# ==========================================================================
def export_dimensions(engine) -> None:
    hotels = pd.read_sql(text("""
        SELECT hotel_id, hotel_name, city, segment, star_rating, total_rooms
        FROM pricing.dim_hotel
    """), engine)
    hotels.to_csv(OUT / "dim_hotel.csv", index=False)

    rooms = pd.read_sql(text("""
        SELECT room_type_id, room_type_name, rate_index
        FROM pricing.dim_room_type
    """), engine)
    # Explicit sort order, because Power BI sorts alphabetically by default
    # and "Deluxe, Standard, Suite" is not the order anyone wants.
    rooms["sort_order"] = rooms["room_type_id"].map(
        {"RT_STD": 1, "RT_DLX": 2, "RT_STE": 3})
    rooms.to_csv(OUT / "dim_room_type.csv", index=False)

    dates = pd.read_sql(text("""
        SELECT calendar_date AS date, year, quarter, month, week_of_year,
               day_of_week, day_name, is_weekend, is_holiday, holiday_name
        FROM pricing.dim_date
        ORDER BY calendar_date
    """), engine, parse_dates=["date"])

    # Power BI needs these for correct sorting and readable axis labels.
    dates["month_name"] = dates["date"].dt.strftime("%b")
    dates["year_month"] = dates["date"].dt.strftime("%Y-%m")
    dates["day_sort"] = dates["day_of_week"]
    dates["is_future"] = dates["date"] > TODAY
    dates["period"] = np.where(dates["date"] > TODAY, "Open", "Closed")
    dates.to_csv(OUT / "dim_date.csv", index=False)

    print(f"  dim_hotel.csv        {len(hotels):>7,} rows")
    print(f"  dim_room_type.csv    {len(rooms):>7,} rows")
    print(f"  dim_date.csv         {len(dates):>7,} rows")


# ==========================================================================
# Facts
# ==========================================================================
def export_recommendations() -> pd.DataFrame:
    """One row per open night. The core fact table for the app."""
    rec = pd.read_csv(REPORTS / "price_recommendations.csv",
                      parse_dates=["stay_date"])
    exp = pd.read_csv(REPORTS / "recommendation_explanations.csv",
                      parse_dates=["stay_date"])

    keys = ["hotel_id", "room_type_id", "stay_date"]
    fact = rec.merge(
        exp[keys + ["constraint_binding", "cost_of_inaction",
                    "rates_within_1pct_of_best", "headroom_pct"]],
        on=keys, how="left")

    # Capacity, so occupancy can be a MEASURE rather than a stored average.
    fact["capacity"] = np.where(
        fact["final_occupancy"] > 0,
        fact["final_on_books"] / fact["final_occupancy"].replace(0, np.nan),
        np.nan)
    fact["capacity"] = fact["capacity"].round().fillna(0).astype(int)
    fact["rooms_on_books_now"] = (fact["baseline_occupancy"]
                                  * fact["capacity"]).round().astype(int)

    fact["days_to_arrival"] = (fact["stay_date"] - TODAY).dt.days
    fact["action"] = np.select(
        [fact["rate_change_pct"] > 0.5, fact["rate_change_pct"] < -0.5],
        ["Increase", "Decrease"], default="Hold")
    # Banding in Python, not DAX: a stored column is faster to slice on and
    # keeps the logic in one place next to everything else.
    fact["occupancy_band"] = pd.cut(
        fact["baseline_occupancy"], [-0.01, 0.5, 0.7, 0.85, 1.01],
        labels=["<50%", "50-70%", "70-85%", ">85%"]).astype(str)
    fact["is_flat_optimum"] = fact["rates_within_1pct_of_best"] >= 3

    columns = [
        "hotel_id", "room_type_id", "stay_date", "days_to_arrival",
        "current_rate", "rate", "rate_change_pct", "action",
        "elasticity", "capacity", "rooms_on_books_now",
        "baseline_occupancy", "final_occupancy", "occupancy_band",
        "baseline_pickup", "expected_pickup",
        "baseline_revenue", "expected_revenue", "revenue_uplift",
        "baseline_profit", "expected_profit", "profit_uplift",
        "constraint_binding", "cost_of_inaction",
        "rates_within_1pct_of_best", "is_flat_optimum",
    ]
    fact = fact[columns].rename(columns={"rate": "recommended_rate"})
    fact.to_csv(OUT / "fact_recommendation.csv", index=False)
    print(f"  fact_recommendation  {len(fact):>7,} rows")
    return fact


def export_scenarios() -> None:
    """One row per (night, candidate rate). Powers the what-if page."""
    scen = pd.read_csv(REPORTS / "price_scenarios.csv",
                       parse_dates=["stay_date"])
    scen["rate_change_pct"] = (scen["rate_multiplier"] - 1) * 100
    scen = scen[["hotel_id", "room_type_id", "stay_date", "rate_multiplier",
                 "rate_change_pct", "rate", "expected_pickup",
                 "expected_revenue", "expected_profit", "final_occupancy"]]
    scen.to_csv(OUT / "fact_scenario.csv", index=False)
    print(f"  fact_scenario        {len(scen):>7,} rows")


def export_history(engine) -> None:
    """Daily historical performance for trend visuals.

    Aggregated to (hotel, room_type, stay_date) - NOT the raw booking-curve
    grain. 682k rows would work but makes the .pbix slow to refresh for no
    analytical gain at this level of reporting.
    """
    history = pd.read_sql(text("""
        SELECT f.hotel_id,
               f.room_type_id,
               d.calendar_date          AS stay_date,
               SUM(f.rooms_booked)      AS rooms_sold,
               SUM(f.room_revenue)      AS revenue,
               MAX(c.rooms_available)   AS capacity
        FROM pricing.fact_booking_pickup f
        JOIN pricing.dim_date d ON d.date_id = f.stay_date_id
        JOIN pricing.dim_capacity c
          ON c.hotel_id = f.hotel_id AND c.room_type_id = f.room_type_id
        GROUP BY f.hotel_id, f.room_type_id, d.calendar_date
    """), engine, parse_dates=["stay_date"])

    # MAX(capacity), not SUM: capacity repeats across ~60 booking-date rows
    # per night. Summing it is the fan-out bug from Phase 3, and it would
    # divide every occupancy figure by 60.
    history = history[history["stay_date"] <= TODAY]
    history.to_csv(OUT / "fact_history.csv", index=False)
    print(f"  fact_history         {len(history):>7,} rows")


def export_model_metrics() -> None:
    """Flat, denormalised table for the model performance page.

    Deliberately not a star schema - these rows are heterogeneous facts
    about different models, and forcing them into dimensions would be
    modelling for its own sake.
    """
    rows = []

    comparison = pd.read_csv(REPORTS / "model_comparison.csv")
    for r in comparison.itertuples():
        for metric in ["MAE", "RMSE", "poisson_deviance", "night_MAE",
                       "fit_seconds"]:
            rows.append({"category": "Demand model", "item": r.model,
                         "metric": metric, "value": getattr(r, metric)})

    validation = pd.read_csv(REPORTS / "elasticity_validation.csv")
    for r in validation.itertuples():
        segment = f"{r.hotel_id} / {r.room_type_id}"
        rows.append({"category": "Elasticity", "item": segment,
                     "metric": "estimated", "value": r.elasticity})
        rows.append({"category": "Elasticity", "item": segment,
                     "metric": "true", "value": r.true_elasticity})
        rows.append({"category": "Elasticity", "item": segment,
                     "metric": "ci_low", "value": r.ci_low})
        rows.append({"category": "Elasticity", "item": segment,
                     "metric": "ci_high", "value": r.ci_high})

    robustness = pd.read_csv(REPORTS / "elasticity_robustness.csv")
    for r in robustness.itertuples():
        rows.append({"category": "Robustness", "item": r.specification,
                     "metric": "mean_elasticity", "value": r.mean_elasticity})
        rows.append({"category": "Robustness", "item": r.specification,
                     "metric": "MAE_vs_truth", "value": r.MAE_vs_truth})

    shap = pd.read_csv(REPORTS / "shap_feature_importance.csv").head(15)
    for r in shap.itertuples():
        rows.append({"category": "SHAP", "item": r.label,
                     "metric": r.category, "value": r.mean_abs_shap})

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "model_metrics.csv", index=False)
    print(f"  model_metrics.csv    {len(metrics):>7,} rows")

    strategies = pd.read_csv(REPORTS / "strategy_comparison.csv")
    strategies.to_csv(OUT / "strategy_comparison.csv", index=False)
    print(f"  strategy_comparison  {len(strategies):>7,} rows")


# ==========================================================================
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    engine = get_engine()

    print("Exporting Power BI star schema")
    print("-" * 60)
    export_dimensions(engine)
    fact = export_recommendations()
    export_scenarios()
    export_history(engine)
    export_model_metrics()

    print("-" * 60)
    print(f"Written to {OUT.relative_to(PROJECT_ROOT)}")
    print()
    print("Reconciliation (these must match the Streamlit dashboard)")
    print("-" * 60)
    print(f"Open nights          : {len(fact):,}")
    print(f"Baseline revenue     : ${fact['baseline_revenue'].sum():,.0f}")
    print(f"Optimized revenue    : ${fact['expected_revenue'].sum():,.0f}")
    print(f"Uplift               : ${fact['revenue_uplift'].sum():,.0f} "
          f"({fact['revenue_uplift'].sum() / fact['baseline_revenue'].sum() * 100:+.2f}%)")
    print(f"Mean rate change     : {fact['rate_change_pct'].mean():+.2f}%")


if __name__ == "__main__":
    main()
