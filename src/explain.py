"""
Phase 7 - Explainability.

A pricing system that cannot explain itself gets overridden. But "explain the
recommendation" is really two questions, and only one of them is a SHAP
question:

  LAYER 1 - why this DEMAND FORECAST?
      The forecast comes from a 48-feature gradient boosting model. It is
      opaque, and SHAP is the right tool.

  LAYER 2 - why this RATE?
      The rate comes from forecast x elasticity x capacity headroom, filtered
      through policy constraints. That pipeline is NOT opaque - we wrote it.
      Running SHAP over it would be theatre. A direct decomposition is both
      more honest and more useful.

Mixing the two is the common mistake. SHAP on the demand model tells you why
we expect 12 rooms to sell; it says nothing about why we chose $214.

A note on units. The demand model uses objective="count:poisson", so it
predicts in LOG space. SHAP values are additive in log space, which means
they are MULTIPLICATIVE in room space:

    predicted_rooms = exp(base + sum of shap values)

A SHAP value of +0.7 therefore means "roughly doubles expected demand"
(exp(0.7) = 2.0), not "+0.7 rooms". Reporting them as rooms would be wrong.

Output:
    reports/shap_feature_importance.csv
    reports/recommendation_explanations.csv

Run from the project root:
    python src/explain.py
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "models"
REPORTS = PROJECT_ROOT / "reports"

import features as F
import optimizer as O

# Plain-English names, because "pace_gap_dow" means nothing to a revenue manager.
FEATURE_LABELS = {
    "typical_pickup": "Typical demand for this night/lead time",
    "typical_occ_on_books": "Typical occupancy at this lead time",
    "typical_occ_dow": "Typical occupancy for this day of week",
    "rooms_on_books": "Rooms already sold",
    "occupancy_on_books": "Occupancy on the books",
    "rooms_remaining": "Rooms still available",
    "pct_remaining": "% of inventory unsold",
    "pace_gap": "Pace vs normal",
    "pace_gap_dow": "Pace vs normal for this weekday",
    "pace_ratio": "Pace ratio vs normal",
    "pickup_lag_1d": "Bookings yesterday",
    "pickup_lag_3d": "Bookings last 3 days",
    "pickup_lag_7d": "Bookings last 7 days",
    "momentum_7d": "Hotel booking activity, last 7 days",
    "momentum_28d": "Hotel booking activity, last 28 days",
    "momentum_ratio": "Booking activity trend",
    "lead_time": "Days before arrival",
    "log_rate": "Our rate",
    "rate_vs_base": "Rate vs reference rate",
    "rate_vs_comp": "Rate vs competitors",
    "log_comp_rate": "Competitor rate level",
    "stay_is_weekend": "Weekend night",
    "stay_is_holiday": "Public holiday",
    "stay_days_until_holiday": "Days until next holiday",
    "has_event": "City event on this date",
    "event_lift": "Event demand lift",
    "rooms_available": "Room type capacity",
    "base_rate": "Reference rate",
    "leisure_share": "Hotel leisure mix",
    "star_rating": "Star rating",
}


def label(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    for prefix, kind in [("hotel_id_", "Hotel"), ("room_type_id_", "Room type"),
                         ("city_", "City"), ("segment_", "Segment")]:
        if feature.startswith(prefix):
            return f"{kind}: {feature[len(prefix):]}"
    return feature.replace("_", " ").capitalize()


# ==========================================================================
# Layer 1 - SHAP on the demand forecast
# ==========================================================================
def build_explainer(bundle):
    import shap
    return shap.TreeExplainer(bundle["model"])


def shap_importance(explainer, X: pd.DataFrame,
                    feature_cols: list) -> pd.DataFrame:
    """Global importance: mean |SHAP| across a sample.

    Preferred to XGBoost's built-in importance, which counts how often a
    feature is split on. A feature can be split on constantly and barely
    move predictions. SHAP measures actual contribution to the output.
    """
    values = explainer.shap_values(X)
    table = pd.DataFrame({
        "feature": feature_cols,
        "label": [label(f) for f in feature_cols],
        "mean_abs_shap": np.abs(values).mean(axis=0),
        "mean_shap": values.mean(axis=0),
    })
    table["category"] = table["feature"].map(_category)
    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def _category(feature: str) -> str:
    if feature in F.FEATURES_POST_TREATMENT:
        return "pace / demand history"
    if feature in F.FEATURES_PRICE:
        return "price"
    if feature in F.FEATURES_EXOGENOUS:
        return "calendar / hotel"
    return "hotel & room identity"


def explain_forecast(explainer, X_row: pd.DataFrame, feature_cols: list,
                     top_n: int = 6) -> pd.DataFrame:
    """Why this one forecast? SHAP values converted to multiplicative effect."""
    values = explainer.shap_values(X_row)[0]
    frame = pd.DataFrame({
        "feature": feature_cols,
        "label": [label(f) for f in feature_cols],
        "value": X_row.iloc[0].to_numpy(),
        "shap_log": values,
        # exp(shap) is the multiplier on expected demand.
        "demand_multiplier": np.exp(values),
    })
    frame["abs"] = frame["shap_log"].abs()
    return (frame.sort_values("abs", ascending=False)
            .head(top_n).drop(columns="abs").reset_index(drop=True))


# ==========================================================================
# Layer 2 - why this RATE?
# ==========================================================================
def explain_recommendation(row: pd.Series, scenarios: pd.DataFrame) -> dict:
    """Decompose the pricing decision into its actual drivers.

    Not SHAP. The optimizer is a pipeline we wrote, so we can report exactly
    what drove it rather than approximating it with an attribution method.
    """
    night = scenarios[(scenarios["hotel_id"] == row["hotel_id"])
                      & (scenarios["room_type_id"] == row["room_type_id"])
                      & (scenarios["stay_date"] == row["stay_date"])]

    change = row["rate_change_pct"]
    at_floor = change < -O.MAX_RATE_CHANGE * 100 + 0.5
    at_ceiling = change > O.MAX_RATE_CHANGE * 100 - 0.5

    # How much revenue would be lost by holding the current rate?
    curve = night.sort_values("rate")
    best = curve["expected_revenue"].max()
    at_current = curve.loc[curve["rate_multiplier"].sub(1.0).abs().idxmin(),
                           "expected_revenue"]

    # Curvature: how flat is the optimum? A flat peak means the exact rate
    # matters little, which is worth telling the user.
    within_1pct = (curve["expected_revenue"] >= best * 0.99).sum()

    return {
        "hotel_id": row["hotel_id"],
        "room_type_id": row["room_type_id"],
        "stay_date": row["stay_date"],
        "current_rate": round(row["current_rate"], 2),
        "recommended_rate": round(row["rate"], 2),
        "rate_change_pct": round(change, 1),
        "elasticity": round(row["elasticity"], 2),
        "occupancy_on_books": round(row["baseline_occupancy"], 3),
        "headroom_pct": round(1 - row["baseline_occupancy"], 3),
        "expected_pickup": round(row["expected_pickup"], 1),
        "revenue_uplift": round(row["revenue_uplift"], 0),
        "revenue_uplift_pct": round(row["revenue_uplift_pct"], 1),
        "constraint_binding": ("policy floor (-15%)" if at_floor
                               else "policy ceiling (+15%)" if at_ceiling
                               else "none - interior optimum"),
        "cost_of_inaction": round(best - at_current, 0),
        "rates_within_1pct_of_best": int(within_1pct),
    }


def narrate(explanation: dict, forecast_drivers: pd.DataFrame) -> str:
    """Plain-English summary. This is what a revenue manager reads."""
    direction = "increase" if explanation["rate_change_pct"] > 0 else "reduce"
    lines = [
        f"{explanation['hotel_id']} / {explanation['room_type_id']} / "
        f"{pd.Timestamp(explanation['stay_date']).date()}",
        f"  Recommendation: {direction} rate "
        f"${explanation['current_rate']:.0f} -> ${explanation['recommended_rate']:.0f} "
        f"({explanation['rate_change_pct']:+.1f}%)",
        "",
        "  Why:",
        f"    - This segment's measured price elasticity is "
        f"{explanation['elasticity']:.2f}, so a 10% rate change moves bookings by "
        f"about {abs(explanation['elasticity']) * 10:.0f}%.",
        f"    - The night is {explanation['occupancy_on_books']:.0%} sold with "
        f"{explanation['headroom_pct']:.0%} of inventory still to fill.",
    ]
    if explanation["constraint_binding"] != "none - interior optimum":
        lines.append(f"    - Capped by {explanation['constraint_binding']}: the "
                     f"unconstrained optimum is further out.")
    lines += [
        f"    - Expected outcome: {explanation['expected_pickup']:.0f} more rooms, "
        f"${explanation['revenue_uplift']:,.0f} additional revenue "
        f"({explanation['revenue_uplift_pct']:+.1f}%).",
        "",
        "  Top drivers of the demand forecast:",
    ]
    for r in forecast_drivers.head(4).itertuples():
        arrow = "increases" if r.shap_log > 0 else "reduces"
        lines.append(f"    - {r.label}: {arrow} expected demand "
                     f"{r.demand_multiplier:.2f}x")
    return "\n".join(lines)


def live_feature_row(splits, row) -> pd.DataFrame:
    """Find the model-input row for one open night.

    build_state() resets its index, so state row numbers do NOT line up with
    the feature matrix. Match on the business keys and take the most recent
    booking date instead - the same snapshot the optimizer priced from.
    """
    X_live, _, frame_live = splits["live"]
    mask = ((frame_live["hotel_id"] == row["hotel_id"])
            & (frame_live["room_type_id"] == row["room_type_id"])
            & (frame_live["stay_date"] == row["stay_date"]))
    idx = frame_live[mask].sort_values("booking_date").index[-1:]
    return X_live.loc[idx]


# ==========================================================================
def main() -> None:
    REPORTS.mkdir(exist_ok=True)

    df, bundle, elasticities, baseline = O.load_artifacts()
    splits, feature_cols = __import__("demand_model").prepare(df)

    print("Layer 1 - SHAP on the demand model")
    print("-" * 68)
    explainer = build_explainer(bundle)
    X_sample = splits["test"][0].sample(3000, random_state=42)
    importance = shap_importance(explainer, X_sample, feature_cols)
    print(importance.head(10)[["label", "mean_abs_shap", "category"]]
          .round(4).to_string(index=False))

    by_cat = importance.groupby("category")["mean_abs_shap"].sum()
    print("\nContribution by category:")
    print((by_cat / by_cat.sum()).sort_values(ascending=False)
          .round(3).to_string())

    print("\n\nLayer 2 - why these rates")
    print("-" * 68)
    recommendations = pd.read_csv(REPORTS / "price_recommendations.csv",
                                  parse_dates=["stay_date"])
    scenarios = pd.read_csv(REPORTS / "price_scenarios.csv",
                            parse_dates=["stay_date"])

    explanations = pd.DataFrame([
        explain_recommendation(row, scenarios)
        for _, row in recommendations.iterrows()
    ])
    print(explanations["constraint_binding"].value_counts().to_string())
    print(f"\nMean cost of inaction per night: "
          f"${explanations['cost_of_inaction'].mean():,.0f}")
    print(f"Nights where many rates are within 1% of best: "
          f"{(explanations['rates_within_1pct_of_best'] >= 3).mean():.1%}")

    print("\n\nExample explanation")
    print("-" * 68)
    top = recommendations.sort_values("revenue_uplift", ascending=False).iloc[0]
    X_row = live_feature_row(splits, top)
    drivers = explain_forecast(explainer, X_row, feature_cols)
    print(narrate(explain_recommendation(top, scenarios), drivers))

    importance.to_csv(REPORTS / "shap_feature_importance.csv", index=False)
    explanations.to_csv(REPORTS / "recommendation_explanations.csv", index=False)
    print(f"\n\nWrote reports/shap_feature_importance.csv")
    print(f"Wrote reports/recommendation_explanations.csv")


if __name__ == "__main__":
    main()
