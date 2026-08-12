"""
Phase 6 - Price optimization.

Combines the two Phase 5 models, each doing the job it is valid for:

    bookings(rate) = demand_forecast x (rate / current_rate) ^ elasticity
                     |__ Phase 5a __|   |______ Phase 5b ______|

The demand model supplies the LEVEL (how many rooms this night sells at the
current rate). The causal elasticity supplies the RESPONSE (how that level
moves when we change the rate). Neither model can do both jobs - 5a's
implied elasticity is wrong by 3x, and 5b cannot forecast volume.

Why a forward SIMULATION rather than a single prediction:
a stay date is not sold once. It fills over the remaining booking window,
and each day's pickup changes the on-books position, which changes the pace
features, which changes the next day's forecast. And capacity binds: once
the room type sells out, further pickup is impossible. Ignoring that would
recommend discounting nights that are about to sell out anyway.

Output:
    reports/price_recommendations.csv

Run from the project root:
    python src/optimizer.py
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

TODAY = F.TODAY

# ---- Business constraints -------------------------------------------------
# Variable cost per occupied room: housekeeping, amenities, utilities,
# commission. Roughly 22% of the reference rate for a mid-scale hotel.
VARIABLE_COST_PCT = 0.22
MIN_MARGIN_PCT = 0.35          # never sell below cost x 1.35
MAX_RATE_CHANGE = 0.15         # +/- 15% vs the current rate, to avoid whiplash

# Candidate rates as multipliers of the current rate.
RATE_GRID = np.round(np.arange(0.80, 1.2501, 0.025), 4)

# Stay-date-level features that do not vary across booking dates.
STATIC_FEATURES = [
    "stay_dow", "stay_month", "stay_week", "doy_sin", "doy_cos",
    "dow_sin", "dow_cos", "stay_is_weekend", "stay_is_holiday",
    "stay_days_until_holiday", "has_event", "event_lift",
    "leisure_share", "star_rating", "rooms_available", "base_rate",
    "momentum_7d", "momentum_28d", "momentum_ratio",
]


# ==========================================================================
# Current state of the open nights
# ==========================================================================
def build_state(df: pd.DataFrame) -> pd.DataFrame:
    """Latest snapshot per open (hotel, room_type, stay_date).

    'Latest' means the most recent booking_date we have, i.e. TODAY.
    """
    live = df[df["split"] == "live"].copy()
    latest = (live.sort_values("booking_date")
              .groupby(["hotel_id", "room_type_id", "stay_date"])
              .tail(1).copy())

    # Revenue already banked for these nights. Sunk - the optimizer can
    # only influence what happens from here on.
    booked = (live.groupby(["hotel_id", "room_type_id", "stay_date"])
              .agg(revenue_to_date=("room_revenue", "sum"))
              .reset_index())

    state = latest.merge(booked, on=["hotel_id", "room_type_id", "stay_date"])
    state["days_to_arrival"] = (state["stay_date"] - TODAY).dt.days
    state["current_rate"] = state["rate"]
    state["variable_cost"] = state["base_rate"] * VARIABLE_COST_PCT
    state["rate_floor"] = np.maximum(
        state["variable_cost"] * (1 + MIN_MARGIN_PCT),
        state["current_rate"] * (1 - MAX_RATE_CHANGE))
    state["rate_ceiling"] = state["current_rate"] * (1 + MAX_RATE_CHANGE)
    return state.reset_index(drop=True)


# ==========================================================================
# Forward simulation
# ==========================================================================
class Simulator:
    """Rolls each open night forward day by day to arrival.

    At each step: build the feature row, predict pickup, cap it at
    remaining inventory, then update on-books and the lag features so the
    next step sees the consequences.
    """

    def __init__(self, model, feature_cols, baseline, elasticities,
                 categorical_template: pd.DataFrame,
                 elasticity_column: str = "elasticity"):
        self.model = model
        self.feature_cols = feature_cols
        self.pace = baseline["pace"].set_index(
            ["hotel_id", "room_type_id", "lead_time"])["typical_occ_on_books"]
        self.pickup = baseline["pickup"].set_index(
            ["hotel_id", "room_type_id", "lead_time"])["typical_pickup"]
        self.pace_dow = baseline["pace_dow"].set_index(
            ["hotel_id", "room_type_id", "stay_dow", "lead_time"])["typical_occ_dow"]
        self.elasticity = elasticities.set_index(
            ["hotel_id", "room_type_id"])
        self.categoricals = categorical_template
        # "elasticity" uses the point estimate. "ci_high" is the CONSERVATIVE
        # choice: the least-elastic end of the confidence interval. If demand
        # is less price-sensitive than we think, discounts do not pay for
        # themselves - so pricing against ci_high protects the downside on
        # segments we know little about.
        self.elasticity_column = elasticity_column

    def _lookup(self, series, keys, default=0.0):
        return series.reindex(keys).fillna(default).to_numpy()

    def run(self, state: pd.DataFrame, rate_multiplier: float) -> pd.DataFrame:
        """Simulate every open night at rate = current_rate * multiplier."""
        n = len(state)
        hotel = state["hotel_id"].to_numpy()
        room = state["room_type_id"].to_numpy()
        dow = state["stay_dow"].to_numpy()
        capacity = state["rooms_available"].to_numpy(dtype=float)

        on_books = state["rooms_on_books"].to_numpy(dtype=float)
        lag1 = state["pickup_lag_1d"].to_numpy(dtype=float)
        lag3 = state["pickup_lag_3d"].to_numpy(dtype=float)
        lag7 = state["pickup_lag_7d"].to_numpy(dtype=float)

        rate = state["current_rate"].to_numpy(dtype=float) * rate_multiplier
        rate = np.clip(rate, state["rate_floor"], state["rate_ceiling"])

        # The causal adjustment. The demand model's own price sensitivity is
        # wrong (Phase 5a showed it understates by 3x), so we neutralise it:
        # forecast at the CURRENT rate, then apply the causal elasticity to
        # the ratio ourselves.
        eps = self._lookup(self.elasticity[self.elasticity_column],
                           pd.MultiIndex.from_arrays([hotel, room]), -1.0)
        price_effect = (rate / state["current_rate"].to_numpy()) ** eps

        max_days = int(state["days_to_arrival"].max())
        total_pickup = np.zeros(n)

        for step in range(max_days):
            lead = state["days_to_arrival"].to_numpy() - step
            active = lead >= 0
            if not active.any():
                break

            X = self._build_features(state, lead, on_books, lag1, lag3, lag7,
                                     hotel, room, dow, capacity)
            # Forecast at the CURRENT rate, then apply the causal response.
            predicted = np.clip(self.model.predict(X), 0, None) * price_effect
            predicted = np.where(active, predicted, 0.0)

            # Capacity is the hard constraint that defines hotel pricing.
            remaining = np.maximum(capacity - on_books, 0.0)
            actual = np.minimum(predicted, remaining)

            on_books = on_books + actual
            total_pickup += actual
            lag7 = (lag7 * 6 + actual) / 7
            lag3 = (lag3 * 2 + actual) / 3
            lag1 = actual

        revenue = total_pickup * rate
        profit = total_pickup * (rate - state["variable_cost"].to_numpy())
        return pd.DataFrame({
            "hotel_id": hotel,
            "room_type_id": room,
            "stay_date": state["stay_date"].to_numpy(),
            "rate_multiplier": rate_multiplier,
            "rate": rate,
            "expected_pickup": total_pickup,
            "expected_revenue": revenue,
            "expected_profit": profit,
            "final_on_books": on_books,
            "final_occupancy": on_books / capacity,
            "elasticity": eps,
        })

    def _build_features(self, state, lead, on_books, lag1, lag3, lag7,
                        hotel, room, dow, capacity):
        lead_clipped = np.clip(lead, 0, 59)
        keys = pd.MultiIndex.from_arrays([hotel, room, lead_clipped])
        keys_dow = pd.MultiIndex.from_arrays([hotel, room, dow, lead_clipped])

        typical_occ = self._lookup(self.pace, keys)
        typical_pick = self._lookup(self.pickup, keys)
        typical_dow = self._lookup(self.pace_dow, keys_dow)

        occupancy = on_books / capacity
        frame = pd.DataFrame({
            "lead_time": lead_clipped,
            # Rate features are held at the CURRENT rate on purpose - the
            # causal adjustment is applied outside the model.
            "log_rate": np.log(state["current_rate"].to_numpy()),
            "rate_vs_base": state["rate_vs_base"].to_numpy(),
            "rate_vs_comp": state["rate_vs_comp"].to_numpy(),
            "log_comp_rate": state["log_comp_rate"].to_numpy(),
            "rooms_on_books": on_books,
            "occupancy_on_books": occupancy,
            "typical_occ_on_books": typical_occ,
            "typical_pickup": typical_pick,
            "pace_gap": occupancy - typical_occ,
            "pace_gap_dow": occupancy - typical_dow,
            "pace_ratio": np.clip(occupancy / np.clip(typical_occ, 0.01, None),
                                  None, 5.0),
            "rooms_remaining": np.maximum(capacity - on_books, 0.0),
            "pct_remaining": np.maximum(capacity - on_books, 0.0) / capacity,
            "pickup_lag_1d": lag1,
            "pickup_lag_3d": lag3,
            "pickup_lag_7d": lag7,
        })
        for col in STATIC_FEATURES:
            frame[col] = state[col].to_numpy()

        X = pd.concat([frame[F.FEATURES_PREDICTIVE].reset_index(drop=True),
                       self.categoricals.reset_index(drop=True)], axis=1)
        return X[self.feature_cols].astype(float)


# ==========================================================================
# Optimization
# ==========================================================================
def optimize(simulator: Simulator, state: pd.DataFrame,
             grid=RATE_GRID, objective: str = "expected_revenue") -> tuple:
    """Grid search. The objective is not smooth once capacity binds, so a
    grid is more robust here than a gradient method - and 19 evaluations is
    cheap."""
    scenarios = [simulator.run(state, m) for m in grid]
    all_scenarios = pd.concat(scenarios, ignore_index=True)

    keys = ["hotel_id", "room_type_id", "stay_date"]
    best_idx = all_scenarios.groupby(keys)[objective].idxmax()
    best = all_scenarios.loc[best_idx].reset_index(drop=True)

    baseline = (all_scenarios[all_scenarios["rate_multiplier"] == 1.0]
                .rename(columns={"rate": "current_rate",
                                 "expected_pickup": "baseline_pickup",
                                 "expected_revenue": "baseline_revenue",
                                 "expected_profit": "baseline_profit",
                                 "final_occupancy": "baseline_occupancy"})
                [keys + ["current_rate", "baseline_pickup", "baseline_revenue",
                         "baseline_profit", "baseline_occupancy"]])

    recommendations = best.merge(baseline, on=keys)
    recommendations["rate_change_pct"] = (
        recommendations["rate"] / recommendations["current_rate"] - 1) * 100
    recommendations["revenue_uplift"] = (
        recommendations["expected_revenue"] - recommendations["baseline_revenue"])
    recommendations["revenue_uplift_pct"] = np.where(
        recommendations["baseline_revenue"] > 0,
        recommendations["revenue_uplift"] / recommendations["baseline_revenue"] * 100,
        0.0)
    recommendations["profit_uplift"] = (
        recommendations["expected_profit"] - recommendations["baseline_profit"])
    return recommendations, all_scenarios


# ==========================================================================
def load_artifacts():
    df = pd.read_parquet(PROCESSED / "features.parquet")
    with open(MODELS / "demand_model.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    elasticities = pd.read_csv(MODELS / "elasticities.csv")
    baseline = F.fit_pace_baseline(df[df["split"] == "train"])
    return df, bundle, elasticities, baseline


def make_simulator(df, bundle, elasticities, baseline, state,
                   elasticity_column: str = "elasticity"):
    """Rebuild the one-hot columns exactly as training saw them."""
    template = df.head(1)
    dummies_all = pd.get_dummies(template[F.CATEGORICALS], drop_first=True)
    state_dummies = pd.get_dummies(state[F.CATEGORICALS], drop_first=True)
    for col in dummies_all.columns:
        if col not in state_dummies:
            state_dummies[col] = 0
    # Match the training column order exactly; a silent mismatch here
    # produces predictions that look plausible and are wrong.
    ordered = [c for c in bundle["features"] if c not in F.FEATURES_PREDICTIVE]
    for col in ordered:
        if col not in state_dummies:
            state_dummies[col] = 0
    return Simulator(bundle["model"], bundle["features"], baseline,
                     elasticities, state_dummies[ordered].astype(int),
                     elasticity_column=elasticity_column)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    df, bundle, elasticities, baseline = load_artifacts()

    state = build_state(df)
    print(f"Open nights to price: {len(state):,} "
          f"({state['stay_date'].nunique()} stay dates x "
          f"{state['hotel_id'].nunique()} hotels x "
          f"{state['room_type_id'].nunique()} room types)")
    print(f"Current mean occupancy on books: "
          f"{(state['rooms_on_books'] / state['rooms_available']).mean():.1%}")

    simulator = make_simulator(df, bundle, elasticities, baseline, state)
    recommendations, scenarios = optimize(simulator, state)

    print("\nRecommendations")
    print("-" * 70)
    print(f"Mean rate change      : {recommendations['rate_change_pct'].mean():+.2f}%")
    print(f"Rate increases        : {(recommendations['rate_change_pct'] > 0.5).mean():.1%}")
    print(f"Rate decreases        : {(recommendations['rate_change_pct'] < -0.5).mean():.1%}")
    print(f"Unchanged             : {(recommendations['rate_change_pct'].abs() <= 0.5).mean():.1%}")
    print()
    total_base = recommendations["baseline_revenue"].sum()
    total_opt = recommendations["expected_revenue"].sum()
    print(f"Baseline revenue      : ${total_base:>12,.0f}")
    print(f"Optimized revenue     : ${total_opt:>12,.0f}")
    print(f"Uplift                : ${total_opt - total_base:>12,.0f} "
          f"({(total_opt / total_base - 1) * 100:+.2f}%)")

    print("\nBy occupancy band")
    print("-" * 70)
    bands = pd.cut(recommendations["baseline_occupancy"],
                   [-0.01, 0.5, 0.7, 0.85, 1.01],
                   labels=["<50%", "50-70%", "70-85%", ">85%"])
    print(recommendations.groupby(bands, observed=True).agg(
        nights=("rate", "size"),
        mean_rate_change=("rate_change_pct", "mean"),
        mean_uplift_pct=("revenue_uplift_pct", "mean")).round(2).to_string())

    # ---- Objective and uncertainty comparison ---------------------------
    print("\nStrategy comparison")
    print("-" * 70)
    variants = [("Revenue max (point estimate)", "elasticity", "expected_revenue"),
                ("Profit max  (point estimate)", "elasticity", "expected_profit"),
                ("Revenue max (conservative CI)", "ci_high", "expected_revenue")]
    rows = []
    for label, column, objective in variants:
        sim = make_simulator(df, bundle, elasticities, baseline, state,
                             elasticity_column=column)
        rec, _ = optimize(sim, state, objective=objective)
        rows.append({
            "strategy": label,
            "mean_rate_change_pct": round(rec["rate_change_pct"].mean(), 2),
            "revenue": round(rec["expected_revenue"].sum()),
            "profit": round(rec["expected_profit"].sum()),
        })
    rows.append({"strategy": "Do nothing (current rates)",
                 "mean_rate_change_pct": 0.0,
                 "revenue": round(recommendations["baseline_revenue"].sum()),
                 "profit": round(recommendations["baseline_profit"].sum())})
    strategies = pd.DataFrame(rows)
    print(strategies.to_string(index=False))

    recommendations.to_csv(REPORTS / "price_recommendations.csv", index=False)
    scenarios.to_csv(REPORTS / "price_scenarios.csv", index=False)
    strategies.to_csv(REPORTS / "strategy_comparison.csv", index=False)
    print(f"\nWrote reports/price_recommendations.csv")


if __name__ == "__main__":
    main()
