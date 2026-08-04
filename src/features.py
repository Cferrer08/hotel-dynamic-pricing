"""
Phase 4 - Feature engineering.

Builds the modelling table from the warehouse, with two guarantees:

  1. NO LEAKAGE. Every feature is computable from information available on
     the booking_date. Baselines (typical pace, typical pickup) are fitted
     on the TRAIN split only and applied to later splits.

  2. TWO FEATURE SETS. Some features are excellent predictors and invalid
     causal controls. See FEATURES_PREDICTIVE vs FEATURES_CAUSAL below.

Output:
    data/processed/features.parquet

Run from the project root:
    python src/features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from db import get_engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

TODAY = pd.Timestamp("2026-07-30")

# Time-based splits. NEVER random-split time series data: a random split
# lets the model see the future, and validation scores become fiction.
SPLIT_TRAIN_END = pd.Timestamp("2026-01-31")
SPLIT_VAL_END = pd.Timestamp("2026-04-30")
# test = SPLIT_VAL_END .. TODAY ; live = after TODAY


# ==========================================================================
# Load
# ==========================================================================
def load_base(engine) -> pd.DataFrame:
    """Pull the analysis view, then attach the small dimensions in pandas.

    Joining dim_capacity and dim_hotel inside this query looks tidier, but
    the view already contains a GROUP BY subquery for competitor averages;
    adding joins on top makes PostgreSQL re-plan that aggregate and the
    query time explodes. Two 15-row lookups are trivial to merge locally.
    """
    sql = """
        SELECT hotel_id, hotel_name, city, segment,
               room_type_id, room_type_name,
               stay_date, booking_date, lead_time,
               rate, rooms_booked, room_revenue,
               rooms_on_books, rooms_available, occupancy_on_books,
               is_available, is_rate_test,
               stay_day_name, stay_is_weekend, stay_is_holiday,
               stay_days_until_holiday, avg_competitor_rate
        FROM pricing.vw_booking_analysis
    """
    df = pd.read_sql(text(sql), engine,
                     parse_dates=["stay_date", "booking_date"])

    capacity = pd.read_sql(
        text("SELECT hotel_id, room_type_id, base_rate FROM pricing.dim_capacity"),
        engine)
    hotels = pd.read_sql(
        text("SELECT hotel_id, leisure_share, star_rating FROM pricing.dim_hotel"),
        engine)

    df = df.merge(capacity, on=["hotel_id", "room_type_id"], how="left")
    df = df.merge(hotels, on="hotel_id", how="left")

    for col in ["rate", "room_revenue", "occupancy_on_books",
                "avg_competitor_rate", "base_rate", "leisure_share"]:
        df[col] = df[col].astype(float)
    return df


def load_events(engine) -> pd.DataFrame:
    return pd.read_sql(text("SELECT * FROM pricing.dim_event"), engine,
                       parse_dates=["event_start", "event_end"])


# ==========================================================================
# Splits
# ==========================================================================
def add_split(df: pd.DataFrame) -> pd.DataFrame:
    conditions = [
        df["stay_date"] <= SPLIT_TRAIN_END,
        df["stay_date"] <= SPLIT_VAL_END,
        df["stay_date"] <= TODAY,
    ]
    df["split"] = np.select(conditions, ["train", "val", "test"],
                            default="live")
    return df


# ==========================================================================
# Calendar features
# ==========================================================================
def add_calendar(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """All known in advance, so all safe: no leakage risk."""
    df["stay_dow"] = df["stay_date"].dt.dayofweek
    df["stay_month"] = df["stay_date"].dt.month
    df["stay_week"] = df["stay_date"].dt.isocalendar().week.astype(int)
    df["stay_year"] = df["stay_date"].dt.year

    # Cyclical encoding: day 365 and day 1 are one day apart, but the raw
    # integers are 364 apart. sin/cos makes that adjacency explicit for
    # linear models. Trees do not need it but are not harmed by it.
    doy = df["stay_date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["dow_sin"] = np.sin(2 * np.pi * df["stay_dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["stay_dow"] / 7)

    # Event flag: city-scoped and date-ranged, so it needs an interval join
    # rather than a simple merge.
    event_days = []
    for e in events.itertuples():
        for d in pd.date_range(e.event_start, e.event_end):
            event_days.append((e.city, d, e.demand_multiplier))
    ev = (pd.DataFrame(event_days, columns=["city", "stay_date", "event_lift"])
          .groupby(["city", "stay_date"], as_index=False)["event_lift"].max())

    df = df.merge(ev, on=["city", "stay_date"], how="left")
    df["has_event"] = df["event_lift"].notna()
    df["event_lift"] = df["event_lift"].fillna(1.0).astype(float)
    return df


# ==========================================================================
# Price features
# ==========================================================================
def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """The treatment variable and its context.

    Ratios rather than raw dollars: a $50 move means something completely
    different on a $160 room than on a $500 suite.
    """
    df["log_rate"] = np.log(df["rate"])
    df["rate_vs_base"] = df["rate"] / df["base_rate"]

    # Competitor rates are quoted at HOTEL level (a standard-room benchmark),
    # so comparing a suite rate to them directly is meaningless. We compare
    # like with like by scaling our rate back to a standard-room equivalent.
    std_equiv = df["rate"] / (df["base_rate"] / df.groupby("hotel_id")
                              ["base_rate"].transform("min"))
    df["rate_vs_comp"] = std_equiv / df["avg_competitor_rate"]
    df["log_comp_rate"] = np.log(df["avg_competitor_rate"])
    df["undercuts_comp"] = df["rate_vs_comp"] < 1.0
    return df


# ==========================================================================
# Pace features - fitted on TRAIN only
# ==========================================================================
def fit_pace_baseline(train: pd.DataFrame) -> dict:
    """Learn what 'normal' looks like at each lead time.

    Phase 3 showed business hotels take 28% of bookings in the final week
    while leisure hotels take 13%. So raw occupancy_on_books is not
    comparable across hotels - it must be judged against that hotel and
    room type's own typical curve.
    """
    typical_pace = (train.groupby(["hotel_id", "room_type_id", "lead_time"])
                    ["occupancy_on_books"].mean()
                    .rename("typical_occ_on_books").reset_index())

    typical_pickup = (train.groupby(["hotel_id", "room_type_id", "lead_time"])
                      ["rooms_booked"].mean()
                      .rename("typical_pickup").reset_index())

    # Same curves split by day of week, since a Saturday fills on a
    # different schedule from a Tuesday.
    typical_pace_dow = (train.groupby(["hotel_id", "room_type_id",
                                       "stay_dow", "lead_time"])
                        ["occupancy_on_books"].mean()
                        .rename("typical_occ_dow").reset_index())

    return {"pace": typical_pace,
            "pickup": typical_pickup,
            "pace_dow": typical_pace_dow}


def apply_pace_baseline(df: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    df = df.merge(baseline["pace"], how="left",
                  on=["hotel_id", "room_type_id", "lead_time"])
    df = df.merge(baseline["pickup"], how="left",
                  on=["hotel_id", "room_type_id", "lead_time"])
    df = df.merge(baseline["pace_dow"], how="left",
                  on=["hotel_id", "room_type_id", "stay_dow", "lead_time"])

    # pace_gap: percentage points ahead of / behind normal.
    # This is THE revenue management signal - "we are 8 points behind pace
    # for this night" is what a revenue manager actually says.
    df["pace_gap"] = df["occupancy_on_books"] - df["typical_occ_on_books"]
    df["pace_gap_dow"] = df["occupancy_on_books"] - df["typical_occ_dow"]

    # Ratio form, guarded against divide-by-zero at long lead times when
    # typical occupancy is near zero.
    df["pace_ratio"] = (df["occupancy_on_books"]
                        / df["typical_occ_on_books"].clip(lower=0.01))
    df["pace_ratio"] = df["pace_ratio"].clip(upper=5.0)

    df["rooms_remaining"] = df["rooms_available"] - df["rooms_on_books"]
    df["pct_remaining"] = df["rooms_remaining"] / df["rooms_available"]
    return df


# ==========================================================================
# Momentum - trailing demand, strictly backward-looking
# ==========================================================================
def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Recent booking activity at the hotel, as of the booking date.

    Uses .shift(1) before rolling so the current day is EXCLUDED. Without
    that shift the feature contains today's bookings - which is the target.
    That single missing line is one of the most common leaks in demand
    forecasting.
    """
    daily = (df.groupby(["hotel_id", "booking_date"])["rooms_booked"]
             .sum().rename("hotel_daily_pickup").reset_index()
             .sort_values(["hotel_id", "booking_date"]))

    grouped = daily.groupby("hotel_id")["hotel_daily_pickup"]
    # min_periods=1 so the first few days of data use whatever history
    # exists rather than becoming NaN. The shift(1) still guarantees no
    # leakage; only the very first booking date has no prior day at all.
    daily["momentum_7d"] = (grouped.shift(1)
                            .rolling(7, min_periods=1).mean()
                            .reset_index(level=0, drop=True))
    daily["momentum_28d"] = (grouped.shift(1)
                             .rolling(28, min_periods=1).mean()
                             .reset_index(level=0, drop=True))
    for col in ["momentum_7d", "momentum_28d"]:
        daily[col] = daily[col].fillna(0.0)
    daily["momentum_ratio"] = (daily["momentum_7d"]
                               / daily["momentum_28d"].clip(lower=0.1))

    return df.merge(
        daily[["hotel_id", "booking_date", "momentum_7d",
               "momentum_28d", "momentum_ratio"]],
        on=["hotel_id", "booking_date"], how="left")


def add_recent_pickup(df: pd.DataFrame) -> pd.DataFrame:
    """Recent pickup for THIS specific stay date.

    Much sharper than hotel-level momentum: if this night took 5 rooms
    yesterday it will probably take a few more today. Strongest predictor
    in the whole table - and the one where forgetting .shift(1) does the
    most damage, because the window is dominated by the target itself.
    """
    df = df.sort_values(["hotel_id", "room_type_id",
                         "stay_date", "booking_date"])
    grouped = df.groupby(["hotel_id", "room_type_id", "stay_date"])["rooms_booked"]

    # shift(1) FIRST, then roll. Today's bookings must be excluded.
    df["pickup_lag_1d"] = grouped.shift(1)
    df["pickup_lag_3d"] = grouped.transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["pickup_lag_7d"] = grouped.transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean())

    for col in ["pickup_lag_1d", "pickup_lag_3d", "pickup_lag_7d"]:
        df[col] = df[col].fillna(0.0)
    return df


# ==========================================================================
# Feature sets
# ==========================================================================
# Safe for BOTH models. Everything here is determined before the rate is
# set, so nothing here is downstream of the treatment.
FEATURES_EXOGENOUS = [
    "lead_time", "stay_dow", "stay_month", "stay_week",
    "doy_sin", "doy_cos", "dow_sin", "dow_cos",
    "stay_is_weekend", "stay_is_holiday", "stay_days_until_holiday",
    "has_event", "event_lift",
    "leisure_share", "star_rating", "rooms_available", "base_rate",
]

# The treatment and its market context.
FEATURES_PRICE = ["log_rate", "rate_vs_base", "rate_vs_comp", "log_comp_rate"]

# POST-TREATMENT. These depend on how many rooms we already sold, which
# depends on the rates we already charged. Great predictors; invalid
# causal controls. Conditioning on them blocks the very path we want to
# measure - see notebooks/05 for the demonstration.
FEATURES_POST_TREATMENT = [
    "rooms_on_books", "occupancy_on_books", "pace_gap", "pace_gap_dow",
    "pace_ratio", "rooms_remaining", "pct_remaining",
    "typical_occ_on_books", "typical_pickup",
    "momentum_7d", "momentum_28d", "momentum_ratio",
    "pickup_lag_1d", "pickup_lag_3d", "pickup_lag_7d",
]

FEATURES_PREDICTIVE = FEATURES_EXOGENOUS + FEATURES_PRICE + FEATURES_POST_TREATMENT
FEATURES_CAUSAL = FEATURES_EXOGENOUS + FEATURES_PRICE

CATEGORICALS = ["hotel_id", "room_type_id", "segment", "city"]
TARGET = "rooms_booked"


# ==========================================================================
def build(engine=None) -> pd.DataFrame:
    engine = engine or get_engine()
    df = load_base(engine)
    events = load_events(engine)

    df = add_split(df)
    df = add_calendar(df, events)
    df = add_price_features(df)

    # Baselines learned from TRAIN only, then applied everywhere. Fitting
    # them on the full dataset would leak future information backwards.
    train = df[df["split"] == "train"]
    baseline = fit_pace_baseline(train)
    df = apply_pace_baseline(df, baseline)

    df = add_momentum(df)
    df = add_recent_pickup(df)
    return df.sort_values(["hotel_id", "room_type_id",
                           "stay_date", "booking_date"]).reset_index(drop=True)


def leakage_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Assertions that must hold. A failure here invalidates everything after."""
    results = []

    def check(name, condition, detail=""):
        results.append({"check": name,
                        "pass": bool(condition),
                        "detail": detail})

    check("booking_date never after stay_date",
          (df["booking_date"] <= df["stay_date"]).all())
    check("no booking_date beyond TODAY",
          (df["booking_date"] <= TODAY).all())
    check("lead_time matches date difference",
          ((df["stay_date"] - df["booking_date"]).dt.days
           == df["lead_time"]).all())

    # Lagged features must be strictly backward-looking. A correctly
    # shifted lag correlates strongly with the target (that is why it is
    # useful) but must not be near-identical to it.
    corr = df[["momentum_7d", TARGET]].dropna().corr().iloc[0, 1]
    check("momentum not a copy of target", abs(corr) < 0.5,
          f"corr={corr:+.3f}")
    corr_lag = df[["pickup_lag_3d", TARGET]].dropna().corr().iloc[0, 1]
    check("pickup lag shifted correctly", abs(corr_lag) < 0.80,
          f"corr={corr_lag:+.3f}")

    # Baselines came from train, so val/test must have some unmatched rows
    # only if a combination never appeared in train.
    missing = df["typical_occ_on_books"].isna().mean()
    check("pace baseline coverage", missing < 0.01, f"missing={missing:.3%}")

    n_missing = df[FEATURES_PREDICTIVE].isna().sum().sum()
    check("no missing values in features", n_missing == 0,
          f"missing={n_missing}")

    check("no infinities",
          not np.isinf(df.select_dtypes(include=[np.number])).to_numpy().any())

    return pd.DataFrame(results)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df = build()

    print("Split sizes")
    print("-" * 62)
    summary = (df.groupby("split")
               .agg(rows=("rooms_booked", "size"),
                    stay_dates=("stay_date", "nunique"),
                    first=("stay_date", "min"),
                    last=("stay_date", "max"))
               .reindex(["train", "val", "test", "live"]))
    print(summary.to_string())

    print("\nLeakage checks")
    print("-" * 62)
    checks = leakage_checks(df)
    for r in checks.itertuples():
        mark = "OK " if r._2 else "!! "
        print(f"{mark}{r.check:<38} {r.detail}")

    out = PROCESSED / "features.parquet"
    df.to_parquet(out, index=False)
    print(f"\nWrote {len(df):,} rows x {df.shape[1]} columns -> "
          f"{out.relative_to(PROJECT_ROOT)}")
    print(f"Predictive features: {len(FEATURES_PREDICTIVE)}")
    print(f"Causal features    : {len(FEATURES_CAUSAL)}")


if __name__ == "__main__":
    main()
