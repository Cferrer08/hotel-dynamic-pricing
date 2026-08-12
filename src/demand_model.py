"""

Predicts `rooms_booked` for a (hotel, room_type, stay_date, booking_date)
cell using all 36 predictive features, including the pace features that
Phase 5b had to exclude.

Different question from 5b, so different rules:
    5b asked  "what happens IF we change the price?"   -> causal
    5a asks   "how many rooms will sell?"              -> prediction

Models compared:
    Poisson GLM      transparent baseline
    Random Forest    non-linear, no distributional assumption
    XGBoost          gradient boosting with a Poisson objective
    CatBoost         gradient boosting, Poisson loss, native categoricals

Output:
    models/demand_model.pkl
    reports/model_comparison.csv

"""

import time
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

SEED = 42
TARGET = F.TARGET


# ==========================================================================
# Data preparation
# ==========================================================================
def prepare(df: pd.DataFrame):
    """One-hot the categoricals and return the three time-ordered splits.

    Note we do NOT scale features. Trees are scale-invariant, and the
    Poisson GLM here is fitted on features that are already in comparable
    units (logs, ratios, small integers).
    """
    data = df.copy()

    # Booleans to ints so every model sees numeric input.
    for col in data.columns:
        if data[col].dtype == bool:
            data[col] = data[col].astype(int)

    dummies = pd.get_dummies(data[F.CATEGORICALS], drop_first=True).astype(int)
    feature_cols = F.FEATURES_PREDICTIVE + list(dummies.columns)
    X = pd.concat([data[F.FEATURES_PREDICTIVE], dummies], axis=1)

    splits = {}
    for name in ["train", "val", "test", "live"]:
        mask = (data["split"] == name).to_numpy()
        splits[name] = (X[mask].astype(float), data.loc[mask, TARGET],
                        data[mask])
    return splits, feature_cols


# ==========================================================================
# Metrics
# ==========================================================================
def poisson_deviance(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y_true > 0, y_true * np.log(y_true / y_pred), 0.0)
    return float(2 * np.mean(term - (y_true - y_pred)))


def cell_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "poisson_deviance": poisson_deviance(y_true, y_pred),
        "bias": float(np.mean(y_pred - y_true)),
    }


def night_metrics(frame: pd.DataFrame, y_pred) -> dict:

    tmp = frame[["hotel_id", "room_type_id", "stay_date",
                 TARGET, "rooms_available"]].copy()
    tmp["pred"] = np.asarray(y_pred, dtype=float)
    agg = (tmp.groupby(["hotel_id", "room_type_id", "stay_date"])
           .agg(actual=(TARGET, "sum"), pred=("pred", "sum"),
                capacity=("rooms_available", "max"))
           .reset_index())
    err = agg["pred"] - agg["actual"]
    occ_err = err / agg["capacity"]
    return {
        "night_MAE": float(err.abs().mean()),
        "night_MAPE": float((err.abs() / agg["actual"].clip(lower=1)).mean()),
        "occupancy_MAE_pp": float(occ_err.abs().mean() * 100),
    }


# ==========================================================================
# Models
# ==========================================================================
def build_models() -> dict:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import PoissonRegressor
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor

    return {
        "Poisson GLM": PoissonRegressor(alpha=1e-4, max_iter=400),
        # max_samples grows each tree on 30% of rows. Random Forest is by
        # far the slowest model here - it fits every tree on the full
        # feature set with no early stopping, and unlike the boosters it
        # cannot stop when validation stops improving.
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=14, min_samples_leaf=25,
            max_samples=0.3, n_jobs=-1, random_state=SEED),
        "XGBoost": XGBRegressor(
            objective="count:poisson", n_estimators=600, learning_rate=0.05,
            max_depth=7, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, n_jobs=-1, random_state=SEED,
            early_stopping_rounds=40, eval_metric="poisson-nloglik"),
        "CatBoost": CatBoostRegressor(
            loss_function="Poisson", iterations=800, learning_rate=0.05,
            depth=7, random_seed=SEED, verbose=0,
            early_stopping_rounds=40),
    }


def train_all(splits) -> tuple:
    X_train, y_train, _ = splits["train"]
    X_val, y_val, _ = splits["val"]
    X_test, y_test, test_frame = splits["test"]

    results, fitted = [], {}
    for name, model in build_models().items():
        start = time.perf_counter()
        if name == "XGBoost":
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        elif name == "CatBoost":
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
        else:
            model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start

        pred = np.clip(model.predict(X_test), 0, None)
        row = {"model": name, "fit_seconds": round(elapsed, 1)}
        row.update({k: round(v, 4) for k, v in cell_metrics(y_test, pred).items()})
        row.update({k: round(v, 3) for k, v in night_metrics(test_frame, pred).items()})
        results.append(row)
        fitted[name] = model
        print(f"  {name:<16} {elapsed:>6.1f}s   "
              f"MAE={row['MAE']:.4f}  night_MAE={row['night_MAE']:.3f}")

    return pd.DataFrame(results), fitted


# ==========================================================================
# The point of this notebook: a great forecaster is a bad pricer
# ==========================================================================
def price_response_curve(model, splits, rate_grid=None) -> pd.DataFrame:

    X_test, _, frame = splits["test"]
    rate_grid = rate_grid if rate_grid is not None else np.linspace(0.7, 1.4, 15)

    sample_idx = frame.sample(n=min(20000, len(frame)), random_state=SEED).index
    positions = frame.index.get_indexer(sample_idx)
    base = X_test.iloc[positions].copy()
    base_rate = frame.loc[sample_idx, "rate"].to_numpy()
    reference = frame.loc[sample_idx, "base_rate"].to_numpy()

    rows = []
    for multiplier in rate_grid:
        scenario = base.copy()
        new_rate = base_rate * multiplier
        scenario["log_rate"] = np.log(new_rate)
        scenario["rate_vs_base"] = new_rate / reference
        # rate_vs_comp scales proportionally with our own rate
        scenario["rate_vs_comp"] = base["rate_vs_comp"].to_numpy() * multiplier
        rows.append({"rate_multiplier": multiplier,
                     "predicted_demand": float(np.clip(
                         model.predict(scenario), 0, None).mean())})
    curve = pd.DataFrame(rows)

    # Implied elasticity from the model's own price response
    slope = np.polyfit(np.log(curve["rate_multiplier"]),
                       np.log(curve["predicted_demand"].clip(lower=1e-6)), 1)[0]
    curve.attrs["implied_elasticity"] = slope
    return curve


# ==========================================================================
def main(sample_frac: float = None) -> None:
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    df = pd.read_parquet(PROCESSED / "features.parquet")
    splits, feature_cols = prepare(df)

    if sample_frac:
        # Subsample the TRAIN split only. Validation and test stay whole,
        # so reported metrics remain comparable across runs.
        X, y, frame = splits["train"]
        idx = X.sample(frac=sample_frac, random_state=SEED).index
        splits["train"] = (X.loc[idx], y.loc[idx], frame.loc[idx])
        print(f"[sampling] training on {sample_frac:.0%} of train rows\n")

    print("Split sizes")
    print("-" * 66)
    for name in ["train", "val", "test", "live"]:
        X, y, _ = splits[name]
        print(f"  {name:<6} {len(X):>8,} rows")
    print(f"  {len(feature_cols)} features after one-hot encoding\n")

    print("Training")
    print("-" * 66)
    comparison, fitted = train_all(splits)

    print("\nTest-set comparison")
    print("-" * 66)
    print(comparison.to_string(index=False))

    best_name = comparison.sort_values("poisson_deviance").iloc[0]["model"]
    best = fitted[best_name]
    print(f"\nBest by Poisson deviance: {best_name}")

    print("\nPrice response implied by the demand model")
    print("-" * 66)
    curve = price_response_curve(best, splits)
    elasticities = pd.read_csv(MODELS / "elasticities.csv")
    print(f"Implied elasticity from {best_name:<14}: "
          f"{curve.attrs['implied_elasticity']:+.3f}")
    print(f"Causal elasticity from Phase 5b        : "
          f"{elasticities['elasticity'].mean():+.3f}")
    print("\nThese disagree. The forecaster is not wrong - it is answering a")
    print("different question. It must not be used to set prices.")

    import pickle
    with open(MODELS / "demand_model.pkl", "wb") as fh:
        pickle.dump({"model": best, "name": best_name,
                     "features": feature_cols}, fh)
    comparison.to_csv(REPORTS / "model_comparison.csv", index=False)
    curve.to_csv(REPORTS / "demand_model_price_response.csv", index=False)
    print(f"\nWrote models/demand_model.pkl ({best_name})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=float, default=None,
                        help="Fraction of train rows to use (e.g. 0.2). "
                             "Useful on low-core machines; full data by default.")
    args = parser.parse_args()
    main(args.sample)
