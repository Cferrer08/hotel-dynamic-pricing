"""
Phase 5b - Causal price elasticity estimation.

The question is NOT "how many rooms will we sell?" (that is Phase 5a). It is
"if we change the rate by 1%, what happens to bookings?" - a causal question
that observational data cannot answer here, because the legacy pricing rule
moved rates with demand.

Identification comes from the ~12% of stay dates where the rate was set
RANDOMLY (`is_rate_test`). On those nights price is independent of demand by
construction, so the association is the causal effect.

Estimator: Poisson GLM with a log link.
    log E[rooms_booked] = a + B*log(rate) + controls
B is then the own-price elasticity directly.

Output:
    models/elasticities.csv       <- consumed by Phase 6 (optimizer)

Run from the project root:
    python src/elasticity.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "models"
REPORTS = PROJECT_ROOT / "reports"

# The causal specification, arrived at in Phase 4:
#   log_rate       the treatment
#   lead_time      booking-curve position (exogenous, known in advance)
#   log_comp_rate  competitor rates move with market demand, so they proxy
#                  for the demand shock we cannot observe
# Deliberately ABSENT: every pace / on-books feature. They are post-treatment.
CAUSAL_CONTROLS = ["lead_time", "log_comp_rate"]
TREATMENT = "log_rate"
TARGET = "rooms_booked"


# ==========================================================================
# Sample construction
# ==========================================================================
def build_estimation_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Three restrictions, each with a reason.

    split != 'live'   the future has no outcomes yet
    is_rate_test      randomized rates only - this IS the identification
    is_available      once sold out no rate is on offer, so a zero is not a
                      demand response. Including these censored rows
                      attenuates the estimate by roughly 40%.
    """
    return df[(df["split"] != "live")
              & df["is_rate_test"]
              & df["is_available"]].copy()


# ==========================================================================
# Estimation
# ==========================================================================
def fit_poisson(group: pd.DataFrame,
                controls: list[str] = None,
                cluster: bool = True):
    """Poisson pseudo-MLE with standard errors clustered by stay date.

    Poisson rather than log-OLS because 60% of rows have zero pickup.
    log(0) is undefined, so log-OLS forces you to drop them - and that is
    selection on the outcome, since zeros are exactly what high prices
    produce. Poisson handles zeros natively.

    Poisson does not require the data to BE Poisson. As a quasi-MLE it is
    consistent for the conditional mean under much weaker assumptions; only
    the standard errors need care, which is what clustering handles.

    Clustered by stay_date because all bookings for one night share that
    night's demand shock. Treating them as independent understates the
    standard errors substantially.
    """
    controls = CAUSAL_CONTROLS if controls is None else controls
    cols = [TREATMENT] + controls
    X = sm.add_constant(group[cols].astype(float))
    model = sm.GLM(group[TARGET], X, family=sm.families.Poisson())
    if cluster:
        return model.fit(cov_type="cluster",
                         cov_kwds={"groups": group["stay_date"]})
    return model.fit()


def estimate_by_segment(sample: pd.DataFrame,
                        keys: list[str] = None,
                        controls: list[str] = None) -> pd.DataFrame:
    """One elasticity per (hotel, room_type) - the grain Phase 6 prices at."""
    keys = keys or ["hotel_id", "room_type_id"]
    rows = []
    for key, group in sample.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        fit = fit_poisson(group, controls)
        ci = fit.conf_int().loc[TREATMENT]
        rows.append({
            **dict(zip(keys, key)),
            "elasticity": fit.params[TREATMENT],
            "std_error": fit.bse[TREATMENT],
            "ci_low": ci[0],
            "ci_high": ci[1],
            "p_value": fit.pvalues[TREATMENT],
            "n_obs": len(group),
            "n_stay_dates": group["stay_date"].nunique(),
        })
    return pd.DataFrame(rows)


# ==========================================================================
# Robustness
# ==========================================================================
def fit_log_ols(group: pd.DataFrame) -> float:
    """Deliberately flawed alternative: drop zeros and run OLS on logs."""
    positive = group[group[TARGET] > 0]
    X = sm.add_constant(positive[[TREATMENT] + CAUSAL_CONTROLS].astype(float))
    return sm.OLS(np.log(positive[TARGET]), X).fit().params[TREATMENT]


def robustness_table(df: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """Each row weakens one design choice, so the cost of it is visible."""
    gt = ground_truth.set_index(["hotel_id", "room_type_id"])["true_elasticity"]
    closed = df[df["split"] != "live"]

    def score(sample, fn, label):
        estimates = []
        for (h, rt), g in sample.groupby(["hotel_id", "room_type_id"]):
            try:
                estimates.append((fn(g), gt.loc[(h, rt)]))
            except Exception:
                continue
        e = pd.DataFrame(estimates, columns=["est", "true"])
        return {"specification": label,
                "mean_elasticity": round(e["est"].mean(), 3),
                "MAE_vs_truth": round((e["est"] - e["true"]).abs().mean(), 3)}

    poisson = lambda g: fit_poisson(g, cluster=False).params[TREATMENT]

    rows = [
        score(build_estimation_sample(df), poisson,
              "1. Poisson | rate tests | available rows  [PREFERRED]"),
        score(build_estimation_sample(df), fit_log_ols,
              "2. log-OLS dropping zero-pickup rows"),
        score(closed[closed["is_rate_test"]], poisson,
              "3. Keeping censored (sold-out) rows"),
        score(closed[closed["is_available"]], poisson,
              "4. Observational data, no rate tests"),
    ]

    # Placebo: apply the identical estimator to NON-randomized rates,
    # labelled as if they were an experiment. It must fail. If it did not,
    # the result would be coming from the estimator rather than the design.
    rng = np.random.default_rng(7)
    observational = closed[~closed["is_rate_test"] & closed["is_available"]].copy()
    nights = observational[["hotel_id", "stay_date"]].drop_duplicates()
    nights["fake_test"] = rng.random(len(nights)) < 0.12
    observational = observational.merge(nights, on=["hotel_id", "stay_date"])
    rows.append(score(observational[observational["fake_test"]], poisson,
                      "5. PLACEBO: fake randomization on legacy rates"))

    return pd.DataFrame(rows)


# ==========================================================================
# Validation
# ==========================================================================
def validate(estimates: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    out = estimates.merge(ground_truth, on=["hotel_id", "room_type_id"])
    out["error"] = out["elasticity"] - out["true_elasticity"]
    out["covered"] = ((out["true_elasticity"] >= out["ci_low"])
                      & (out["true_elasticity"] <= out["ci_high"]))
    return out


# ==========================================================================
def main() -> None:
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    df = pd.read_parquet(PROCESSED / "features.parquet")
    ground_truth = pd.read_csv(PROJECT_ROOT / "data" / "raw"
                               / "_ground_truth_elasticity.csv")

    sample = build_estimation_sample(df)
    print("Estimation sample")
    print("-" * 66)
    print(f"Rows                 : {len(sample):,} "
          f"({len(sample) / len(df):.1%} of all data)")
    print(f"Stay dates           : {sample['stay_date'].nunique():,}")
    print(f"Zero-pickup rows     : {(sample[TARGET] == 0).mean():.1%} "
          f"<- why Poisson, not log-OLS")

    estimates = estimate_by_segment(sample)
    checked = validate(estimates, ground_truth)

    print("\nElasticity by hotel and room type")
    print("-" * 66)
    print(checked[["hotel_id", "room_type_id", "elasticity", "std_error",
                   "ci_low", "ci_high", "true_elasticity", "error", "covered"]]
          .round(3).to_string(index=False))

    print("\nValidation against ground truth")
    print("-" * 66)
    print(f"Mean absolute error       : {checked['error'].abs().mean():.3f}")
    print(f"Correlation with truth    : "
          f"{checked['elasticity'].corr(checked['true_elasticity']):.3f}")
    print(f"95% CI coverage           : {checked['covered'].mean():.1%} "
          f"({checked['covered'].sum()}/{len(checked)})")
    print(f"Correct sign              : {(checked['elasticity'] < 0).mean():.0%}")

    print("\nRobustness")
    print("-" * 66)
    robustness = robustness_table(df, ground_truth)
    print(robustness.to_string(index=False))

    estimates.to_csv(MODELS / "elasticities.csv", index=False)
    checked.to_csv(REPORTS / "elasticity_validation.csv", index=False)
    robustness.to_csv(REPORTS / "elasticity_robustness.csv", index=False)
    print(f"\nWrote models/elasticities.csv  (input to Phase 6)")


if __name__ == "__main__":
    main()
