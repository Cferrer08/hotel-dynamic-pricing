"""

Grain of the main fact table is one row per:
    (hotel_id, room_type_id, stay_date, booking_date)

i.e. "how many rooms were picked up for this future night, on this day,
at the rate we were showing that day". This is the booking curve.

Outputs (data/raw/):
    fact_booking_pickup.parquet
    fact_competitor_rate.parquet
    _ground_truth_elasticity.csv     <- hidden answer key, used in Step 5

"""

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SEED = 42
TODAY = pd.Timestamp("2026-07-30")
BOOKING_WINDOW = 60          # we model the last 60 days before arrival
RAW_DIR = Path("data/raw")

# Share of stay dates where the rate is set RANDOMLY instead of by the
# legacy rule. This is the randomized experiment that makes the true
# elasticity recoverable in Step 5. Without it, the data is unidentified.
RATE_TEST_SHARE = 0.12

# How aggressively the legacy rule marks up on expected-busy nights.
# Higher = stronger confounding = harder (and more interesting) problem.
LEGACY_PRICE_SENSITIVITY = 0.55

# True price elasticity by room type, before segment adjustment.
# Cheaper rooms are shopped harder; suite buyers are far less price sensitive.
BASE_ELASTICITY = {"RT_STD": -1.70, "RT_DLX": -1.45, "RT_STE": -1.05}

# Room types do not all fill at the same rate relative to their inventory.
ROOM_TYPE_CAPTURE = {"RT_STD": 1.00, "RT_DLX": 0.90, "RT_STE": 0.75}

COMPETITORS = ["CompA", "CompB", "CompC"]

rng = np.random.default_rng(SEED)


# --------------------------------------------------------------------------
# Booking curve
# --------------------------------------------------------------------------
def booking_curve(lead_times: np.ndarray, shape: float, scale: float) -> np.ndarray:
    """Gamma-shaped pickup weights over the booking window.

    Business travellers book late (mass near lead_time 0), leisure travellers
    book far ahead. Returns weights that sum to 1.
    """
    x = lead_times + 0.5
    w = x ** (shape - 1.0) * np.exp(-x / scale)
    return w / w.sum()


def blended_curve(lead_times: np.ndarray, leisure_share: float) -> np.ndarray:
    business = booking_curve(lead_times, shape=2.0, scale=6.0)    # mean ~12 days
    leisure = booking_curve(lead_times, shape=2.2, scale=13.0)    # mean ~29 days
    return leisure_share * leisure + (1.0 - leisure_share) * business


# --------------------------------------------------------------------------
# Ground truth elasticity
# --------------------------------------------------------------------------
def build_ground_truth(dim_hotel: pd.DataFrame) -> pd.DataFrame:
    """The answer key. Step 5 must recover these WITHOUT looking at this file."""
    rows = []
    for h in dim_hotel.itertuples():
        # Leisure guests comparison-shop more than business travellers, whose
        # trips are often non-discretionary and expensed.
        segment_adj = 0.85 + 0.35 * h.leisure_share
        for rt, base in BASE_ELASTICITY.items():
            eps = base * segment_adj * rng.uniform(0.95, 1.05)
            rows.append({
                "hotel_id": h.hotel_id,
                "room_type_id": rt,
                "true_elasticity": round(float(eps), 4),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Main generation
# --------------------------------------------------------------------------
def main() -> None:
    dim_hotel = pd.read_csv(RAW_DIR / "dim_hotel.csv")
    dim_capacity = pd.read_csv(RAW_DIR / "dim_capacity.csv")
    dim_date = pd.read_csv(RAW_DIR / "dim_date.csv",
                           parse_dates=["calendar_date"])
    # Only bookable nights become stay dates; the earlier calendar rows exist
    # solely so booking_date_id has something to reference.
    dim_date = (dim_date[dim_date["is_stay_date"]]
                .rename(columns={"calendar_date": "stay_date"})
                .reset_index(drop=True))
    dim_event = pd.read_csv(RAW_DIR / "dim_event.csv",
                            parse_dates=["event_start", "event_end"])

    ground_truth = build_ground_truth(dim_hotel)
    elasticity_lookup = {
        (r.hotel_id, r.room_type_id): r.true_elasticity
        for r in ground_truth.itertuples()
    }

    lead_times = np.arange(BOOKING_WINDOW)          # 0 .. 59
    stay_dates = dim_date["stay_date"].to_numpy()
    n_stay = len(stay_dates)
    start_date = dim_date["stay_date"].min()

    # ---- Per stay-date calendar multipliers, looked up once ---------------
    cal = dim_date.set_index("stay_date")

    # ---- Event lift per (city, stay_date) ---------------------------------
    event_lift = {}
    for e in dim_event.itertuples():
        for d in pd.date_range(e.event_start, e.event_end):
            key = (e.city, d)
            event_lift[key] = max(event_lift.get(key, 1.0), e.demand_multiplier)

    booking_rows = []
    competitor_rows = []

    for h in dim_hotel.itertuples():
        curve = blended_curve(lead_times, h.leisure_share)

        # Which stay dates are randomized rate tests for this hotel?
        is_test = rng.random(n_stay) < RATE_TEST_SHARE

        # Blended day-of-week multiplier for this hotel's guest mix
        dow_mult = (h.leisure_share * cal["dow_mult_leisure"].to_numpy()
                    + (1 - h.leisure_share) * cal["dow_mult_business"].to_numpy())

        season = cal["season_mult"].to_numpy()
        holiday = cal["holiday_mult"].to_numpy()
        events = np.array([event_lift.get((h.city, d), 1.0)
                           for d in cal.index])

        # Slow market growth over the two years of history
        years_elapsed = (cal.index - start_date).days.to_numpy() / 365.25
        growth = 1.0 + 0.06 * years_elapsed

        # Unobserved night-level demand shock. This is the piece that makes
        # naive regression fail: the legacy rule partially "sees" it.
        night_noise = rng.lognormal(mean=0.0, sigma=0.18, size=n_stay)

        # Calendar demand index the legacy rule reacts to
        demand_index = dow_mult * season * holiday * events * growth
        forecast_index = demand_index * (night_noise ** 0.70)

        hotel_caps = dim_capacity[dim_capacity["hotel_id"] == h.hotel_id]

        # ---- Competitor rates (hotel level, not room type) ----------------
        # Competitors face the same demand, so their rates move with it.
        # This makes competitor rate a useful CONTROL variable in Step 5.
        for comp in COMPETITORS:
            comp_position = rng.uniform(0.90, 1.10)
            for si in range(n_stay):
                sd = stay_dates[si]
                valid = lead_times[(sd - lead_times.astype("timedelta64[D]")) <= TODAY]
                if len(valid) == 0:
                    continue
                comp_rate = (
                    h.base_rate * comp_position
                    * (demand_index[si] ** 0.50)
                    * rng.uniform(0.95, 1.05, size=len(valid))
                    * (1.0 + 0.08 * (1.0 - valid / BOOKING_WINDOW))
                )
                competitor_rows.append(pd.DataFrame({
                    "hotel_id": h.hotel_id,
                    "stay_date": sd,
                    "booking_date": sd - valid.astype("timedelta64[D]"),
                    "lead_time": valid,
                    "competitor_name": comp,
                    "competitor_rate": np.round(comp_rate, 2),
                }))

        # Average competitor index per stay date, used by our pricing rule
        comp_index = demand_index ** 0.50

        for cap in hotel_caps.itertuples():
            rooms = cap.rooms_available
            base_rate = cap.base_rate
            elasticity = elasticity_lookup[(h.hotel_id, cap.room_type_id)]
            capture = ROOM_TYPE_CAPTURE[cap.room_type_id]

            for si in range(n_stay):
                sd = stay_dates[si]
                booking_dates = sd - lead_times.astype("timedelta64[D]")
                mask = booking_dates <= TODAY
                if not mask.any():
                    continue

                lt = lead_times[mask]
                bd = booking_dates[mask]

                # ---- THE LEGACY PRICING RULE ---------------------------
                if is_test[si]:
                    # Randomized test: rate ignores demand entirely.
                    price_mult = np.full(len(lt), rng.uniform(0.70, 1.40))
                else:
                    # Prices UP on expected-busy nights -> confounding.
                    price_mult = np.full(
                        len(lt),
                        np.clip(forecast_index[si] ** LEGACY_PRICE_SENSITIVITY,
                                0.75, 1.90),
                    )
                # Rates drift up as arrival approaches
                lead_mult = 1.0 + 0.12 * (1.0 - lt / BOOKING_WINDOW)
                comp_mult = comp_index[si] ** 0.15
                rate = np.round(base_rate * price_mult * lead_mult * comp_mult, 2)

                # ---- DEMAND RESPONSE -----------------------------------
                lam_total = rooms * capture * demand_index[si] * night_noise[si]
                lam = lam_total * curve[mask]
                price_effect = (rate / base_rate) ** elasticity
                lam = lam * price_effect

                booked = rng.poisson(np.maximum(lam, 0.0))

                # ---- CAPACITY CONSTRAINT -------------------------------
                # Iterate from the earliest booking forward; once the room
                # type sells out, later pickup is truncated to zero.
                order = np.argsort(-lt)                 # far out -> arrival
                cum = np.cumsum(booked[order])
                capped = np.diff(np.minimum(cum, rooms), prepend=0)

                # Rooms already sold BEFORE this booking date ("on the books").
                on_books_ordered = np.cumsum(capped) - capped

                booked_final = np.empty_like(booked)
                on_books = np.empty_like(booked)
                booked_final[order] = capped
                on_books[order] = on_books_ordered

                # Once the room type is sold out, no rate is on sale any more.
                # Those rows are CENSORED - they are not demand observations,
                # and Step 5 must exclude them or elasticity will be badly
                # attenuated. Flagging them is what makes that possible.
                is_available = on_books < rooms

                # Zero-pickup rows are kept: no bookings is genuine signal.
                booking_rows.append(pd.DataFrame({
                    "stay_date": sd,
                    "booking_date": bd,
                    "lead_time": lt,
                    "hotel_id": h.hotel_id,
                    "room_type_id": cap.room_type_id,
                    "rate": rate,
                    "rooms_booked": booked_final,
                    "room_revenue": np.round(booked_final * rate, 2),
                    "rooms_on_books": on_books,
                    "is_available": is_available,
                    "is_rate_test": bool(is_test[si]),
                }))

        print(f"  generated {h.hotel_id} ({h.hotel_name})")

    fact_booking = pd.concat(booking_rows, ignore_index=True)
    fact_comp = pd.concat(competitor_rows, ignore_index=True)

    fact_booking.to_parquet(RAW_DIR / "fact_booking_pickup.parquet", index=False)
    fact_comp.to_parquet(RAW_DIR / "fact_competitor_rate.parquet", index=False)
    ground_truth.to_csv(RAW_DIR / "_ground_truth_elasticity.csv", index=False)

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------
    print("\nFiles written")
    print("-" * 52)
    print(f"fact_booking_pickup.parquet   {len(fact_booking):>9,} rows")
    print(f"fact_competitor_rate.parquet  {len(fact_comp):>9,} rows")
    print(f"_ground_truth_elasticity.csv  {len(ground_truth):>9,} rows")

    print("\nSanity checks")
    print("-" * 52)
    print(f"Total rooms sold        : {fact_booking['rooms_booked'].sum():,}")
    print(f"Total revenue           : ${fact_booking['room_revenue'].sum():,.0f}")
    print(f"Rate-test share of rows : "
          f"{fact_booking['is_rate_test'].mean():.1%}")
    print(f"Booking dates range     : {fact_booking['booking_date'].min().date()}"
          f" -> {fact_booking['booking_date'].max().date()}")

    # Did we ever oversell? Must be False.
    sold = (fact_booking.groupby(["hotel_id", "room_type_id", "stay_date"])
            ["rooms_booked"].sum().reset_index())
    sold = sold.merge(pd.read_csv(RAW_DIR / "dim_capacity.csv"),
                      on=["hotel_id", "room_type_id"])
    oversold = (sold["rooms_booked"] > sold["rooms_available"]).sum()
    print(f"Oversold night-types    : {oversold}  (must be 0)")

    closed = sold[sold["stay_date"] <= TODAY]
    occ = (closed["rooms_booked"] / closed["rooms_available"]).mean()
    print(f"Mean occupancy (closed) : {occ:.1%}")

    print(f"Censored (sold-out) rows: "
          f"{(~fact_booking['is_available']).mean():.1%}")


if __name__ == "__main__":
    main()
