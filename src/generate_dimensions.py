"""
Generates the "who / what / when" scaffolding that the booking facts will
reference in Step 1c:

    dim_hotel      one row per hotel
    dim_room_type  one row per room type
    dim_capacity   one row per (hotel, room_type) -> how many rooms exist
    dim_date       one row per calendar date  (national / seasonal effects)
    dim_event      one row per city-level event  (local demand shocks)

"""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

# --------------------------------------------------------------------------
# Configuration - every knob for this step lives here
# --------------------------------------------------------------------------
SEED = 42

# "Today" for the whole project. Stay dates after this are still unsold,
# which is what makes the final pricing problem a live one.
TODAY = pd.Timestamp("2026-07-30")

STAY_START = pd.Timestamp("2024-08-01")   # first night we have history for
STAY_END = pd.Timestamp("2026-09-28")     # last bookable night (~60 days out)

# dim_date is a GENERIC calendar, not a list of stay dates. The booking fact
# joins to it twice - once as the stay date, once as the booking date - and
# bookings for the first stay night were taken up to 60 days earlier. The
# calendar must therefore start before STAY_START or those foreign keys break.
BOOKING_WINDOW = 60
CALENDAR_START = STAY_START - pd.Timedelta(days=BOOKING_WINDOW)

RAW_DIR = Path("data/raw")

rng = np.random.default_rng(SEED)


# --------------------------------------------------------------------------
# 1. Hotels
# --------------------------------------------------------------------------
# leisure_share drives the day-of-week shape of demand. A business hotel
# fills Tue/Wed and empties on weekends; a resort does the exact opposite.
# Getting this right is what makes the data feel real to anyone in the industry.
HOTELS = [
    # hotel_id, name,               city,     star, rooms, leisure_share, base_rate
    ("HTL001", "Harborview Grand",  "Boston",   4,   220,   0.35,  245.0),
    ("HTL002", "Lakeside Inn",      "Chicago",  3,   140,   0.50,  158.0),
    ("HTL003", "Beachcomber Resort", "Miami",   4,   260,   0.85,  289.0),
    ("HTL004", "Metro Central",     "New York", 3,   180,   0.25,  212.0),
    ("HTL005", "Canyon Lodge",      "Denver",   4,   160,   0.75,  196.0),
]


def build_dim_hotel() -> pd.DataFrame:
    df = pd.DataFrame(
        HOTELS,
        columns=[
            "hotel_id", "hotel_name", "city", "star_rating",
            "total_rooms", "leisure_share", "base_rate",
        ],
    )
    # Segment label is redundant with leisure_share but useful for readable EDA.
    df["segment"] = np.where(
        df["leisure_share"] >= 0.65, "Leisure",
        np.where(df["leisure_share"] <= 0.35, "Business", "Mixed"),
    )
    return df


# --------------------------------------------------------------------------
# 2. Room types
# --------------------------------------------------------------------------
# rate_index multiplies the hotel's base rate. room_share splits the physical
# building into inventory buckets.
ROOM_TYPES = [
    # room_type_id, name,      rate_index, room_share
    ("RT_STD", "Standard", 1.00, 0.70),
    ("RT_DLX", "Deluxe",   1.35, 0.22),
    ("RT_STE", "Suite",    2.05, 0.08),
]


def build_dim_room_type() -> pd.DataFrame:
    return pd.DataFrame(
        ROOM_TYPES,
        columns=["room_type_id", "room_type_name", "rate_index", "room_share"],
    )


# --------------------------------------------------------------------------
# 3. Capacity  (hotel x room_type)
# --------------------------------------------------------------------------
# Fixed capacity is THE constraint in hotel pricing. Unlike retail you cannot
# restock, and an unsold night is revenue that is gone forever. This table is
# what turns "units sold" into "occupancy".
def build_dim_capacity(dim_hotel: pd.DataFrame,
                       dim_room_type: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in dim_hotel.itertuples():
        allocated = 0
        for i, rt in enumerate(dim_room_type.itertuples()):
            if i < len(dim_room_type) - 1:
                n = int(round(h.total_rooms * rt.room_share))
                allocated += n
            else:
                # Last bucket absorbs rounding so the parts sum to the whole.
                n = h.total_rooms - allocated
            rows.append({
                "hotel_id": h.hotel_id,
                "room_type_id": rt.room_type_id,
                "rooms_available": n,
                "base_rate": round(h.base_rate * rt.rate_index, 2),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4. Calendar
# --------------------------------------------------------------------------
def build_dim_date(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    df = pd.DataFrame({"calendar_date": dates})

    df["year"] = df["calendar_date"].dt.year
    df["month"] = df["calendar_date"].dt.month
    df["quarter"] = df["calendar_date"].dt.quarter
    df["day_of_week"] = df["calendar_date"].dt.dayofweek          # 0 = Monday
    df["day_name"] = df["calendar_date"].dt.day_name()
    df["week_of_year"] = df["calendar_date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["day_of_week"].isin([4, 5])          # Fri/Sat nights

    # --- Day-of-week demand shape, one column per traveller type -----------
    # Index 0=Mon ... 6=Sun. Business travel dies on the weekend; leisure
    # travel is the mirror image. Hotels blend these by their leisure_share.
    business_dow = np.array([1.05, 1.18, 1.20, 0.95, 0.55, 0.50, 0.72])
    leisure_dow = np.array([0.72, 0.68, 0.72, 0.88, 1.35, 1.45, 0.95])
    df["dow_mult_business"] = business_dow[df["day_of_week"].to_numpy()]
    df["dow_mult_leisure"] = leisure_dow[df["day_of_week"].to_numpy()]

    # --- Annual seasonality -------------------------------------------------
    # Smooth sine wave peaking in mid-July, plus a hard shoulder-season dip in
    # January/February. Two overlapping effects, because real seasonality is
    # never a single clean curve.
    doy = df["calendar_date"].dt.dayofyear.to_numpy()
    seasonal = 1.0 + 0.22 * np.sin(2 * np.pi * (doy - 105) / 365.25)
    winter_dip = np.where(df["month"].isin([1, 2]).to_numpy(), 0.88, 1.0)
    df["season_mult"] = np.round(seasonal * winter_dip, 4)

    # --- US federal holidays -----------------------------------------------
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=start, end=end, return_name=True)
    holiday_map = holidays.to_dict()
    df["holiday_name"] = df["calendar_date"].map(holiday_map)
    df["is_holiday"] = df["holiday_name"].notna()

    # Days until the *next* holiday - a genuinely predictive feature, since
    # demand builds in the nights leading up to a long weekend.
    holiday_dates = pd.DatetimeIndex(sorted(holiday_map.keys()))
    pos = holiday_dates.searchsorted(df["calendar_date"].to_numpy(), side="left")
    pos = np.clip(pos, 0, len(holiday_dates) - 1)
    df["days_until_holiday"] = (
        holiday_dates[pos].to_numpy() - df["calendar_date"].to_numpy()
    ).astype("timedelta64[D]").astype(int)

    # Holiday demand lift: strongest on the day itself, tapering 3 days out.
    lift = np.where(df["is_holiday"], 1.30,
                    np.where(df["days_until_holiday"] <= 3, 1.12, 1.0))
    df["holiday_mult"] = lift

    df["date_id"] = df["calendar_date"].dt.strftime("%Y%m%d").astype(int)
    # Marks which calendar days are actually bookable nights in our dataset.
    df["is_stay_date"] = df["calendar_date"] >= STAY_START

    cols = ["date_id", "calendar_date", "is_stay_date", "year", "quarter", "month", "week_of_year",
            "day_of_week", "day_name", "is_weekend", "dow_mult_business",
            "dow_mult_leisure", "season_mult", "is_holiday", "holiday_name",
            "days_until_holiday", "holiday_mult"]
    return df[cols]


# --------------------------------------------------------------------------
# 5. City events
# --------------------------------------------------------------------------
# Events are city-scoped, not national, so they get their own table rather
# than being crammed into dim_date. Keeping tables at one clean grain is the
# habit that makes the star schema in Step 2 easy instead of painful.
EVENT_TEMPLATES = {
    "Boston":   [("Biotech Summit", 2.1), ("Marathon Weekend", 2.6),
                 ("University Graduation", 2.3)],
    "Chicago":  [("Manufacturing Expo", 1.9), ("Lakefront Music Fest", 2.2),
                 ("Medical Convention", 2.0)],
    "Miami":    [("Art Basel Week", 2.8), ("Boat Show", 2.1),
                 ("New Year Beach Fest", 2.4)],
    "New York": [("Fashion Week", 2.5), ("Tech Conference", 2.0),
                 ("Marathon Weekend", 2.3)],
    "Denver":   [("Ski Season Opener", 2.2), ("Outdoor Retailer Show", 2.4),
                 ("Craft Beer Festival", 1.9)],
}


def build_dim_event(start: pd.Timestamp, end: pd.Timestamp,
                    cities: list[str]) -> pd.DataFrame:
    rows = []
    event_id = 1
    for city in cities:
        templates = EVENT_TEMPLATES[city]
        # Roughly one event per city per month, drawn from that city's list.
        n_months = (end.year - start.year) * 12 + (end.month - start.month) + 1
        for m in range(n_months):
            if rng.random() > 0.55:          # not every month gets an event
                continue
            month_start = (start + pd.DateOffset(months=m)).replace(day=1)
            if month_start > end:
                break
            name, lift = templates[rng.integers(len(templates))]
            day_offset = int(rng.integers(0, 26))
            ev_start = month_start + pd.Timedelta(days=day_offset)
            duration = int(rng.integers(2, 5))
            ev_end = ev_start + pd.Timedelta(days=duration - 1)
            if ev_start < start or ev_end > end:
                continue
            rows.append({
                "event_id": f"EVT{event_id:04d}",
                "city": city,
                "event_name": name,
                "event_start": ev_start,
                "event_end": ev_end,
                "duration_nights": duration,
                # Jitter the lift so identical events are not identical shocks.
                "demand_multiplier": round(float(lift * rng.uniform(0.85, 1.15)), 3),
            })
            event_id += 1
    return pd.DataFrame(rows).sort_values(["city", "event_start"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    dim_hotel = build_dim_hotel()
    dim_room_type = build_dim_room_type()
    dim_capacity = build_dim_capacity(dim_hotel, dim_room_type)
    dim_date = build_dim_date(CALENDAR_START, STAY_END)
    dim_event = build_dim_event(STAY_START, STAY_END,
                                dim_hotel["city"].tolist())

    outputs = {
        "dim_hotel.csv": dim_hotel,
        "dim_room_type.csv": dim_room_type,
        "dim_capacity.csv": dim_capacity,
        "dim_date.csv": dim_date,
        "dim_event.csv": dim_event,
    }
    for filename, frame in outputs.items():
        frame.to_csv(RAW_DIR / filename, index=False)
        print(f"  {filename:<20} {len(frame):>6,} rows -> {RAW_DIR / filename}")

    print("\nSanity checks")
    print("-" * 46)
    total_rooms = dim_capacity["rooms_available"].sum()
    print(f"Total rooms across portfolio : {total_rooms:,}")
    print(f"Matches dim_hotel total      : "
          f"{total_rooms == dim_hotel['total_rooms'].sum()}")
    print(f"Calendar days                : {len(dim_date):,} "
          f"({CALENDAR_START.date()} -> {STAY_END.date()})")
    print(f"Bookable stay dates          : {dim_date['is_stay_date'].sum():,}")
    print(f"Open (unsold) stay dates     : "
          f"{((dim_date['calendar_date'] > TODAY) & dim_date['is_stay_date']).sum():,}")
    print(f"Holidays flagged             : {dim_date['is_holiday'].sum()}")
    print(f"Events generated             : {len(dim_event)}")
    print(f"Season multiplier range      : "
          f"{dim_date['season_mult'].min():.3f} - {dim_date['season_mult'].max():.3f}")


if __name__ == "__main__":
    main()
