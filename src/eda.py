"""
Exploratory Data Analysis for the Dynamic Pricing Intelligence Platform.

Reads from PostgreSQL (not the parquet files) so that the warehouse built in
Step 2 is actually exercised. Writes six figures to reports/ and prints the
findings that Steps 4-6 depend on.

"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # write files, no interactive window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sqlalchemy import text

from db import get_engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

TODAY = pd.Timestamp("2026-07-30")

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"


def save(fig, name: str) -> None:
    path = REPORTS / name
    fig.savefig(path)
    plt.close(fig)
    print(f"    saved {path.relative_to(PROJECT_ROOT)}")


# ==========================================================================
# Load
# ==========================================================================
def load_data(engine) -> pd.DataFrame:
    """Pull the analysis view. ~680k rows, comfortable in memory."""
    sql = """
        SELECT hotel_id, hotel_name, city, segment, room_type_id,
               room_type_name, stay_date, booking_date, lead_time, rate,
               rooms_booked, room_revenue, rooms_on_books, rooms_available,
               occupancy_on_books, is_available, is_rate_test,
               stay_day_name, stay_is_weekend, stay_is_holiday,
               avg_competitor_rate
        FROM pricing.vw_booking_analysis
    """
    df = pd.read_sql(text(sql), engine, parse_dates=["stay_date", "booking_date"])
    for col in ["rate", "room_revenue", "occupancy_on_books",
                "avg_competitor_rate"]:
        df[col] = df[col].astype(float)
    return df


# ==========================================================================
# 1. Portfolio KPIs
# ==========================================================================
def kpi_summary(df: pd.DataFrame) -> pd.DataFrame:
    """ADR, occupancy and RevPAR - the three numbers hotels live by.

    Only CLOSED stay dates (already past) are used. Including future nights
    that are still filling would understate occupancy and make the whole
    portfolio look like it is failing.
    """
    closed = df[df["stay_date"] <= TODAY]

    night = (closed.groupby(["hotel_name", "room_type_id", "stay_date"])
             .agg(rooms_sold=("rooms_booked", "sum"),
                  revenue=("room_revenue", "sum"),
                  capacity=("rooms_available", "max"))
             .reset_index())

    kpi = (night.groupby("hotel_name")
           .agg(rooms_sold=("rooms_sold", "sum"),
                revenue=("revenue", "sum"),
                room_nights=("capacity", "sum"))
           .reset_index())
    kpi["adr"] = kpi["revenue"] / kpi["rooms_sold"]
    kpi["occupancy"] = kpi["rooms_sold"] / kpi["room_nights"]
    kpi["revpar"] = kpi["revenue"] / kpi["room_nights"]
    return kpi.sort_values("revpar", ascending=False)


# ==========================================================================
# 2. Seasonality and day of week
# ==========================================================================
def plot_seasonality(df: pd.DataFrame) -> None:
    closed = df[df["stay_date"] <= TODAY]

    # Aggregate to (hotel, room_type, stay_date) FIRST. Summing
    # rooms_available directly over the raw rows would multiply capacity by
    # the number of booking dates - a classic fan-out error that silently
    # divides occupancy by 60.
    by_rt = (closed.groupby(["hotel_name", "segment", "room_type_id", "stay_date"])
             .agg(rooms_sold=("rooms_booked", "sum"),
                  revenue=("room_revenue", "sum"),
                  capacity=("rooms_available", "max"))
             .reset_index())
    night = (by_rt.groupby(["hotel_name", "segment", "stay_date"])
             [["rooms_sold", "revenue", "capacity"]].sum().reset_index())
    night["occupancy"] = night["rooms_sold"] / night["capacity"]
    night["month"] = night["stay_date"].dt.month
    night["day_name"] = night["stay_date"].dt.day_name()

    order = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]

    def ratio(frame, keys):
        g = frame.groupby(keys)[["rooms_sold", "revenue", "capacity"]].sum()
        g["occupancy"] = g["rooms_sold"] / g["capacity"]
        g["adr"] = g["revenue"] / g["rooms_sold"]
        return g.reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    sns.lineplot(data=ratio(night, ["segment", "month"]), x="month",
                 y="occupancy", hue="segment", marker="o", ax=axes[0])
    axes[0].set_title("Occupancy by month")
    axes[0].set_ylim(0.6, 1.0)

    sns.barplot(data=ratio(night, ["segment", "day_name"]), x="day_name",
                y="occupancy", hue="segment", order=order, ax=axes[1])
    axes[1].set_title("Occupancy by day of week")
    axes[1].set_ylim(0.6, 1.0)
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=45)

    # The panel that explains the other two: demand variation is absorbed by
    # RATE, not by occupancy. The legacy rule prices up on busy nights until
    # occupancy is nearly flat - which is what a rate-driven revenue
    # management system is designed to do.
    sns.barplot(data=ratio(night, ["segment", "day_name"]), x="day_name",
                y="adr", hue="segment", order=order, ax=axes[2])
    axes[2].set_title("ADR by day of week")
    axes[2].set_xlabel("")
    axes[2].tick_params(axis="x", rotation=45)

    for ax in axes:
        ax.legend(fontsize=7, title=None)

    fig.suptitle("Demand patterns", fontweight="bold")
    save(fig, "01_seasonality_dow.png")
    return night


# ==========================================================================
# 3. Booking curves
# ==========================================================================
def plot_booking_curves(df: pd.DataFrame) -> pd.DataFrame:
    """Pickup by lead time - the pace signal at the heart of the project."""
    closed = df[df["stay_date"] <= TODAY]
    curve = (closed.groupby(["segment", "lead_time"])["rooms_booked"]
             .mean().reset_index())

    pace = (closed.groupby(["segment", "lead_time"])["occupancy_on_books"]
            .mean().reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.lineplot(data=curve, x="lead_time", y="rooms_booked",
                 hue="segment", ax=axes[0])
    axes[0].invert_xaxis()
    axes[0].set_title("Daily pickup by lead time")
    axes[0].set_xlabel("Days before arrival")
    axes[0].set_ylabel("Mean rooms booked per day")

    sns.lineplot(data=pace, x="lead_time", y="occupancy_on_books",
                 hue="segment", ax=axes[1])
    axes[1].invert_xaxis()
    axes[1].set_title("Pace: occupancy already on the books")
    axes[1].set_xlabel("Days before arrival")
    axes[1].set_ylabel("Occupancy on books")

    fig.suptitle("Booking curves", fontweight="bold")
    save(fig, "02_booking_curves.png")
    return curve


# ==========================================================================
# 4. THE TRAP
# ==========================================================================
def plot_price_demand(df: pd.DataFrame) -> None:
    """The naive price-demand relationship, and what a controlled view shows.

    Left panel is what an analyst sees if they plot rate against bookings.
    Right panel restricts to randomized rate tests within one hotel and room
    type. The slopes point in opposite directions.
    """
    # Like-for-like comparison: SAME hotel, SAME room type, SAME lead-time
    # band. The only difference between the panels is whether the rate was
    # set by the legacy rule or randomized. Anything else would be comparing
    # price levels across products rather than a demand response.
    d = df[df["is_available"] & (df["hotel_id"] == "HTL001")
           & (df["room_type_id"] == "RT_STD")
           & (df["lead_time"].between(20, 40))].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

    def binned(frame, n):
        frame = frame.copy()
        frame["rate_bin"] = pd.qcut(frame["rate"], n, duplicates="drop")
        return frame.groupby("rate_bin", observed=True).agg(
            rate=("rate", "mean"),
            booked=("rooms_booked", "mean")).reset_index()

    naive = binned(d, 12)
    axes[0].plot(naive["rate"], naive["booked"], "o-", color="crimson")
    axes[0].set_title("Legacy rates (observational)")
    axes[0].set_xlabel("Rate ($)")
    axes[0].set_ylabel("Mean rooms booked per day")

    ctrl = binned(d[d["is_rate_test"]], 12)
    axes[1].plot(ctrl["rate"], ctrl["booked"], "o-", color="seagreen")
    axes[1].set_title("Randomized rate tests (experimental)")
    axes[1].set_xlabel("Rate ($)")

    fig.suptitle("Price vs demand - HTL001 standard rooms, 20-40 days out",
                 fontweight="bold")
    save(fig, "03_price_demand_trap.png")

    # Slopes in log-log space, i.e. crude elasticities
    naive_slope = np.polyfit(np.log(naive["rate"]),
                             np.log(naive["booked"].clip(lower=0.01)), 1)[0]
    ctrl_slope = np.polyfit(np.log(ctrl["rate"]),
                            np.log(ctrl["booked"].clip(lower=0.01)), 1)[0]
    return naive_slope, ctrl_slope


# ==========================================================================
# 5. Competitor positioning
# ==========================================================================
def plot_competitor(df: pd.DataFrame) -> None:
    std = df[(df["room_type_id"] == "RT_STD") & (df["stay_date"] <= TODAY)].copy()
    std["gap_pct"] = (std["rate"] / std["avg_competitor_rate"] - 1) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.boxplot(data=std, x="hotel_name", y="gap_pct", ax=axes[0],
                showfliers=False)
    axes[0].axhline(0, color="black", lw=1, ls="--")
    axes[0].set_title("Rate premium vs competitor set (standard rooms)")
    axes[0].set_ylabel("% above/below competitors")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=30)

    monthly = (std.set_index("stay_date")
               .groupby([pd.Grouper(freq="ME"), "hotel_name"])
               [["rate", "avg_competitor_rate"]].mean().reset_index())
    sns.lineplot(data=monthly, x="stay_date", y="rate", hue="hotel_name",
                 ax=axes[1], legend=False)
    sns.lineplot(data=monthly, x="stay_date", y="avg_competitor_rate",
                 hue="hotel_name", ax=axes[1], ls="--", legend=False)
    axes[1].set_title("Our rate (solid) vs competitors (dashed)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Mean rate ($)")
    axes[1].tick_params(axis="x", rotation=30)

    fig.suptitle("Competitive positioning", fontweight="bold")
    save(fig, "04_competitor_positioning.png")


# ==========================================================================
# 6. Are the rate tests actually random?
# ==========================================================================
def audit_rate_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Before trusting the experiment, check it looks like one.

    If rate-test nights differ systematically from normal nights on
    PRE-TREATMENT characteristics (day of week, holidays, season), the
    randomization is broken and the identification strategy fails.
    """
    closed = df[df["stay_date"] <= TODAY]
    nights = (closed.groupby(["hotel_id", "stay_date"])
              .agg(is_rate_test=("is_rate_test", "max"),
                   is_weekend=("stay_is_weekend", "max"),
                   is_holiday=("stay_is_holiday", "max"),
                   month=("stay_date", lambda s: s.iloc[0].month))
              .reset_index())
    balance = nights.groupby("is_rate_test").agg(
        nights=("stay_date", "size"),
        pct_weekend=("is_weekend", "mean"),
        pct_holiday=("is_holiday", "mean"),
        mean_month=("month", "mean"))
    return balance


# ==========================================================================
# 7. The live pricing problem
# ==========================================================================
def plot_open_dates(df: pd.DataFrame) -> pd.DataFrame:
    """The 60 stay dates still on sale - what Steps 5-6 must actually price."""
    open_nights = df[df["stay_date"] > TODAY]
    latest = (open_nights.sort_values("booking_date")
              .groupby(["hotel_name", "room_type_id", "stay_date"])
              .tail(1))
    by_night = (latest.groupby(["hotel_name", "stay_date"])
                .agg(on_books=("rooms_on_books", "sum"),
                     capacity=("rooms_available", "sum"),
                     rate=("rate", "mean"))
                .reset_index())
    by_night["occupancy"] = by_night["on_books"] / by_night["capacity"]
    by_night["days_out"] = (by_night["stay_date"] - TODAY).dt.days

    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.lineplot(data=by_night, x="stay_date", y="occupancy",
                 hue="hotel_name", marker="o", ms=3, ax=ax)
    ax.set_title("Current occupancy on the books for unsold stay dates",
                 fontweight="bold")
    ax.set_xlabel("Stay date")
    ax.set_ylabel("Occupancy on books")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=7)
    save(fig, "05_open_stay_dates.png")
    return by_night


# ==========================================================================
def main() -> None:
    engine = get_engine()
    print("Loading from PostgreSQL...")
    df = load_data(engine)
    print(f"  {len(df):,} rows, {df['stay_date'].nunique()} stay dates\n")

    print("Portfolio KPIs (closed stay dates)")
    print("-" * 68)
    kpi = kpi_summary(df)
    print(kpi.assign(
        adr=lambda x: x["adr"].round(2),
        occupancy=lambda x: (x["occupancy"] * 100).round(1),
        revpar=lambda x: x["revpar"].round(2),
        revenue=lambda x: (x["revenue"] / 1e6).round(2),
    )[["hotel_name", "revenue", "adr", "occupancy", "revpar"]]
        .rename(columns={"revenue": "revenue_$M", "occupancy": "occupancy_%"})
        .to_string(index=False))

    print("\nCharts")
    print("-" * 68)
    plot_seasonality(df)
    curve = plot_booking_curves(df)
    naive_slope, ctrl_slope = plot_price_demand(df)
    plot_competitor(df)
    open_nights = plot_open_dates(df)

    print("\nBooking pace by segment")
    print("-" * 68)
    pivot = (curve.assign(band=pd.cut(curve["lead_time"],
                                      [-1, 7, 21, 45, 60],
                                      labels=["0-7", "8-21", "22-45", "46-60"]))
             .groupby(["segment", "band"], observed=True)["rooms_booked"]
             .sum().unstack())
    print((pivot.div(pivot.sum(axis=1), axis=0) * 100).round(1)
          .to_string(float_format=lambda v: f"{v:.1f}%"))
    print("(share of total pickup by days-before-arrival band)")

    print("\nRandomization audit")
    print("-" * 68)
    print(audit_rate_tests(df).round(3).to_string())

    print("\nPrice vs demand")
    print("-" * 68)
    print(f"Naive log-log slope (pooled)      : {naive_slope:+.3f}")
    print(f"Controlled slope (rate tests only): {ctrl_slope:+.3f}")

    print("\nLive pricing problem")
    print("-" * 68)
    print(f"Open stay dates      : {open_nights['stay_date'].nunique()}")
    print(f"Mean occupancy       : {open_nights['occupancy'].mean():.1%}")
    print(f"Rooms still unsold   : "
          f"{(open_nights['capacity'] - open_nights['on_books']).sum():,.0f}")


if __name__ == "__main__":
    main()
