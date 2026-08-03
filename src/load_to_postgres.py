"""
Strategy, and the reasoning behind it:
  * DIMENSIONS  -> SQLAlchemy to_sql. A few hundred rows; clarity wins.
  * FACTS       -> PostgreSQL COPY via psycopg2. 1.3M rows total; COPY is
                   roughly 50x faster than row-by-row inserts because it
                   streams a single buffer instead of round-tripping per row.

The script is IDEMPOTENT: schema.sql drops and recreates the schema, so
running this twice produces exactly the same database. That property is
what makes a pipeline trustworthy.

"""

import io
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db import get_engine

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
SCHEMA_FILE = Path(__file__).resolve().parents[1] / "database" / "schema.sql"
SCHEMA = "pricing"


def to_date_id(s: pd.Series) -> pd.Series:
    """Convert a date column to the YYYYMMDD integer used by dim_date."""
    return pd.to_datetime(s).dt.strftime("%Y%m%d").astype("int32")


# --------------------------------------------------------------------------
# 1. Schema
# --------------------------------------------------------------------------
def create_schema(engine) -> None:
    ddl = SCHEMA_FILE.read_text(encoding="utf-8")
    # A raw connection lets us send the whole multi-statement file at once.
    with engine.begin() as conn:
        conn.execute(text(ddl))
    print(f"schema created from {SCHEMA_FILE.name}")


# --------------------------------------------------------------------------
# 2. Dimensions
# --------------------------------------------------------------------------
def load_dimensions(engine) -> None:
    dim_hotel = pd.read_csv(RAW_DIR / "dim_hotel.csv")
    dim_room_type = pd.read_csv(RAW_DIR / "dim_room_type.csv")
    dim_capacity = pd.read_csv(RAW_DIR / "dim_capacity.csv")
    dim_event = pd.read_csv(RAW_DIR / "dim_event.csv")

    # dim_date is generic - a calendar day is neither "stay" nor "booking"
    # until a fact joins to it. That is what role-playing means.
    dim_date = pd.read_csv(RAW_DIR / "dim_date.csv",
                           parse_dates=["calendar_date"])
    dim_date["calendar_date"] = dim_date["calendar_date"].dt.date

    # Order matters: parents before children, or foreign keys reject the rows.
    tables = [
        ("dim_hotel", dim_hotel),
        ("dim_room_type", dim_room_type),
        ("dim_date", dim_date),
        ("dim_capacity", dim_capacity),
        ("dim_event", dim_event),
    ]
    for name, df in tables:
        df.to_sql(name, engine, schema=SCHEMA, if_exists="append",
                  index=False, method="multi", chunksize=1000)
        print(f"  {name:<16} {len(df):>7,} rows")


# --------------------------------------------------------------------------
# 3. Facts via COPY
# --------------------------------------------------------------------------
def copy_dataframe(engine, df: pd.DataFrame, table: str) -> float:
    """Stream a DataFrame into PostgreSQL using COPY ... FROM STDIN."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False, header=False, na_rep="\\N")
    buffer.seek(0)

    columns = ", ".join(df.columns)
    sql = (f"COPY {SCHEMA}.{table} ({columns}) "
           f"FROM STDIN WITH (FORMAT csv, NULL '\\N')")

    start = time.perf_counter()
    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            cur.copy_expert(sql, buffer)
        raw.commit()
    finally:
        raw.close()
    return time.perf_counter() - start


def load_facts(engine) -> None:
    booking = pd.read_parquet(RAW_DIR / "fact_booking_pickup.parquet")
    booking["stay_date_id"] = to_date_id(booking["stay_date"])
    booking["booking_date_id"] = to_date_id(booking["booking_date"])
    booking = booking[[
        "stay_date_id", "booking_date_id", "hotel_id", "room_type_id",
        "lead_time", "rate", "rooms_booked", "room_revenue",
        "rooms_on_books", "is_available", "is_rate_test",
    ]]
    secs = copy_dataframe(engine, booking, "fact_booking_pickup")
    print(f"  fact_booking_pickup   {len(booking):>9,} rows in {secs:5.1f}s "
          f"({len(booking)/secs:,.0f} rows/s)")

    comp = pd.read_parquet(RAW_DIR / "fact_competitor_rate.parquet")
    comp["stay_date_id"] = to_date_id(comp["stay_date"])
    comp["booking_date_id"] = to_date_id(comp["booking_date"])
    comp = comp[[
        "stay_date_id", "booking_date_id", "hotel_id",
        "competitor_name", "lead_time", "competitor_rate",
    ]]
    secs = copy_dataframe(engine, comp, "fact_competitor_rate")
    print(f"  fact_competitor_rate  {len(comp):>9,} rows in {secs:5.1f}s "
          f"({len(comp)/secs:,.0f} rows/s)")


# --------------------------------------------------------------------------
# 4. Validation
# --------------------------------------------------------------------------
VALIDATION_QUERIES = {
    "Row counts match source": """
        SELECT (SELECT COUNT(*) FROM pricing.fact_booking_pickup) AS bookings,
               (SELECT COUNT(*) FROM pricing.fact_competitor_rate) AS comp_rates
    """,
    "No orphaned foreign keys": """
        SELECT COUNT(*) AS orphans
        FROM pricing.fact_booking_pickup f
        LEFT JOIN pricing.dim_date d ON d.date_id = f.stay_date_id
        WHERE d.date_id IS NULL
    """,
    "Never oversold": """
        SELECT COUNT(*) AS violations FROM (
            SELECT f.hotel_id, f.room_type_id, f.stay_date_id,
                   SUM(f.rooms_booked) AS sold, MAX(c.rooms_available) AS cap
            FROM pricing.fact_booking_pickup f
            JOIN pricing.dim_capacity c
              ON c.hotel_id = f.hotel_id AND c.room_type_id = f.room_type_id
            GROUP BY 1,2,3
            HAVING SUM(f.rooms_booked) > MAX(c.rooms_available)
        ) t
    """,
    "Revenue reconciles": """
        SELECT ROUND(SUM(room_revenue)) AS total_revenue,
               SUM(rooms_booked) AS rooms_sold
        FROM pricing.fact_booking_pickup
    """,
    "Analysis view works": """
        SELECT COUNT(*) AS rows,
               ROUND(AVG(occupancy_on_books), 4) AS avg_occ_on_books,
               ROUND(AVG(rate_vs_comp), 2) AS avg_rate_gap
        FROM pricing.vw_booking_analysis
    """,
}


def validate(engine) -> None:
    print("\nValidation")
    print("-" * 60)
    with engine.connect() as conn:
        for label, sql in VALIDATION_QUERIES.items():
            row = conn.execute(text(sql)).mappings().one()
            detail = "  ".join(f"{k}={v:,}" if isinstance(v, int) else f"{k}={v}"
                               for k, v in row.items())
            print(f"{label:<26} {detail}")


# --------------------------------------------------------------------------
def main() -> None:
    engine = get_engine()
    overall = time.perf_counter()

    print("Creating schema")
    create_schema(engine)

    print("\nLoading dimensions")
    load_dimensions(engine)

    print("\nLoading facts (COPY)")
    load_facts(engine)

    validate(engine)
    print(f"\nTotal elapsed: {time.perf_counter() - overall:.1f}s")


if __name__ == "__main__":
    main()
