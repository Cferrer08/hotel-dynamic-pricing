import pandas as pd
d = pd.read_parquet("data/raw/fact_booking_pickup.parquet")

# One night's booking curve
night = d[(d.hotel_id == "HTL001") & (d.room_type_id == "RT_STD") &
          (d.stay_date == "2025-10-15")].sort_values("lead_time", ascending=False)
print(night[["lead_time", "rate", "rooms_booked", "rooms_on_books"]].to_string(index=False))