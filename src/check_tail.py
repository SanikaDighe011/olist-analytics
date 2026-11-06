from pathlib import Path
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
o = pd.read_csv(RAW / "olist_orders_dataset.csv")
o["purchase_ts"] = pd.to_datetime(o["order_purchase_timestamp"])
o["day"] = o["purchase_ts"].dt.date

daily = o.groupby("day").size()
print("Last 20 days with orders in Aug-Sep 2018:")
print(daily[daily.index >= pd.Timestamp("2018-08-10").date()].head(20).to_string())

# days with zero orders inside our provisional window
window = pd.date_range("2017-01-01", "2018-08-31").date
missing = [d for d in window if d not in daily.index]
print("\nDays with NO orders inside 2017-01-01..2018-08-31:", len(missing))
print(missing[:10])