"""Phase 8a: build the daily revenue series used for forecasting."""
from pathlib import Path
import pandas as pd
from config import FORECAST_START, FORECAST_END

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"


def check(label, ok):
    print(("PASS  " if ok else "FAIL  ") + label)


orders = pd.read_csv(PROC / "fact_orders.csv", parse_dates=["purchase_date"])

# A SALE = not canceled/unavailable AND has at least one item (same rule as EDA)
sales = orders[orders["is_valid_sale"] & (orders["n_items"] > 0)]
in_window = sales[(sales["purchase_date"] >= FORECAST_START) &
                  (sales["purchase_date"] <= FORECAST_END)]

daily = in_window.groupby("purchase_date").agg(
    revenue=("product_revenue", "sum"),
    orders=("order_id", "count"),
)

# make sure EVERY calendar day exists
full_range = pd.date_range(FORECAST_START, FORECAST_END, freq="D")
missing_days = len(full_range.difference(daily.index))
daily = daily.reindex(full_range)
daily.index.name = "date"
daily = daily.fillna(0)

print("=== DAILY SERIES ===")
print(f"Window: {daily.index.min().date()} -> {daily.index.max().date()}")
print(f"Days: {len(daily)}   (expected 593)")
print(f"Days that had no orders (filled with 0): {missing_days}")
print(f"Days with zero revenue: {(daily['revenue'] == 0).sum()}")
print(f"Mean daily revenue: {daily['revenue'].mean():,.0f} BRL")
print(f"Min / Max daily revenue: {daily['revenue'].min():,.0f} / {daily['revenue'].max():,.0f}")

print("\n=== CHECKS ===")
check("593 consecutive days, no gaps", len(daily) == 593 and daily.index.is_unique)
check("daily revenue total == revenue of sales in the window",
      abs(daily["revenue"].sum() - in_window["product_revenue"].sum()) < 0.01)
check("no negative revenue", (daily["revenue"] >= 0).all())

daily.to_csv(PROC / "daily_sales.csv")
print("\nSaved data/processed/daily_sales.csv")