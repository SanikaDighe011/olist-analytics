"""Phase 8e: customer segmentation on recency, spend and review score."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "forecasts"

orders = pd.read_csv(PROC / "fact_orders.csv", parse_dates=["purchase_date"])
sales = orders[orders["is_valid_sale"] & (orders["n_items"] > 0)]

# reference date = day after the last good day of data (same cutoff as forecasting)
REF = pd.Timestamp("2018-08-21")
sales = sales[sales["purchase_date"] < REF]

cust = sales.groupby("customer_unique_id").agg(
    last_purchase=("purchase_date", "max"),
    orders=("order_id", "count"),
    total_spend=("product_revenue", "sum"),
    avg_review=("review_score", "mean"),
    state=("customer_state", "last"),
).reset_index()
cust["recency_days"] = (REF - cust["last_purchase"]).dt.days

# rule-based, easy to explain (thresholds are OUR choices, not from Olist)
spend_cut = cust["total_spend"].quantile(0.75)
recent_cut = 180


def segment(r):
    if r["orders"] >= 2:
        return "Repeat customer"
    if r["total_spend"] >= spend_cut and r["recency_days"] <= recent_cut:
        return "High-value recent"
    if r["recency_days"] <= recent_cut:
        return "Recent one-time"
    return "Lapsed one-time"


cust["segment"] = cust.apply(segment, axis=1)

summary = cust.groupby("segment").agg(
    customers=("customer_unique_id", "count"),
    revenue=("total_spend", "sum"),
    avg_spend=("total_spend", "mean"),
    avg_review=("avg_review", "mean"),
    avg_recency_days=("recency_days", "mean"),
).round(2)
summary["revenue_share_%"] = (summary["revenue"] / summary["revenue"].sum() * 100).round(1)

print(f"Customers: {len(cust):,} | spend threshold (top 25%): {spend_cut:,.2f} BRL")
print(summary.sort_values("revenue", ascending=False).to_string())
print(f"\nRevenue check: segments {cust['total_spend'].sum():,.2f} vs sales {sales['product_revenue'].sum():,.2f}")
assert abs(cust["total_spend"].sum() - sales["product_revenue"].sum()) < 0.01
assert cust["customer_unique_id"].is_unique

cust.to_csv(OUT / "customer_segments.csv", index=False)
print("Saved data/forecasts/customer_segments.csv")