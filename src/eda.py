"""Phase 6: Exploratory Data Analysis on the cleaned Olist tables."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # save charts to files, no pop-up windows
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=150)
    plt.close()
    print("   saved", name)


orders = pd.read_csv(PROC / "fact_orders.csv",
                     parse_dates=["purchase_date", "order_purchase_timestamp"])
items = pd.read_csv(PROC / "fact_order_items.csv", parse_dates=["purchase_date"])

# A SALE = not canceled/unavailable AND has at least one item
sales = orders[orders["is_valid_sale"] & (orders["n_items"] > 0)].copy()
sales_items = items[items["is_valid_sale"]].copy()

# ===== 1. HEADLINE KPIs =====
print("=== 1. HEADLINE KPIs (BRL) ===")
revenue = sales["product_revenue"].sum()
print(f"Total product revenue : {revenue:,.2f}")
print(f"Total freight         : {sales['freight_total'].sum():,.2f}")
print(f"Number of sales/orders: {len(sales):,}")
print(f"Units sold (items)    : {len(sales_items):,}")
print(f"Average order value   : {revenue / len(sales):,.2f}")
print(f"Median order value    : {sales['product_revenue'].median():,.2f}")
print(f"Unique customers      : {sales['customer_unique_id'].nunique():,}")
orders_per_cust = sales.groupby("customer_unique_id").size()
print(f"Repeat customers (2+ orders): {(orders_per_cust > 1).sum():,} "
      f"({(orders_per_cust > 1).mean():.1%} of customers)")
print(f"Date range of sales   : {sales['purchase_date'].min().date()} -> "
      f"{sales['purchase_date'].max().date()}")

# ===== 2. MONTHLY TREND (full months only: Jan 2017 - Jul 2018) =====
print("\n=== 2. MONTHLY TREND ===")
w = sales[(sales["purchase_date"] >= "2017-01-01") & (sales["purchase_date"] <= "2018-07-31")]
monthly = w.groupby(w["purchase_date"].dt.to_period("M")).agg(
    revenue=("product_revenue", "sum"), orders=("order_id", "count"))
monthly["revenue_growth_%"] = (monthly["revenue"].pct_change() * 100).round(1)
print(monthly.round(0).to_string())
monthly.index = monthly.index.to_timestamp()

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(monthly.index, monthly["revenue"], marker="o")
ax.set(title="Monthly product revenue (BRL), Jan 2017 - Jul 2018",
       xlabel="Month", ylabel="Revenue (BRL)")
save("01_monthly_revenue.png")

# ===== 3. DAILY SALES + 7-DAY AVERAGE (forecast window) =====
from config import FORECAST_START, FORECAST_END
d = sales[(sales["purchase_date"] >= FORECAST_START) & (sales["purchase_date"] <= FORECAST_END)]
daily = d.groupby("purchase_date")["product_revenue"].sum()
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(daily.index, daily.values, alpha=0.4, label="Daily revenue")
ax.plot(daily.index, daily.rolling(7).mean(), color="red", label="7-day average")
ax.set(title="Daily product revenue (BRL)", ylabel="Revenue (BRL)")
ax.legend()
save("02_daily_revenue.png")
print(f"\nTop 5 revenue days:\n{daily.sort_values(ascending=False).head(5).round(0).to_string()}")

# ===== 4. WEEKDAY AND HOUR PATTERNS =====
print("\n=== 4. WEEKDAY / HOUR ===")
order_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow = d.groupby(d["purchase_date"].dt.day_name())["product_revenue"].mean().reindex(order_days)
print("Average revenue per day, by weekday:\n", dow.round(0).to_string())
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=dow.index, y=dow.values, ax=ax, color="steelblue")
ax.set(title="Average daily revenue by weekday", xlabel="", ylabel="BRL")
plt.xticks(rotation=30)
save("03_weekday.png")

hour = sales.groupby(sales["order_purchase_timestamp"].dt.hour).size()
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=hour.index, y=hour.values, ax=ax, color="darkorange")
ax.set(title="Orders by hour of day", xlabel="Hour", ylabel="Orders")
save("04_hour.png")

# ===== 5. CATEGORIES =====
print("\n=== 5. CATEGORIES ===")
cat = sales_items.groupby("category").agg(revenue=("price", "sum"), units=("order_id", "count"))
cat["share_%"] = (cat["revenue"] / cat["revenue"].sum() * 100).round(1)
top10 = cat.sort_values("revenue", ascending=False).head(10)
bottom10 = cat.sort_values("revenue").head(10)
print("TOP 10 categories by revenue:\n", top10.round(1).to_string())
print("\nBOTTOM 10 categories by revenue:\n", bottom10.round(1).to_string())
print(f"\nTop 10 categories make up {top10['share_%'].sum():.1f}% of revenue")
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(x=top10["revenue"], y=top10.index, ax=ax, color="seagreen")
ax.set(title="Top 10 categories by revenue (BRL)", xlabel="Revenue", ylabel="")
save("05_top_categories.png")

# ===== 6. PRODUCTS =====
print("\n=== 6. PRODUCTS ===")
prod = sales_items.groupby("product_id").agg(revenue=("price", "sum"), units=("order_id", "count"))
print("Distinct products sold:", len(prod))
print("Top 10 products by revenue:\n", prod.sort_values("revenue", ascending=False).head(10).round(1).to_string())
print(f"Products sold only once: {(prod['units'] == 1).sum():,} "
      f"({(prod['units'] == 1).mean():.1%})")

# ===== 7. REGIONS (customer state) =====
print("\n=== 7. REGIONS ===")
state = sales.groupby("customer_state").agg(revenue=("product_revenue", "sum"),
                                            orders=("order_id", "count"))
state["share_%"] = (state["revenue"] / state["revenue"].sum() * 100).round(1)
top_states = state.sort_values("revenue", ascending=False).head(10)
print(top_states.round(1).to_string())
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.barplot(x=top_states.index, y=top_states["revenue"], ax=ax, color="purple")
ax.set(title="Top 10 states by revenue (BRL)", xlabel="State", ylabel="Revenue")
save("06_top_states.png")

# ===== 8. PAYMENTS =====
print("\n=== 8. PAYMENT TYPES (main method per order) ===")
print(sales["main_payment_type"].value_counts(normalize=True).mul(100).round(1).to_string())

# ===== 9. REVIEWS AND DELIVERY =====
print("\n=== 9. REVIEWS AND DELIVERY ===")
rv = sales[sales["review_score"].notna()]
print("Average review score:", round(rv["review_score"].mean(), 2))
print("Score distribution (%):\n", rv["review_score"].value_counts(normalize=True)
      .sort_index().mul(100).round(1).to_string())
dl = rv[rv["is_late"].notna()]
print("\nAverage score, on time (0) vs late (1):\n",
      dl.groupby("is_late")["review_score"].mean().round(2).to_string())
print("Late deliveries:", f"{dl['is_late'].mean():.1%}")
print("Average delivery days:", round(dl["delivery_days"].mean(), 1))
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x="review_score", data=rv, ax=ax, color="teal")
ax.set(title="Review score distribution")
save("07_reviews.png")
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x="is_late", y="review_score", data=dl, ax=ax, color="crimson")
ax.set(title="Average review: on time (0) vs late (1)")
save("08_late_vs_review.png")

# ===== 10. ORDER VALUE DISTRIBUTION =====
cap = sales["product_revenue"].quantile(0.99)
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(sales.loc[sales["product_revenue"] <= cap, "product_revenue"], bins=50, ax=ax)
ax.set(title=f"Order value distribution (orders up to the 99th percentile = {cap:,.0f} BRL)",
       xlabel="Order value (BRL)")
save("09_order_value.png")
print("\nDone. Charts are in reports/figures/")