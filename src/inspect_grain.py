from pathlib import Path
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
def load(f): return pd.read_csv(RAW / f)

o = load("olist_orders_dataset.csv")
i = load("olist_order_items_dataset.csv")
c = load("olist_customers_dataset.csv")
p = load("olist_order_payments_dataset.csv")
r = load("olist_order_reviews_dataset.csv")
pr = load("olist_products_dataset.csv")
tr = load("product_category_name_translation.csv")

print("=== GRAIN CHECKS ===")
print("Orders with more than 1 item:     ", (i.groupby("order_id").size() > 1).sum())
print("Orders with more than 1 payment:  ", (p.groupby("order_id").size() > 1).sum())
print("Orders with more than 1 review:   ", (r.groupby("order_id").size() > 1).sum())
print("Unique customer_id:               ", c["customer_id"].nunique())
print("Unique customer_unique_id:        ", c["customer_unique_id"].nunique())

print("\n=== ORDERS WITHOUT ITEMS / PAYMENTS ===")
print("Orders with no item rows:   ", len(set(o["order_id"]) - set(i["order_id"])))
print("Orders with no payment rows:", len(set(o["order_id"]) - set(p["order_id"])))

print("\n=== PRODUCT CATEGORIES ===")
print("Categories in products but not in translation:",
      sorted(set(pr["product_category_name"].dropna()) - set(tr["product_category_name"])))

print("\n=== THE JOIN TRAP, WITH YOUR REAL NUMBERS ===")
true_total = p["payment_value"].sum()
naive_total = i.merge(p, on="order_id")["payment_value"].sum()
print(f"Correct total of payments:         {true_total:,.2f}")
print(f"Total after naive items+payments:  {naive_total:,.2f}")
print(f"Overcount:                         {naive_total / true_total - 1:.1%}")

print("\n=== RECONCILIATION: items(price+freight) vs payments, per order ===")
item_tot = i.groupby("order_id").agg(price=("price", "sum"), freight=("freight_value", "sum"))
item_tot["items_plus_freight"] = item_tot["price"] + item_tot["freight"]
pay_tot = p.groupby("order_id")["payment_value"].sum()
both = item_tot.join(pay_tot, how="inner")
diff = (both["payment_value"] - both["items_plus_freight"]).abs()
print("Orders compared:                 ", len(both))
print("Orders matching within 0.01:     ", (diff <= 0.01).sum())
print("Orders with a bigger gap:        ", (diff > 0.01).sum())

print("\n=== ORDER STATUS ===")
print(o["order_status"].value_counts().to_string())

o["purchase_ts"] = pd.to_datetime(o["order_purchase_timestamp"], errors="coerce")
print("\nPurchase range:", o["purchase_ts"].min(), "->", o["purchase_ts"].max())
print("Unparseable dates:", o["purchase_ts"].isna().sum())

print("\n=== ORDERS PER MONTH ===")
print(o.groupby(o["purchase_ts"].dt.to_period("M")).size().to_string())