
from pathlib import Path
import pandas as pd

RAW = Path("data/raw")

FILES = {
    "orders":     "olist_orders_dataset.csv",
    "items":      "olist_order_items_dataset.csv",
    "customers":  "olist_customers_dataset.csv",
    "products":   "olist_products_dataset.csv",
    "payments":   "olist_order_payments_dataset.csv",
    "reviews":    "olist_order_reviews_dataset.csv",
    "sellers":    "olist_sellers_dataset.csv",
    "geolocation":"olist_geolocation_dataset.csv",
    "translation":"product_category_name_translation.csv",
}

# Which column(s) we EXPECT to be unique per row
EXPECTED_KEY = {
    "orders":     ["order_id"],
    "items":      ["order_id", "order_item_id"],
    "customers":  ["customer_id"],
    "products":   ["product_id"],
    "payments":   ["order_id", "payment_sequential"],
    "reviews":    ["review_id"],
    "sellers":    ["seller_id"],
    "geolocation":["geolocation_zip_code_prefix"],
    "translation":["product_category_name"],
}

for name, fname in FILES.items():
    df = pd.read_csv(RAW / fname)

    print("=" * 70)
    print(f"{name.upper()}  ({fname})")
    print(f"Rows: {len(df):,}   Columns: {df.shape[1]}")

    print("\nColumns and types:")
    print(df.dtypes.to_string())

    missing = df.isna().sum()
    missing = missing[missing > 0]

    print("\nMissing values:")
    print(missing.to_string() if len(missing) else "  none")

    print(f"\nFully duplicated rows: {df.duplicated().sum():,}")

    key = EXPECTED_KEY[name]
    dup_keys = df.duplicated(subset=key).sum()

    print(f"Rows repeating the expected key {key}: {dup_keys:,}")

    print("\nFirst 3 rows:")
    print(df.head(3).to_string())
    print()
