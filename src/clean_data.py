"""Phase 5: clean the 9 raw Olist files and build analysis-ready tables.
Raw files are only READ. Results go to data/processed/."""
from pathlib import Path
import numpy as np
import pandas as pd
from config import NOT_A_SALE

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def read(name):
    return pd.read_csv(RAW / name)


def check(label, ok):
    print(("PASS  " if ok else "FAIL  ") + label)


# ---------- ORDERS ----------
def clean_orders():
    df = read("olist_orders_dataset.csv")
    date_cols = ["order_purchase_timestamp", "order_approved_at",
                 "order_delivered_carrier_date", "order_delivered_customer_date",
                 "order_estimated_delivery_date"]
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")   # text -> real dates

    day = pd.Timedelta(days=1)
    df["purchase_date"] = df["order_purchase_timestamp"].dt.normalize()
    df["delivery_days"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]) / day
    df["estimated_days"] = (df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]) / day
    df["delay_days"] = (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]) / day
    # 1.0 = late, 0.0 = on time, blank = not delivered yet
    df["is_late"] = (df["delay_days"] > 0).astype(float).where(df["delay_days"].notna())
    df["is_valid_sale"] = ~df["order_status"].isin(NOT_A_SALE)

    print("   delivered but no delivery date:",
          int(((df["order_status"] == "delivered") & df["order_delivered_customer_date"].isna()).sum()))
    print("   delivery date before purchase date:", int((df["delivery_days"] < 0).sum()))
    return df


# ---------- ITEMS ----------
def clean_items():
    df = read("olist_order_items_dataset.csv")
    df["shipping_limit_date"] = pd.to_datetime(df["shipping_limit_date"], errors="coerce")
    print("   items with price <= 0:", int((df["price"] <= 0).sum()))
    print("   items with negative freight:", int((df["freight_value"] < 0).sum()))
    return df


# ---------- PRODUCTS ----------
def clean_products():
    df = read("olist_products_dataset.csv")
    # fix the misspelled column names from the original data
    df = df.rename(columns={"product_name_lenght": "product_name_length",
                            "product_description_lenght": "product_description_length"})
    tr = read("product_category_name_translation.csv")
    # OUR OWN additions (not from Olist) for 2 categories missing in their translation file
    extra = pd.DataFrame({
        "product_category_name": ["pc_gamer", "portateis_cozinha_e_preparadores_de_alimentos"],
        "product_category_name_english": ["pc_gamer", "portable_kitchen_and_food_preparers"],
    })
    tr = pd.concat([tr, extra], ignore_index=True)
    n_before = len(df)
    df = df.merge(tr, on="product_category_name", how="left", validate="many_to_one")
    assert len(df) == n_before, "translation join changed the number of products!"
    df["category"] = df["product_category_name_english"].fillna("unknown")
    print("   products with unknown category:", int((df["category"] == "unknown").sum()))
    return df


# ---------- CUSTOMERS ----------
def clean_customers():
    df = read("olist_customers_dataset.csv")
    df["customer_state"] = df["customer_state"].str.strip().str.upper()
    df["customer_city"] = df["customer_city"].str.strip().str.lower()
    return df


# ---------- PAYMENTS (collapsed to ONE row per order) ----------
def clean_payments():
    df = read("olist_order_payments_dataset.csv")
    print("   payment rows with value 0:", int((df["payment_value"] == 0).sum()))
    print("   payment types:", df["payment_type"].value_counts().to_dict())
    agg = df.groupby("order_id").agg(
        payment_total=("payment_value", "sum"),
        payment_rows=("payment_sequential", "count"),
        max_installments=("payment_installments", "max"),
    ).reset_index()
    # the payment method that paid the largest share of the order
    main = (df.sort_values(["order_id", "payment_value"], ascending=[True, False])
              .drop_duplicates("order_id")[["order_id", "payment_type"]]
              .rename(columns={"payment_type": "main_payment_type"}))
    return agg.merge(main, on="order_id", validate="one_to_one")


# ---------- REVIEWS (collapsed to ONE row per order) ----------
def clean_reviews():
    df = read("olist_order_reviews_dataset.csv")
    df["review_creation_date"] = pd.to_datetime(df["review_creation_date"], errors="coerce")
    df["review_answer_timestamp"] = pd.to_datetime(df["review_answer_timestamp"], errors="coerce")
    df["has_comment"] = df["review_comment_message"].notna()
    print("   review scores outside 1-5:", int((~df["review_score"].between(1, 5)).sum()))
    # RULE: if an order has several reviews, keep the most recent one
    latest = (df.sort_values(["order_id", "review_answer_timestamp"], ascending=[True, False])
                .drop_duplicates("order_id"))
    return latest[["order_id", "review_score", "has_comment",
                   "review_creation_date", "review_answer_timestamp"]]


# ---------- SELLERS ----------
def clean_sellers():
    df = read("olist_sellers_dataset.csv")
    df["seller_state"] = df["seller_state"].str.strip().str.upper()
    df["seller_city"] = df["seller_city"].str.strip().str.lower()
    return df


# ---------- GEOLOCATION (collapsed to ONE row per zip prefix) ----------
def clean_geo():
    df = read("olist_geolocation_dataset.csv")
    n0 = len(df)
    df = df.drop_duplicates()
    # rough bounding box around Brazil; points outside are bad coordinates
    inside = df["geolocation_lat"].between(-34, 6) & df["geolocation_lng"].between(-74, -34)
    print(f"   rows: {n0:,} -> {len(df):,} after dropping exact duplicates")
    print(f"   coordinates outside Brazil's bounding box (removed): {int((~inside).sum()):,}")
    df = df[inside]
    return df.groupby("geolocation_zip_code_prefix").agg(
        lat=("geolocation_lat", "median"),      # median resists stray points
        lng=("geolocation_lng", "median"),
        city=("geolocation_city", lambda s: s.mode().iat[0]),
        state=("geolocation_state", lambda s: s.mode().iat[0]),
    ).reset_index()

def same_money(a, b):
    return abs(a - b) < 0.01


def main():
    print("Orders:");      orders = clean_orders()
    print("Items:");       items = clean_items()
    print("Products:");    products = clean_products()
    print("Customers:");   customers = clean_customers()
    print("Payments:");    pay_order = clean_payments()
    print("Reviews:");     reviews_order = clean_reviews()
    print("Sellers:");     sellers = clean_sellers()
    print("Geolocation:"); geo_zip = clean_geo()

    # ----- order-level aggregation of items (BEFORE any join) -----
    item_agg = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_products=("product_id", "nunique"),
        n_sellers=("seller_id", "nunique"),
        product_revenue=("price", "sum"),
        freight_total=("freight_value", "sum"),
    ).reset_index()

    # ----- ONE ROW PER ORDER table -----
    fact = (orders
            .merge(customers[["customer_id", "customer_unique_id", "customer_city",
                              "customer_state", "customer_zip_code_prefix"]],
                   on="customer_id", how="left", validate="one_to_one")
            .merge(item_agg, on="order_id", how="left", validate="one_to_one")
            .merge(pay_order, on="order_id", how="left", validate="one_to_one")
            .merge(reviews_order, on="order_id", how="left", validate="one_to_one"))
    for col in ["n_items", "n_products", "n_sellers"]:
        fact[col] = fact[col].fillna(0).astype(int)
    for col in ["product_revenue", "freight_total", "payment_total"]:
        fact[col] = fact[col].fillna(0.0)
    # revenue that counts as a real sale (canceled/unavailable orders count as 0)
    fact["valid_revenue"] = fact["product_revenue"].where(fact["is_valid_sale"], 0.0)

    # outlier FLAG using the IQR rule (flagged, NOT removed)
    v = fact.loc[fact["n_items"] > 0, "product_revenue"]
    q1, q3 = v.quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)
    fact["is_high_value_outlier"] = fact["product_revenue"] > upper
    print(f"\nIQR upper fence: {upper:,.2f} BRL | orders above it: "
          f"{int(fact['is_high_value_outlier'].sum()):,} (flagged, NOT removed)")

    # ----- ONE ROW PER ORDER ITEM table (for product/category analysis) -----
    fact_items = (items
                  .merge(products[["product_id", "category"]], on="product_id",
                         how="left", validate="many_to_one")
                  .merge(fact[["order_id", "purchase_date", "order_status", "is_valid_sale",
                               "customer_unique_id", "customer_state", "customer_city"]],
                         on="order_id", how="left", validate="many_to_one"))
    fact_items["category"] = fact_items["category"].fillna("unknown")

    # ----- VALIDATION: prove we did not create or lose money -----
    print("\n=== VALIDATION ===")
    raw_pay_total = read("olist_order_payments_dataset.csv")["payment_value"].sum()
    check("fact_orders: one row per order, no duplicates",
          len(fact) == len(orders) and fact["order_id"].is_unique)
    check("fact_orders: product revenue == sum of raw item prices",
          same_money(fact["product_revenue"].sum(), items["price"].sum()))
    check("fact_orders: payment total == sum of raw payments",
          same_money(fact["payment_total"].sum(), raw_pay_total))
    check("fact_items: same row count as raw items", len(fact_items) == len(items))
    check("fact_items: price total unchanged after joins",
          same_money(fact_items["price"].sum(), items["price"].sum()))
    check("fact_items: no duplicate (order_id, order_item_id)",
          not fact_items.duplicated(["order_id", "order_item_id"]).any())
    check("no missing purchase dates", fact["purchase_date"].notna().all())
    cov = customers["customer_zip_code_prefix"].isin(geo_zip["geolocation_zip_code_prefix"]).mean()
    print(f"INFO  customers whose zip prefix exists in geo table: {cov:.1%}")

    print("\n=== SUMMARY (BRL) ===")
    print(f"Orders: {len(fact):,} | valid sales: {int(fact['is_valid_sale'].sum()):,}")
    print(f"Product revenue, valid sales only: {fact['valid_revenue'].sum():,.2f}")
    print(f"Freight (all orders):              {fact['freight_total'].sum():,.2f}")
    print(f"Raw payments total:                {raw_pay_total:,.2f}")

    # ----- SAVE -----
    fact.to_csv(OUT / "fact_orders.csv", index=False)
    fact_items.to_csv(OUT / "fact_order_items.csv", index=False)
    products.to_csv(OUT / "dim_products.csv", index=False)
    customers.to_csv(OUT / "dim_customers.csv", index=False)
    sellers.to_csv(OUT / "dim_sellers.csv", index=False)
    geo_zip.to_csv(OUT / "dim_geo_zip.csv", index=False)
    print("\nSaved 6 files to data/processed/")


if __name__ == "__main__":
    main()