"""Phase 7: build the SQLite star-schema database from the cleaned CSVs."""
import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
DB_PATH = ROOT / "database" / "olist.db"
DB_PATH.parent.mkdir(exist_ok=True)

SCHEMA = """
CREATE TABLE dim_date (
    date TEXT PRIMARY KEY, year INTEGER NOT NULL, month INTEGER NOT NULL,
    day INTEGER NOT NULL, month_name TEXT NOT NULL, year_month TEXT NOT NULL,
    weekday_num INTEGER NOT NULL, weekday_name TEXT NOT NULL, is_weekend INTEGER NOT NULL
);
CREATE TABLE dim_customers (
    customer_id TEXT PRIMARY KEY, customer_unique_id TEXT NOT NULL,
    customer_zip_code_prefix INTEGER, customer_city TEXT, customer_state TEXT
);
CREATE TABLE dim_sellers (
    seller_id TEXT PRIMARY KEY, seller_zip_code_prefix INTEGER,
    seller_city TEXT, seller_state TEXT
);
CREATE TABLE dim_products (
    product_id TEXT PRIMARY KEY, category TEXT NOT NULL, product_category_name TEXT,
    product_name_length REAL, product_description_length REAL, product_photos_qty REAL,
    product_weight_g REAL, product_length_cm REAL, product_height_cm REAL, product_width_cm REAL
);
CREATE TABLE dim_geo_zip (
    zip_code_prefix INTEGER PRIMARY KEY, lat REAL, lng REAL, city TEXT, state TEXT
);
CREATE TABLE fact_orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    order_status TEXT NOT NULL,
    purchase_date TEXT NOT NULL,
    order_purchase_timestamp TEXT, order_approved_at TEXT,
    order_delivered_carrier_date TEXT, order_delivered_customer_date TEXT,
    order_estimated_delivery_date TEXT,
    delivery_days REAL, estimated_days REAL, delay_days REAL, is_late INTEGER,
    is_valid_sale INTEGER NOT NULL,
    n_items INTEGER, n_products INTEGER, n_sellers INTEGER,
    product_revenue REAL, freight_total REAL, valid_revenue REAL,
    payment_total REAL, payment_rows INTEGER, max_installments INTEGER, main_payment_type TEXT,
    review_score INTEGER, has_comment INTEGER, is_high_value_outlier INTEGER,
    FOREIGN KEY (customer_id) REFERENCES dim_customers(customer_id),
    FOREIGN KEY (purchase_date) REFERENCES dim_date(date)
);
CREATE TABLE fact_order_items (
    order_id TEXT NOT NULL, order_item_id INTEGER NOT NULL,
    product_id TEXT NOT NULL, seller_id TEXT NOT NULL,
    shipping_limit_date TEXT, price REAL NOT NULL, freight_value REAL NOT NULL,
    PRIMARY KEY (order_id, order_item_id),
    FOREIGN KEY (order_id) REFERENCES fact_orders(order_id),
    FOREIGN KEY (product_id) REFERENCES dim_products(product_id),
    FOREIGN KEY (seller_id) REFERENCES dim_sellers(seller_id)
);
CREATE INDEX idx_orders_date ON fact_orders(purchase_date);
CREATE INDEX idx_orders_customer ON fact_orders(customer_id);
CREATE INDEX idx_items_product ON fact_order_items(product_id);
CREATE INDEX idx_items_seller ON fact_order_items(seller_id);
CREATE INDEX idx_customers_unique ON dim_customers(customer_unique_id);
CREATE INDEX idx_customers_state ON dim_customers(customer_state);
"""


def make_dim_date():
    d = pd.Series(pd.date_range("2016-09-01", "2018-10-31"))
    return pd.DataFrame({
        "date": d.dt.strftime("%Y-%m-%d"),
        "year": d.dt.year,
        "month": d.dt.month,
        "day": d.dt.day,
        "month_name": d.dt.month_name(),
        "year_month": d.dt.strftime("%Y-%m"),
        "weekday_num": d.dt.dayofweek + 1,          # Monday = 1
        "weekday_name": d.dt.day_name(),
        "is_weekend": (d.dt.dayofweek >= 5).astype(int),
    })


def to_flag(series):
    """True/False/blank -> 1/0/None (SQLite has no real boolean type)."""
    return series.map(lambda v: None if pd.isna(v) else int(str(v) == "True"))


def load(conn, table, df):
    cols = list(df.columns)
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
    rows = [tuple(None if pd.isna(v) else v for v in r)
            for r in df.itertuples(index=False, name=None)]
    conn.executemany(sql, rows)
    print(f"   {table:17s} {len(rows):>9,} rows")


def check(label, ok):
    print(("PASS  " if ok else "FAIL  ") + label)


def main():
    if DB_PATH.exists():
        DB_PATH.unlink()                      # rebuild from scratch each time
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")  # make SQLite ENFORCE foreign keys
    conn.executescript(SCHEMA)

    customers = pd.read_csv(PROC / "dim_customers.csv")
    sellers = pd.read_csv(PROC / "dim_sellers.csv")
    products = pd.read_csv(PROC / "dim_products.csv")
    geo = pd.read_csv(PROC / "dim_geo_zip.csv").rename(
        columns={"geolocation_zip_code_prefix": "zip_code_prefix"})
    orders = pd.read_csv(PROC / "fact_orders.csv")
    items = pd.read_csv(PROC / "fact_order_items.csv")

    for col in ["is_valid_sale", "has_comment", "is_high_value_outlier"]:
        orders[col] = to_flag(orders[col])

    order_cols = ["order_id", "customer_id", "order_status", "purchase_date",
                  "order_purchase_timestamp", "order_approved_at",
                  "order_delivered_carrier_date", "order_delivered_customer_date",
                  "order_estimated_delivery_date", "delivery_days", "estimated_days",
                  "delay_days", "is_late", "is_valid_sale", "n_items", "n_products",
                  "n_sellers", "product_revenue", "freight_total", "valid_revenue",
                  "payment_total", "payment_rows", "max_installments", "main_payment_type",
                  "review_score", "has_comment", "is_high_value_outlier"]
    item_cols = ["order_id", "order_item_id", "product_id", "seller_id",
                 "shipping_limit_date", "price", "freight_value"]
    product_cols = ["product_id", "category", "product_category_name",
                    "product_name_length", "product_description_length", "product_photos_qty",
                    "product_weight_g", "product_length_cm", "product_height_cm",
                    "product_width_cm"]

    print("Loading (parents first, so foreign keys are satisfied):")
    load(conn, "dim_date", make_dim_date())
    load(conn, "dim_customers", customers[["customer_id", "customer_unique_id",
                                           "customer_zip_code_prefix", "customer_city",
                                           "customer_state"]])
    load(conn, "dim_sellers", sellers)
    load(conn, "dim_products", products[product_cols])
    load(conn, "dim_geo_zip", geo[["zip_code_prefix", "lat", "lng", "city", "state"]])
    load(conn, "fact_orders", orders[order_cols])
    load(conn, "fact_order_items", items[item_cols])
    conn.commit()

    print("\n=== VALIDATION ===")
    q = lambda sql: conn.execute(sql).fetchone()[0]
    check("fact_orders row count matches CSV", q("SELECT COUNT(*) FROM fact_orders") == len(orders))
    check("fact_order_items row count matches CSV",
          q("SELECT COUNT(*) FROM fact_order_items") == len(items))
    check("dim_customers row count matches CSV",
          q("SELECT COUNT(*) FROM dim_customers") == len(customers))
    check("dim_products row count matches CSV",
          q("SELECT COUNT(*) FROM dim_products") == len(products))
    check("no foreign key violations", conn.execute("PRAGMA foreign_key_check").fetchall() == [])
    check("SQLite integrity_check is ok", q("PRAGMA integrity_check") == "ok")
    pandas_rev = orders["valid_revenue"].sum()
    check("valid revenue in SQL == pandas (within 0.01 BRL)",
          abs(q("SELECT SUM(valid_revenue) FROM fact_orders") - pandas_rev) < 0.01)
    check("item prices joined to orders == valid revenue (join did not duplicate money)",
          abs(q("""SELECT SUM(i.price) FROM fact_order_items i
                   JOIN fact_orders o ON o.order_id = i.order_id
                   WHERE o.is_valid_sale = 1""") - pandas_rev) < 0.01)
    check("joining items to orders keeps every item exactly once",
          q("""SELECT COUNT(*) FROM fact_order_items i
               JOIN fact_orders o ON o.order_id = i.order_id""") == len(items))
    print(f"\nSQL valid revenue: {q('SELECT SUM(valid_revenue) FROM fact_orders'):,.2f} BRL")
    print(f"Database file: {DB_PATH}")
    conn.close()


if __name__ == "__main__":
    main()