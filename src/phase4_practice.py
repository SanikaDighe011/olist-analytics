from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

# ---------- 1. Variables and data types ----------
total_orders = 99441          # int (whole number)
avg_price = 120.65            # float (decimal)
marketplace = "Olist"         # str (text)
is_growing = True             # bool (True/False)
print("1.", type(total_orders).__name__, type(avg_price).__name__,
      type(marketplace).__name__, type(is_growing).__name__)

# ---------- 2. List, tuple, dictionary ----------
statuses = ["delivered", "shipped", "canceled"]    # list: changeable, ordered
statuses.append("invoiced")
window = ("2017-01-01", "2018-08-31")              # tuple: fixed (example window)
revenue_by_state = {"SP": 100.0, "RJ": 60.0}       # dict: key -> value
revenue_by_state["MG"] = 45.5
print("2.", statuses, window, revenue_by_state["SP"])

# ---------- 3. Conditionals and functions ----------
def classify_order(value):
    """Label an order by its value."""
    if value >= 500:
        return "high"
    elif value >= 100:
        return "medium"
    return "low"

print("3.", classify_order(750), classify_order(150), classify_order(40))

# ---------- 4. Loops ----------
for state, rev in revenue_by_state.items():
    print("4.", state, "->", rev)

# ---------- 5. File handling ----------
with open(RAW / "olist_orders_dataset.csv", encoding="utf-8") as f:
    header = f.readline().strip()
print("5. Header line:", header)

# ---------- 6. Exception handling ----------
try:
    pd.read_csv(RAW / "this_file_does_not_exist.csv")
except FileNotFoundError:
    print("6. Caught the error: file not found (program did not crash)")

# ---------- 7. NumPy: fast math on arrays ----------
prices = np.array([58.9, 239.9, 199.0, 12.5])
print("7. mean:", prices.mean().round(2), "max:", prices.max(),
      "total:", prices.sum().round(2))

# ---------- 8. Pandas: Series and DataFrame ----------
items = pd.read_csv(RAW / "olist_order_items_dataset.csv")   # DataFrame = table
price = items["price"]                                        # Series = one column
print("8a. Type of items:", type(items).__name__, "| type of price:", type(price).__name__)
print("8b. Price summary:\n", price.describe().round(2))

expensive = items[items["price"] > 1000]                      # filter rows
print("8c. Items priced above 1000:", len(expensive))

order_totals = items.groupby("order_id")["price"].sum()       # one row per order
print("8d. Five biggest orders by item price:\n",
      order_totals.sort_values(ascending=False).head(5))
print("8e. Orders:", len(order_totals), "| items:", len(items))