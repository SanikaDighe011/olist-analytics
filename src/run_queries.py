"""Phase 7: run practice SQL queries from Python."""
import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "database" / "olist.db"

QUERIES = {
"1. SELECT + WHERE + ORDER BY + LIMIT: 5 biggest valid orders": """
    SELECT order_id, order_status, n_items, product_revenue
    FROM fact_orders
    WHERE is_valid_sale = 1
    ORDER BY product_revenue DESC
    LIMIT 5;
""",
"2. Aggregates: KPIs (expect 98,199 orders | 13,494,400.74 | AOV 137.42)": """
    SELECT COUNT(*) AS orders,
           ROUND(SUM(valid_revenue), 2) AS revenue,
           ROUND(AVG(valid_revenue), 2) AS avg_order_value
    FROM fact_orders
    WHERE is_valid_sale = 1 AND n_items > 0;
""",
"3. GROUP BY + JOIN dim_date: monthly revenue (expect 2017-11: 7,421 orders, ~1,003,862)": """
    SELECT d.year_month, COUNT(*) AS orders, ROUND(SUM(o.valid_revenue), 0) AS revenue
    FROM fact_orders o
    JOIN dim_date d ON d.date = o.purchase_date
    WHERE o.is_valid_sale = 1 AND o.n_items > 0
    GROUP BY d.year_month
    ORDER BY d.year_month;
""",
"4. Three-table JOIN: top categories (expect health_beauty ~1,255,695.1 | 9,634 units)": """
    SELECT p.category, COUNT(*) AS units, ROUND(SUM(i.price), 1) AS revenue
    FROM fact_order_items i
    JOIN dim_products p ON p.product_id = i.product_id
    JOIN fact_orders o ON o.order_id = i.order_id
    WHERE o.is_valid_sale = 1
    GROUP BY p.category
    ORDER BY revenue DESC
    LIMIT 10;
""",
"5. JOIN dim_customers: revenue by state (expect SP ~5,163,867.2 | 41,125 orders)": """
    SELECT c.customer_state, COUNT(*) AS orders, ROUND(SUM(o.valid_revenue), 1) AS revenue
    FROM fact_orders o
    JOIN dim_customers c ON c.customer_id = o.customer_id
    WHERE o.is_valid_sale = 1 AND o.n_items > 0
    GROUP BY c.customer_state
    ORDER BY revenue DESC
    LIMIT 10;
""",
"6. Subquery: valid orders worth more than the average order": """
    SELECT COUNT(*) AS orders_above_average
    FROM fact_orders
    WHERE is_valid_sale = 1
      AND valid_revenue > (SELECT AVG(valid_revenue)
                           FROM fact_orders
                           WHERE is_valid_sale = 1 AND n_items > 0);
""",
"7. GROUP BY + HAVING in a subquery: repeat customers (expect 2,887)": """
    SELECT COUNT(*) AS repeat_customers
    FROM (SELECT c.customer_unique_id
          FROM fact_orders o
          JOIN dim_customers c ON c.customer_id = o.customer_id
          WHERE o.is_valid_sale = 1 AND o.n_items > 0
          GROUP BY c.customer_unique_id
          HAVING COUNT(*) > 1);
""",
"8. Delivery vs reviews (expect 4.29 on time | 2.57 late)": """
    SELECT is_late, COUNT(*) AS orders, ROUND(AVG(review_score), 2) AS avg_review
    FROM fact_orders
    WHERE is_valid_sale = 1 AND n_items > 0
      AND review_score IS NOT NULL AND is_late IS NOT NULL
    GROUP BY is_late;
""",
"9. Average DAILY revenue by weekday (expect Monday ~25,886 | Saturday ~17,385)": """
    SELECT d.weekday_name, ROUND(AVG(daily.revenue), 0) AS avg_daily_revenue
    FROM (SELECT purchase_date, SUM(valid_revenue) AS revenue
          FROM fact_orders
          WHERE is_valid_sale = 1 AND n_items > 0
            AND purchase_date BETWEEN '2017-01-05' AND '2018-08-20'
          GROUP BY purchase_date) AS daily
    JOIN dim_date d ON d.date = daily.purchase_date
    GROUP BY d.weekday_num, d.weekday_name
    ORDER BY d.weekday_num;
""",
"10. The join trap in SQL: payments counted once vs after joining items": """
    SELECT
      (SELECT ROUND(SUM(payment_total), 2) FROM fact_orders) AS correct_payments,
      (SELECT ROUND(SUM(o.payment_total), 2)
       FROM fact_orders o JOIN fact_order_items i ON i.order_id = o.order_id) AS after_join_with_items;
""",
}

conn = sqlite3.connect(DB_PATH)
for title, sql in QUERIES.items():
    print("=" * 78)
    print(title)
    print(pd.read_sql_query(sql, conn).to_string(index=False))
conn.close()