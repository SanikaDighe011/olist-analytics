
import sys
import pandas, numpy, matplotlib, seaborn, sklearn, statsmodels, sqlalchemy, streamlit
import sqlite3

print("Python:", sys.version.split()[0])

for lib in [pandas, numpy, matplotlib, seaborn, sklearn, statsmodels, sqlalchemy, streamlit]:
    print(f"{lib.__name__:12s} {lib.__version__}")

print("SQLite engine:", sqlite3.sqlite_version)
print("Setup OK")
