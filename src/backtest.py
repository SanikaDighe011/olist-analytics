"""Phase 8c: rolling backtest. Same models, six different 30-day test windows."""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

H = 30                 # forecast length
N_WINDOWS = 6          # number of different test windows
BLACK_FRIDAY = pd.Timestamp("2017-11-24")

daily = (pd.read_csv(PROC / "daily_sales.csv", parse_dates=["date"])
           .set_index("date")["revenue"].asfreq("D"))


def make_features(idx, origin):
    X = pd.DataFrame(index=idx)
    X["day_number"] = (idx - origin).days
    for k in range(1, 7):
        X[f"dow_{k}"] = (idx.day_of_week == k).astype(int)
    X["black_friday"] = (idx == BLACK_FRIDAY).astype(int)
    return X


def last_day(train, idx):
    return np.repeat(train.iloc[-1], len(idx))


def ma28(train, idx):
    return np.repeat(train.iloc[-28:].mean(), len(idx))


def weekday_avg(train, idx):
    recent = train.iloc[-56:]
    m = recent.groupby(recent.index.day_of_week).mean()
    return idx.day_of_week.map(m).values


def grad_boost(train, idx):
    model = GradientBoostingRegressor(random_state=42)
    model.fit(make_features(train.index, train.index[0]), train.values)
    return model.predict(make_features(idx, train.index[0]))


def holt_winters(train, idx):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(train, trend="add", damped_trend=True,
                                   seasonal="add", seasonal_periods=7).fit()
    return fit.forecast(len(idx)).values


MODELS = {
    "Baseline: last day repeated": last_day,
    "Baseline: 28-day moving average": ma28,
    "Baseline: weekday average (8 wks)": weekday_avg,
    "Gradient Boosting": grad_boost,
    "Holt-Winters": holt_winters,
}

rows = []
print("Test windows (each model trains only on days BEFORE its window):")
for w in range(N_WINDOWS, 0, -1):
    end = len(daily) - H * (w - 1)
    test, train = daily.iloc[end - H:end], daily.iloc[:end - H]
    assert train.index.max() < test.index.min()
    label = f"W{N_WINDOWS - w + 1}"
    print(f"  {label}: {test.index.min().date()} -> {test.index.max().date()} "
          f"(train days: {len(train)}, avg daily revenue: {test.mean():,.0f})")
    for name, fn in MODELS.items():
        p = np.clip(fn(train, test.index), 0, None)
        rows.append({"model": name, "window": label,
                     "MAE": np.mean(np.abs(test.values - p)),
                     "MAPE_%": np.mean(np.abs((test.values - p) / test.values)) * 100})

res = pd.DataFrame(rows)
pd.options.display.float_format = "{:,.0f}".format
mae = res.pivot(index="model", columns="window", values="MAE")
mae["AVERAGE"] = mae.mean(axis=1)
print("\n=== MAE (BRL per day) in each window; lower is better ===")
print(mae.sort_values("AVERAGE").to_string())

mape = res.pivot(index="model", columns="window", values="MAPE_%")
mape["AVERAGE"] = mape.mean(axis=1)
print("\n=== MAPE (%) in each window ===")
print(mape.sort_values("AVERAGE").to_string())

wins = mae.drop(columns="AVERAGE").idxmin().value_counts()
print("\nBest model per window (by MAE):")
print(wins.to_string())

res.to_csv(REPORTS / "backtest_results.csv", index=False)
print("\nSaved reports/backtest_results.csv")
