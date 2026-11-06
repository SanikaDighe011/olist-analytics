"""Phase 8b: compare forecasting models with a chronological split."""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

TEST_DAYS = 90                      # the last 90 days are the unseen test period
HORIZONS = [7, 30, 90]              # we score the first 7, 30 and all 90 test days
BLACK_FRIDAY = pd.Timestamp("2017-11-24")
STARTS = {"train from 2017-01-05": None,           # all available history
          "train from 2017-07-01": "2017-07-01"}   # skip the early ramp-up

daily = (pd.read_csv(PROC / "daily_sales.csv", parse_dates=["date"])
           .set_index("date")["revenue"].asfreq("D"))
train_full = daily.iloc[:-TEST_DAYS]
test = daily.iloc[-TEST_DAYS:]
assert train_full.index.max() < test.index.min()       # past strictly before future

print(f"Train: {train_full.index.min().date()} -> {train_full.index.max().date()} "
      f"({len(train_full)} days)")
print(f"Test : {test.index.min().date()} -> {test.index.max().date()} ({len(test)} days)")
print(f"Average daily revenue in the test period: {test.mean():,.0f} BRL\n")


def make_features(idx, origin):
    """Calendar features only, so we can forecast any horizon without peeking."""
    X = pd.DataFrame(index=idx)
    X["day_number"] = (idx - origin).days            # a straight-line trend
    for k in range(1, 7):                            # weekday dummies (Monday = 0 is the base)
        X[f"dow_{k}"] = (idx.day_of_week == k).astype(int)
    X["black_friday"] = (idx == BLACK_FRIDAY).astype(int)   # one-off spike flag
    return X


def ml_forecast(model, train, test_idx):
    origin = train.index[0]
    model.fit(make_features(train.index, origin), train.values)
    return model.predict(make_features(test_idx, origin))


def holt_winters(train, test_idx):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(train, trend="add", damped_trend=True,
                                   seasonal="add", seasonal_periods=7).fit()
    return fit.forecast(len(test_idx)).values


def score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if (y_true > 0).all() else np.nan
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAPE_%": mape,
            "R2": r2_score(y_true, y_pred)}


# ---------- collect forecasts for the whole 90-day test period ----------
n = len(test)
preds = {}

# baselines (they only look at the most recent days)
preds["Baseline: last day repeated"] = np.repeat(train_full.iloc[-1], n)
preds["Baseline: last 7 days repeated"] = np.tile(train_full.iloc[-7:].values, n // 7 + 1)[:n]
preds["Baseline: 28-day moving average"] = np.repeat(train_full.iloc[-28:].mean(), n)
recent = train_full.iloc[-56:]
wd_mean = recent.groupby(recent.index.day_of_week).mean()
preds["Baseline: weekday average (last 8 wks)"] = test.index.day_of_week.map(wd_mean).values

# trained models, once per choice of training start date
for label, start in STARTS.items():
    tr = train_full if start is None else train_full.loc[start:]
    preds[f"Linear Regression | {label}"] = ml_forecast(LinearRegression(), tr, test.index)
    preds[f"Random Forest | {label}"] = ml_forecast(
        RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=42), tr, test.index)
    preds[f"Gradient Boosting | {label}"] = ml_forecast(
        GradientBoostingRegressor(random_state=42), tr, test.index)
    preds[f"Holt-Winters | {label}"] = holt_winters(tr, test.index)

# revenue cannot be negative
preds = {k: np.clip(v, 0, None) for k, v in preds.items()}

# ---------- score at 7, 30 and 90 days ----------
rows = []
for name, p in preds.items():
    for h in HORIZONS:
        rows.append({"model": name, "horizon_days": h, **score(test.iloc[:h], p[:h])})
results = pd.DataFrame(rows)

pd.options.display.float_format = "{:,.2f}".format
for h in HORIZONS:
    print("=" * 100)
    print(f"FIRST {h} DAYS OF THE TEST PERIOD (lower MAE / RMSE / MAPE is better)")
    view = results[results["horizon_days"] == h].drop(columns="horizon_days")
    print(view.sort_values("MAE").to_string(index=False))

results.to_csv(REPORTS / "model_comparison.csv", index=False)
pd.DataFrame(preds, index=test.index).assign(actual=test.values).to_csv(
    REPORTS / "test_predictions.csv")
print("\nSaved reports/model_comparison.csv and reports/test_predictions.csv")