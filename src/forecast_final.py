"""Phase 8d: fit the final model on all data, forecast 90 days, save everything."""
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "forecasts"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(exist_ok=True)
HORIZON = 90


def check(label, ok):
    print(("PASS  " if ok else "FAIL  ") + label)


daily = (pd.read_csv(PROC / "daily_sales.csv", parse_dates=["date"])
           .set_index("date")["revenue"].asfreq("D"))
assert len(daily) == 593 and daily.isna().sum() == 0

# ---------- fit the model (same settings as the backtest) ----------
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fit = ExponentialSmoothing(daily, trend="add", damped_trend=True,
                               seasonal="add", seasonal_periods=7).fit()

last_day = daily.index[-1]
future = pd.date_range(last_day + pd.Timedelta(days=1), periods=HORIZON, freq="D")
point = np.clip(np.asarray(fit.forecast(HORIZON)), 0, None)

# approximate 80% range from 1,000 simulated futures
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sims = np.asarray(fit.simulate(HORIZON, repetitions=1000, error="add", random_state=42))
lower = np.clip(np.percentile(sims, 10, axis=1), 0, None)
upper = np.clip(np.percentile(sims, 90, axis=1), 0, None)

# ---------- forecasts.csv: last 90 actual days + 90 predicted days ----------
pred_df = pd.DataFrame({"date": future, "type": "Predicted", "revenue": point.round(2),
                        "lower_80": lower.round(2), "upper_80": upper.round(2),
                        "horizon_day": range(1, HORIZON + 1)})
actual_df = pd.DataFrame({"date": daily.index[-90:], "type": "Actual",
                          "revenue": daily.iloc[-90:].round(2).values,
                          "lower_80": np.nan, "upper_80": np.nan, "horizon_day": np.nan})
forecasts = pd.concat([actual_df, pred_df], ignore_index=True)
forecasts.to_csv(OUT / "forecasts.csv", index=False)

# ---------- monthly forecast = SUM of daily forecasts (option a) ----------
m = pred_df.assign(month=pred_df["date"].dt.to_period("M").astype(str))
monthly = m.groupby("month").agg(days_forecast=("date", "count"),
                                 predicted_revenue=("revenue", "sum")).reset_index()
monthly["days_in_month"] = pd.PeriodIndex(monthly["month"], freq="M").days_in_month
monthly["is_full_month"] = monthly["days_forecast"] == monthly["days_in_month"]
monthly["predicted_revenue"] = monthly["predicted_revenue"].round(2)
monthly.to_csv(OUT / "forecast_monthly.csv", index=False)

# ---------- save model + notes ----------
joblib.dump(fit, MODELS / "holt_winters.pkl")
meta = {"model": "Holt-Winters (damped additive trend, additive weekly seasonality)",
        "trained_on": f"{daily.index[0].date()} to {last_day.date()}",
        "last_training_date": str(last_day.date()),
        "training_days": len(daily),
        "note": "Forecast dates are after the end of the Olist data, so no actual values exist for them."}
(MODELS / "model_info.json").write_text(json.dumps(meta, indent=2))

# backtest summary for the dashboard (30-day forecasts, 6 windows)
bt = pd.read_csv(REPORTS / "backtest_results.csv")
(bt.groupby("model")[["MAE", "MAPE_%"]].mean().round(1).reset_index()
   .to_csv(OUT / "model_metrics_backtest.csv", index=False))

# ---------- sanity checks ----------
print(f"Trained on {daily.index[0].date()} -> {last_day.date()} ({len(daily)} days)")
print(f"Forecast   {future[0].date()} -> {future[-1].date()} ({HORIZON} days)\n")
for h in (7, 30, 90):
    print(f"Next {h:>2} days: total {point[:h].sum():>12,.0f} BRL | "
          f"average/day {point[:h].mean():>8,.0f}")
print(f"\nActual average/day, last 28 days: {daily.iloc[-28:].mean():,.0f}")
print(f"Actual average/day, last 90 days: {daily.iloc[-90:].mean():,.0f}")
wd = pred_df.groupby(pred_df["date"].dt.day_name())["revenue"].mean().round(0)
print("\nForecast average by weekday:\n" + wd.to_string())
print("\nMonthly forecast (sum of daily):\n" + monthly.to_string(index=False))

print("\n=== CHECKS ===")
check("90 forecast rows, consecutive days", len(pred_df) == 90 and pred_df["date"].is_unique)
check("forecast starts the day after the data ends", future[0] == last_day + pd.Timedelta(days=1))
check("no negative or missing forecasts", (point >= 0).all() and not np.isnan(point).any())
check("range brackets the forecast on every day",
      ((lower <= point) & (point <= upper)).all())
check("average forecast is within 0.5x to 2x of the last 90 days (rough sanity)",
      0.5 * daily.iloc[-90:].mean() < point.mean() < 2 * daily.iloc[-90:].mean())
print("\nSaved: data/forecasts/forecasts.csv, forecast_monthly.csv, model_metrics_backtest.csv,")
print("       models/holt_winters.pkl, models/model_info.json")