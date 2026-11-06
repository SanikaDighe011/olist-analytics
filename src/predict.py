"""Reusable prediction script.  Usage:
    python src/predict.py --days 30
    python src/predict.py --days 90 --save data/forecasts/my_forecast.csv
Only load model files that you created yourself (pickle files can run code)."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "models" / "holt_winters.pkl"
INFO_FILE = ROOT / "models" / "model_info.json"


def predict(days):
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Model not found. Run: python src/forecast_final.py")
    if not 1 <= days <= 90:
        raise ValueError("days must be between 1 and 90 (accuracy is not tested beyond that)")
    fit = joblib.load(MODEL_FILE)
    info = json.loads(INFO_FILE.read_text())
    start = pd.Timestamp(info["last_training_date"]) + pd.Timedelta(days=1)
    values = np.clip(np.asarray(fit.forecast(days)), 0, None)
    return pd.DataFrame({"date": pd.date_range(start, periods=days, freq="D"),
                         "type": "Predicted", "predicted_revenue": values.round(2)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecast daily revenue (BRL).")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--save", type=str, default=None, help="optional CSV path")
    args = parser.parse_args()
    result = predict(args.days)
    print(result.to_string(index=False))
    print(f"\nTotal for {args.days} days: {result['predicted_revenue'].sum():,.0f} BRL")
    if args.save:
        result.to_csv(args.save, index=False)
        print("Saved to", args.save)