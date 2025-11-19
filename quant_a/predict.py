import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import os

def predict_arima(csv_path: str, steps_ahead: int = 30):
    """
    Train ARIMA model on given CSV and forecast next steps_ahead days.
    Save output as quant_a/data/ARIMA_<TICKER>.csv
    """
    # Load data
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    df = df.asfreq("B")
    df["Adj Close"].interpolate(method="linear", inplace=True)

    # Fit ARIMA model
    model = ARIMA(df["Adj Close"], order=(5, 1, 0))
    model_fit = model.fit()

    # Forecast
    forecast = model_fit.forecast(steps=steps_ahead)
    forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=steps_ahead, freq="B")
    forecast_df = pd.DataFrame({"Forecast": forecast}, index=forecast_dates)

    # Extract ticker name for saving
    ticker_name = os.path.basename(csv_path).split("_")[0].upper()

    # Ensure directory exists
    save_dir = os.path.join("quant_a", "data")
    os.makedirs(save_dir, exist_ok=True)

    # Save predictions
    out_path = os.path.join(save_dir, f"ARIMA_{ticker_name}.csv")
    forecast_df.to_csv(out_path)
    print(f"✅ ARIMA forecast saved to {out_path}")

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(df["Adj Close"], label="Historical (Adj Close)")
    plt.plot(forecast_df["Forecast"], label=f"ARIMA Forecast ({steps_ahead} days)", color="orange")
    plt.title(f"ARIMA Forecast - {ticker_name}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return forecast_df

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <CSV_PATH> [STEPS]")
        print("Example: python predict.py quant_a/data/AAPL_adj_close.csv 30")
        sys.exit(1)

    csv_path = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    predict_arima(csv_path, steps)
