import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

def predict_arima(csv_path: str, steps_ahead: int = 30):
    """
    Train an ARIMA model on a CSV file containing an 'Adj Close' column
    and forecast the next `steps_ahead` days.
    """
    # Load data
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    df = df.asfreq("B")  # business days frequency
    df["Adj Close"].interpolate(method="linear", inplace=True)

    # Build and fit ARIMA model (simple baseline)
    model = ARIMA(df["Adj Close"], order=(5, 1, 0))
    model_fit = model.fit()

    # Forecast future values
    forecast = model_fit.forecast(steps=steps_ahead)
    forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=steps_ahead, freq="B")
    forecast_df = pd.DataFrame({"Forecast": forecast}, index=forecast_dates)

    # Plot results
    plt.figure(figsize=(10,5))
    plt.plot(df["Adj Close"], label="Historical (Adj Close)")
    plt.plot(forecast_df["Forecast"], label=f"ARIMA Forecast ({steps_ahead} days)", color="orange")
    plt.title("ARIMA Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("✅ Forecast complete.")
    return forecast_df


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <CSV_PATH> [STEPS]")
        print("Example: python predict.py AAPL_adj_close.csv 30")
        sys.exit(1)

    csv_path = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    predict_arima(csv_path, steps)
