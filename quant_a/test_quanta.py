import sys
import os

from quant_a.download_data import download_ticker
from quant_a.performance import PerformanceMetrics
from quant_a.predict import predict_arima


def main():
    # Usage :
    #   python -m quant_a.test_quanta AAPL 2020-01-01 30
    #   (3rd arg = number of forecast days, optional)

    if len(sys.argv) >= 3:
        ticker = sys.argv[1]
        start_date = sys.argv[2]
    else:
        print("No arguments provided, using default: AAPL from 2020-01-01")
        ticker = "AAPL"
        start_date = "2020-01-01"

    if len(sys.argv) >= 4:
        steps_ahead = int(sys.argv[3])
    else:
        steps_ahead = 30  # default forecast horizon

    print(f"\n=== Downloading data for {ticker} since {start_date} ===")
    df = download_ticker(ticker, start_date)

    print("\n=== Head of downloaded data ===")
    print(df.head())

    perf = PerformanceMetrics(df)

    print("\n=== Annualized metrics ===")
    metrics = perf.annualized_metrics()
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n=== Cumulative return sample ===")
    cum_df = perf.cumulative_return()
    print(cum_df.head())

    # === ARIMA PREDICTION PART ===
    csv_path = os.path.join("quant_a", "data", f"{ticker.upper()}_adj_close.csv")
    print(f"\n=== ARIMA forecast on {csv_path} for {steps_ahead} business days ===")
    forecast_df = predict_arima(csv_path, steps_ahead)
    print("\n=== ARIMA forecast (head) ===")
    print(forecast_df.head())


if __name__ == "__main__":
    main()
