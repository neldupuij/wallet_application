import sys
import os
import matplotlib.pyplot as plt

from quant_a.download_data import download_ticker
from quant_a.performance import PerformanceMetrics
from quant_a.predict import predict_arima
from quant_a.backtest import backtest_buy_and_hold, backtest_momentum


def main():
    # Usage :
    #   python -m quant_a.test_quanta AAPL 2020-01-01 30

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
        steps_ahead = 30

    print(f"\n=== Downloading data for {ticker} since {start_date} ===")
    df = download_ticker(ticker, start_date)

    perf = PerformanceMetrics(df, ticker)
    metrics = perf.annualized_metrics()
    cum_df = perf.cumulative_return()

    print("\n=== Annualized metrics ===")
    for k, v in metrics.items():
        if isinstance(v, (float, int)):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    # === Backtests ===
    bh_df = backtest_buy_and_hold(df, ticker)
    mom_df = backtest_momentum(df, ticker, lookback=5)

    # === Normaliser le prix pour comparaison ===
    df["Normalized Price"] = df["Adj Close"] / df["Adj Close"].iloc[0]

    # === Graphique comparatif ===
    print("\n=== Plotting normalized price vs strategies ===")
    plt.figure(figsize=(10, 6))
    plt.plot(df["Normalized Price"], label="Normalized Price", color="gray", alpha=0.7)
    plt.plot(bh_df["Equity"], label="Buy & Hold", color="blue")
    plt.plot(mom_df["Equity"], label="Momentum (5d)", color="orange")
    plt.title(f"{ticker.upper()} — Normalized Price vs Strategies")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value (Start = 1)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === ARIMA ===
    csv_path = os.path.join("quant_a", "data", f"{ticker.upper()}_adj_close.csv")
    forecast_df = predict_arima(csv_path, steps_ahead)
    print("\n=== ARIMA forecast (head) ===")
    print(forecast_df.head())


if __name__ == "__main__":
    main()
