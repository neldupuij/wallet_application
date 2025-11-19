import sys
import os
import matplotlib.pyplot as plt

from quant_a.download_data import download_ticker
from quant_a.performance import PerformanceMetrics
from quant_a.predict import predict_arima
from quant_a.backtest import backtest_buy_and_hold, backtest_momentum, backtest_arima


def main():
    if len(sys.argv) >= 3:
        ticker = sys.argv[1]
        start_date = sys.argv[2]
    else:
        ticker, start_date = "AAPL", "2020-01-01"

    steps_ahead = int(sys.argv[3]) if len(sys.argv) >= 4 else 30

    df = download_ticker(ticker, start_date)
    perf = PerformanceMetrics(df, ticker)
    metrics = perf.annualized_metrics()

    bh_df = backtest_buy_and_hold(df, ticker)
    mom_df = backtest_momentum(df, ticker, lookback=5)
    arima_df = backtest_arima(df, ticker, order=(5, 1, 0))

    # normalize price for fair comparison
    df["Normalized Price"] = df["Adj Close"] / df["Adj Close"].iloc[0]

    plt.figure(figsize=(10, 6))
    plt.plot(df["Normalized Price"], label="Normalized Price", color="gray", alpha=0.7)
    plt.plot(bh_df["Equity"], label="Buy & Hold", color="blue")
    plt.plot(mom_df["Equity"], label="Momentum (5d)", color="orange")
    plt.plot(arima_df["Equity"], label="ARIMA Strategy", color="green")
    plt.title(f"{ticker.upper()} — Price vs Strategies (BuyHold, Momentum, ARIMA)")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value (Start = 1)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    csv_path = os.path.join("quant_a", "data", f"{ticker.upper()}_adj_close.csv")
    forecast_df = predict_arima(csv_path, steps_ahead)
    print("\n=== ARIMA forecast (head) ===")
    print(forecast_df.head())


if __name__ == "__main__":
    main()
