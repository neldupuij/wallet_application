import sys

from quant_a.download_data import download_ticker
from quant_a.performance import PerformanceMetrics


def main():
    # Arguments : ticker et date de début
    # Usage : python quant_a/test_quanta.py AAPL 2020-01-01
    if len(sys.argv) >= 3:
        ticker = sys.argv[1]
        start_date = sys.argv[2]
    else:
        print("No arguments provided, using default: AAPL from 2020-01-01")
        ticker = "AAPL"
        start_date = "2020-01-01"

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


if __name__ == "__main__":
    main()
