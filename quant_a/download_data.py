import yfinance as yf
import sys
from datetime import datetime
import pandas as pd
import os


def download_ticker(ticker: str, start_date: str):
    """
    Download daily adjusted (auto-adjusted) close prices for a given ticker since start_date.
    Saves the file in quant_a/data/.
    """
    end_date = datetime.today().strftime("%Y-%m-%d")
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True)

    # Handle MultiIndex (yfinance ≥ 0.2.66)
    if isinstance(data.columns, pd.MultiIndex):
        if ("Close", ticker) in data.columns:
            data = data[("Close", ticker)].to_frame()
            data.columns = ["Adj Close"]
        else:
            raise ValueError(f"No ('Close','{ticker}') column found in data.")
    else:
        if "Adj Close" in data.columns:
            data = data[["Adj Close"]]
        elif "Close" in data.columns:
            data = data[["Close"]].rename(columns={"Close": "Adj Close"})
        else:
            raise ValueError(f"No 'Close' or 'Adj Close' column found for {ticker}")

    # === Save to quant_a/data/ ===
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)

    filename = os.path.join(data_dir, f"{ticker.upper()}_adj_close.csv")
    data.to_csv(filename)
    print(f"✅ Data for {ticker.upper()} saved to {filename}")
    return data


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python download_data.py <TICKER> <START_DATE>")
        print("Example: python download_data.py AAPL 2020-01-01")
        sys.exit(1)

    ticker = sys.argv[1]
    start_date = sys.argv[2]
    download_ticker(ticker, start_date)
