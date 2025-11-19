import yfinance as yf
import sys
from datetime import datetime
import pandas as pd
import os


def download_ticker(ticker: str, start_date: str):
    """
    Download daily adjusted (auto-adjusted) close prices for a given ticker since start_date.
    Save in quant_a/data/<TICKER>_adj_close.csv
    """
    end_date = datetime.today().strftime("%Y-%m-%d")

    data = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True)

    # Handle MultiIndex (e.g., ('Close','AAPL'))
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

    # Ensure directory exists
    save_dir = os.path.join("quant_a", "data")
    os.makedirs(save_dir, exist_ok=True)

    # Save file
    filename = os.path.join(save_dir, f"{ticker.upper()}_adj_close.csv")
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
