import yfinance as yf
from datetime import datetime
import pandas as pd
import os
import sys


def download_ticker(ticker: str, start_date: str):
    """
    Downloads daily adjusted prices for the ticker since start_date.
    Saves result in quant_a/data/<TICKER>_adj_close.csv
    Returns a clean DataFrame with a single column: 'Adj Close'.
    """
    end_date = datetime.today().strftime("%Y-%m-%d")

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    # Handle MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        if ("Close", ticker) in data.columns:
            data = data[("Close", ticker)].rename("Adj Close").to_frame()
        else:
            raise ValueError(f"No ('Close','{ticker}') column found in data.")
    else:
        if "Adj Close" in data.columns:
            data = data[["Adj Close"]]
        elif "Close" in data.columns:
            data = data[["Close"]].rename(columns={"Close": "Adj Close"})
        else:
            raise ValueError(f"No Close or Adj Close for {ticker}")

    save_dir = os.path.join("quant_a", "data")
    os.makedirs(save_dir, exist_ok=True)

    file_path = os.path.join(save_dir, f"{ticker.upper()}_adj_close.csv")
    data.to_csv(file_path)

    print(f"✅ Downloaded: {ticker} → {file_path}")
    return data


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python download_data.py <TICKER> <START_DATE>")
        sys.exit(1)

    download_ticker(sys.argv[1], sys.argv[2])
