import yfinance as yf
import sys
from datetime import datetime
import pandas as pd


def download_ticker(ticker: str, start_date: str):
    """
    Download daily adjusted (auto-adjusted) close prices for a given ticker since start_date.
    Fully compatible with yfinance ≥ 0.2.66 (MultiIndex with ['Price','Ticker']).
    """
    end_date = datetime.today().strftime("%Y-%m-%d")

    # Force auto_adjust=True to get adjusted prices
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True)

    # --- Handle MultiIndex (like ('Close','AAPL')) ---
    if isinstance(data.columns, pd.MultiIndex):
        # Select ('Close', ticker)
        if ("Close", ticker) in data.columns:
            data = data[("Close", ticker)].to_frame()
            data.columns = ["Adj Close"]
        else:
            raise ValueError(f"No ('Close','{ticker}') column found in data.")
    else:
        # Old flat format
        if "Adj Close" in data.columns:
            data = data[["Adj Close"]]
        elif "Close" in data.columns:
            data = data[["Close"]].rename(columns={"Close": "Adj Close"})
        else:
            raise ValueError(f"No 'Close' or 'Adj Close' column found for {ticker}")

    # --- Save to CSV ---
    filename = f"{ticker.upper()}_adj_close.csv"
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
