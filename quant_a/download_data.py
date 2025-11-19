import yfinance as yf
import sys
from datetime import datetime

def download_ticker(ticker: str, start_date: str):
    """
    Download daily adjusted close prices for a given ticker since start_date.
    """
    end_date = datetime.today().strftime("%Y-%m-%d")

    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")[["Adj Close"]]
    if data.empty:
        raise ValueError(f"No data found for {ticker} between {start_date} and {end_date}")

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
