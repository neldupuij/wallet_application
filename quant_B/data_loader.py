import yfinance as yf
import pandas as pd


def download_prices(tickers, start=None, end=None, interval: str = "1d") -> pd.DataFrame:
    """
    Download adjusted close prices for multiple tickers.

    Parameters
    ----------
    tickers : list[str] or str
        List of tickers (e.g. ["AAPL", "MSFT", "GOOG"]) or single ticker.
    start : str or None
        Start date in "YYYY-MM-DD" format.
    end : str or None
        End date in "YYYY-MM-DD" format.
    interval : str
        yfinance interval (default "1d").

    Returns
    -------
    pd.DataFrame
        Index = dates, columns = tickers, values = prices.
    """
    data = yf.download(tickers, start=start, end=end, interval=interval)["Close"]

    # If single ticker → Series, convert to DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers if isinstance(tickers, str) else None)

    return data.dropna()
