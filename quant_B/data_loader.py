import pandas as pd
from datetime import datetime

from quant_a.download_data import download_ticker


def download_prices(tickers, start=None, end=None):
    """
    Multi-asset price downloader for Quant_B, built on top of Quant_A's download_ticker.

    Parameters
    ----------
    tickers : str or list[str]
        Single ticker or list of tickers.
    start : str or None
        Start date "YYYY-MM-DD". If None, defaults to "2010-01-01".
    end : str or None
        End date "YYYY-MM-DD". If None, uses today's date.

    Returns
    -------
    pd.DataFrame
        Index = dates, columns = tickers, values = adjusted close prices.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    if start is None:
        start = "2010-01-01"
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    price_dict = {}

    for t in tickers:
        # utilise la fonction de Quant_A
        df = download_ticker(t, start)
        # df contient une colonne "Adj Close" et un index de dates
        series = df["Adj Close"]

        # filtre sur end si nécessaire
        series = series.loc[:end]
        price_dict[t] = series

    # concatène toutes les séries en un DataFrame
    prices = pd.DataFrame(price_dict).dropna(how="all")

    return prices
