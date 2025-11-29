from typing import Mapping, Sequence

import pandas as pd

from quant_B.metrics import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
)


def asset_metrics(prices: pd.DataFrame, risk_free_rate: float = 0.0) -> pd.DataFrame:
    """
    Compute asset-level metrics using Quant_A's PerformanceMetrics via Quant_B.metrics.

    Parameters
    ----------
    prices : pd.DataFrame
        Price matrix, columns = tickers.
    risk_free_rate : float
        Annual risk-free rate.

    Returns
    -------
    pd.DataFrame
        Index = tickers, columns = ["Annual Return", "Annual Volatility", "Sharpe"].
    """
    rows = []

    for col in prices.columns:
        series = prices[col].dropna()
        rows.append(
            {
                "Ticker": col,
                "Annual Return": annualized_return(series, risk_free_rate),
                "Annual Volatility": annualized_volatility(series, risk_free_rate),
                "Sharpe": sharpe_ratio(series, risk_free_rate),
            }
        )

    return pd.DataFrame(rows).set_index("Ticker")


def equal_weight(tickers: Sequence[str]) -> pd.Series:
    """
    Equal-weight allocation among all tickers.
    """
    n = len(tickers)
    if n == 0:
        raise ValueError("No tickers provided for equal-weight allocation.")
    w = 1.0 / n
    return pd.Series({t: w for t in tickers})


def normalize_weights(weights: Mapping[str, float]) -> pd.Series:
    """
    Normalize raw weights so they sum to 1. If they sum to 0, raise an error.
    """
    w = pd.Series(weights, dtype=float)
    total = w.sum()
    if total == 0:
        raise ValueError("Sum of weights is zero; cannot normalize.")
    return w / total
