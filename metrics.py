import numpy as np
import pandas as pd

from quant_a.performance import PerformanceMetrics


def daily_returns(price_series: pd.Series) -> pd.Series:
    """
    Simple daily returns from a price series.
    """
    return price_series.pct_change().dropna()


def equity_from_returns(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """
    Equity curve starting from `initial`.
    """
    return (1 + returns).cumprod() * initial


def _metrics_from_prices(price_series: pd.Series, risk_free_rate: float = 0.0) -> dict:
    """
    Helper: wraps Quant_A PerformanceMetrics on a single price series.
    Returns a dict with Annual Return, Annual Volatility, Sharpe Ratio, Max Drawdown.
    """
    df = price_series.to_frame(name="Adj Close")
    pm = PerformanceMetrics(df)
    return pm.annualized_metrics(risk_free_rate=risk_free_rate)


def annualized_return(price_series: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Annualized return computed via Quant_A PerformanceMetrics.
    """
    m = _metrics_from_prices(price_series, risk_free_rate)
    return m["Annual Return"]


def annualized_volatility(price_series: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Annualized volatility computed via Quant_A PerformanceMetrics.
    """
    m = _metrics_from_prices(price_series, risk_free_rate)
    return m["Annual Volatility"]


def sharpe_ratio(price_series: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Sharpe ratio computed via Quant_A PerformanceMetrics.
    """
    m = _metrics_from_prices(price_series, risk_free_rate)
    return m["Sharpe Ratio"]


def max_drawdown_from_prices(price_series: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Max drawdown computed via Quant_A PerformanceMetrics.
    """
    m = _metrics_from_prices(price_series, risk_free_rate)
    return m["Max Drawdown"]


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Max drawdown computed from an equity curve (alternative, pure Quant_B).

    Gardée pour compatibilité avec ton code existant (portfolio_equity).
    """
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return drawdown.min()
