import numpy as np
import pandas as pd


TRADING_DAYS = 252


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
    Compute annualized return, volatility, Sharpe ratio and max drawdown
    directly from a price series.
    """
    rets = daily_returns(price_series)

    if rets.empty:
        return {
            "Annual Return": np.nan,
            "Annual Volatility": np.nan,
            "Sharpe Ratio": np.nan,
            "Max Drawdown": np.nan,
        }

    mean_daily = rets.mean()
    std_daily = rets.std()

    # Annualisation
    ann_return = (1 + mean_daily) ** TRADING_DAYS - 1
    ann_vol = std_daily * np.sqrt(TRADING_DAYS)

    # Sharpe (risk_free_rate est supposé annuel)
    if ann_vol == 0 or np.isnan(ann_vol):
        sharpe = np.nan
    else:
        sharpe = (ann_return - risk_free_rate) / ann_vol

    # Max drawdown à partir de l’equity curve
    equity = equity_from_returns(rets, initial=1.0)
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = drawdown.min()

    return {
        "Annual Return": ann_return,
        "Annual Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
    }


def annualized_return(price_series: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Annualized return computed from a price series.
    """
    m = _metrics_from_prices(price_series, risk_free_rate)
    return m["Annual Return"]


def annualized_volatility(price_series: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Annualized volatility computed from a price series.
    """
    m = _metrics_from_prices(price_series, risk_free_rate)
    return m["Annual Volatility"]


def sharpe_ratio(price_series: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Sharpe ratio computed from a price series.
    """
    m = _metrics_from_prices(price_series, risk_free_rate)
    return m["Sharpe Ratio"]


def max_drawdown_from_prices(price_series: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Max drawdown computed from a price series.
    """
    m = _metrics_from_prices(price_series, risk_free_rate)
    return m["Max Drawdown"]


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Max drawdown computed from an equity curve (pour compatibilité avec ton ancien code).
    """
    if equity_curve.empty:
        return np.nan

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return drawdown.min()
