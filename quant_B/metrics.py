import numpy as np
import pandas as pd


def daily_returns(price_series: pd.Series) -> pd.Series:
    """Simple daily returns from price series."""
    return price_series.pct_change().dropna()


def equity_from_returns(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Equity curve starting from `initial`."""
    return (1 + returns).cumprod() * initial


def annualized_return(returns: pd.Series) -> float:
    """
    Annualized return based on daily returns.
    """
    mean_daily = returns.mean()
    return (1 + mean_daily) ** 252 - 1


def annualized_volatility(returns: pd.Series) -> float:
    """
    Annualized volatility based on daily returns.
    """
    return returns.std() * np.sqrt(252)


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """
    Annualized Sharpe ratio with risk-free rate rf (annual).
    """
    excess = returns - rf / 252
    if excess.std() == 0:
        return np.nan
    return np.sqrt(252) * excess.mean() / excess.std()


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum drawdown (negative number).
    """
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return drawdown.min()
