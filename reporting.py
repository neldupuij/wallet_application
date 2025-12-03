import pandas as pd

from metrics import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown,
)


def portfolio_metrics(
    port_rets: pd.Series,
    port_equity: pd.Series,
    asset_metrics_df: pd.DataFrame,
    weights: pd.Series,
) -> dict:
    """
    Compute portfolio-level metrics and diversification effect.

    Parameters
    ----------
    port_rets : pd.Series
        Portfolio daily returns.
    port_equity : pd.Series
        Portfolio cumulative equity curve.
    asset_metrics_df : pd.DataFrame
        Output of weights.asset_metrics(prices).
    weights : pd.Series
        Portfolio weights.

    Returns
    -------
    dict
        Annual return, vol, Sharpe, MDD, weighted avg vol, diversification effect.
    """
    ann_ret = annualized_return(port_rets)
    ann_vol = annualized_volatility(port_rets)
    sharpe = sharpe_ratio(port_rets)
    mdd = max_drawdown(port_equity)

    # Weighted average asset volatility (no diversification)
    weights = weights.reindex(asset_metrics_df.index).fillna(0.0)
    avg_vol = (weights * asset_metrics_df["Annual Volatility"]).sum()

    divers_effect = avg_vol - ann_vol

    return {
        "Annual Return": ann_ret,
        "Annual Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": mdd,
        "Weighted Avg Asset Vol": avg_vol,
        "Diversification Effect (avg_vol - port_vol)": divers_effect,
    }


def build_correlation_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Correlation matrix of asset daily returns.
    """
    rets = prices.pct_change().dropna()
    return rets.corr()
