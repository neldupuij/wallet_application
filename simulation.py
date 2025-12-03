import numpy as np
import pandas as pd

from quant_B.metrics import daily_returns


def simulate_portfolio(
    prices: pd.DataFrame,
    weights: pd.Series,
    rebalance_freq: str = "M",
) -> tuple[pd.Series, pd.Series]:
    """
    Simulate a portfolio with a given static allocation and rebalancing.

    Parameters
    ----------
    prices : pd.DataFrame
        Price matrix (columns = tickers).
    weights : pd.Series
        Target weights indexed by tickers; should sum to 1.
    rebalance_freq : str
        'D' (daily), 'W' (weekly), or 'M' (monthly).

    Returns
    -------
    (portfolio_returns, portfolio_equity)
    """
    # Ensure tickers match
    weights = weights.reindex(prices.columns).fillna(0.0)

    rets = prices.pct_change().dropna()

    # Build weight matrix over time
    if rebalance_freq == "D":
        w_t = pd.DataFrame(
            np.tile(weights.values, (len(rets), 1)),
            index=rets.index,
            columns=rets.columns,
        )
    else:
        w_t = pd.DataFrame(index=rets.index, columns=rets.columns)
        rebal_dates = rets.resample(rebalance_freq).first().index
        current_w = weights.copy()

        for d in rets.index:
            if d in rebal_dates:
                current_w = weights.copy()
            w_t.loc[d] = current_w

    # Portfolio returns and equity
    port_rets = (w_t * rets).sum(axis=1)
    port_equity = (1 + port_rets).cumprod()

    return port_rets, port_equity
