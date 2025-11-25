import numpy as np
import pandas as pd


def _empty_metrics() -> pd.Series:
    return pd.Series(
        {
            "final_equity": np.nan,
            "annual_return": np.nan,
            "annual_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }
    )


def compute_metrics_df(equity: pd.Series) -> pd.Series:
    """
    equity : série de la valeur du portefeuille (Equity) dans le temps.
    Retourne une Series avec les métriques standardisées.
    """
    if equity is None:
        return _empty_metrics()

    equity = equity.dropna()
    if equity.empty:
        return _empty_metrics()

    # Rendements quotidiens
    returns = equity.pct_change().dropna()
    if returns.empty:
        return _empty_metrics()

    n = len(equity)
    # On suppose des données daily, 252 jours ouvrés
    annual_factor = 252 / n

    final_equity = float(equity.iloc[-1])
    annual_return = float((equity.iloc[-1] / equity.iloc[0]) ** annual_factor - 1.0)

    annual_vol = float(returns.std() * np.sqrt(252))
    sharpe = float(annual_return / annual_vol) if annual_vol != 0 else np.nan

    # Max drawdown
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    max_drawdown = float(drawdown.min())

    return pd.Series(
        {
            "final_equity": final_equity,
            "annual_return": annual_return,
            "annual_vol": annual_vol,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
        }
    )


def compute_metrics_dict(equity: pd.Series, name: str) -> dict:
    """
    Wrapper pratique pour retourner un dict prêt à être mis dans un DataFrame.
    name : nom de la stratégie (ex: "Buy & Hold", "Momentum (L=20)", "ARIMA(2,0,1)", etc.)
    """
    s = compute_metrics_df(equity)
    d = s.to_dict()
    d["strategy"] = name
    return d
