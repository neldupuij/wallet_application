import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def backtest_arima(prices, order=(1, 0, 0), ticker=None):
    """
    Simple ARIMA timing strategy.

    Parameters
    ----------
    prices : pandas.Series or pandas.DataFrame
        Série de prix ou DataFrame contenant au moins une colonne de prix.
    order : tuple (p, d, q)
        Paramètres ARIMA.
    ticker : str, optionnel
        Non utilisé ici, gardé pour compatibilité d’interface.

    Returns
    -------
    pandas.Series
        Série d’equity (valeur du portefeuille) commençant à 1.0.
    """
    # Construire un DataFrame propre avec une colonne 'Adj Close'
    if isinstance(prices, pd.Series):
        df = prices.to_frame(name="Adj Close")
    else:
        df = prices.copy()
        if "Adj Close" not in df.columns:
            if "Adj_Close" in df.columns:
                df["Adj Close"] = df["Adj_Close"]
            elif "Close" in df.columns:
                df["Adj Close"] = df["Close"]
            else:
                first_col = df.columns[0]
                df["Adj Close"] = df[first_col]

    df = df.dropna(subset=["Adj Close"]).copy()

    # Rendements simples
    ret = df["Adj Close"].pct_change().fillna(0.0)

    # Si pas de mouvement, renvoyer equity plate
    if ret.abs().sum() == 0:
        equity = pd.Series(1.0, index=df.index, name="ARIMA")
        return equity

    # Modèle ARIMA sur les rendements
    model = SARIMAX(
        ret,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)

    # Prévision à un pas, signal = signe de la prévision (décalé d’un jour)
    mu = res.get_prediction().predicted_mean
    signal = np.sign(mu).shift(1).fillna(0.0)

    strat_ret = signal * ret
    equity = (1.0 + strat_ret).cumprod()
    equity.name = f"ARIMA{order}"

    return equity
