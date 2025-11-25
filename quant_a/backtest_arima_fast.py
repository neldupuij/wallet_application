import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def backtest_arima_fast(df, ticker, order=(1,0,0)):
    s = df["Adj Close"].astype(float)

    # Fit un seul ARIMA sur toute la série
    model = ARIMA(s, order=order)
    res = model.fit()

    # Prévoir toute la série
    preds = res.predict()

    # Signal simple long/short selon la prédiction
    signal = np.sign(preds - s)

    returns = s.pct_change()
    strat_ret = signal.shift(1) * returns
    equity = (1 + strat_ret.fillna(0)).cumprod()

    # Retourne simplement la courbe d'equity
    equity.name = "Equity"
    return equity
