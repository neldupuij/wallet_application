import os
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def _check_prices(df: pd.DataFrame) -> pd.Series:
    if "Adj Close" not in df.columns:
        raise ValueError("DataFrame must contain an 'Adj Close' column.")
    return df["Adj Close"].copy()


def backtest_buy_and_hold(df, ticker, initial_capital=1.0, save_csv=True):
    prices = _check_prices(df)
    daily_ret = prices.pct_change().fillna(0.0)
    equity = initial_capital * (1.0 + daily_ret).cumprod()
    out = pd.DataFrame({"Adj Close": prices, "Return": daily_ret, "Equity": equity})
    if save_csv:
        os.makedirs("quant_a/data", exist_ok=True)
        out.to_csv(f"quant_a/data/BH_{ticker.upper()}.csv")
        print(f"✅ Buy & Hold backtest saved to quant_a/data/BH_{ticker.upper()}.csv")
    return out


def backtest_momentum(df, ticker, lookback=5, initial_capital=1.0, save_csv=True):
    prices = _check_prices(df)
    daily_ret = prices.pct_change()
    window_ret = prices / prices.shift(lookback) - 1.0
    position = (window_ret > 0).astype(float).shift(1).fillna(0.0)
    strat_ret = (position * daily_ret).fillna(0.0)
    equity = initial_capital * (1.0 + strat_ret).cumprod()
    out = pd.DataFrame({
        "Adj Close": prices,
        "Return": daily_ret,
        f"Lookback_{lookback}_Return": window_ret,
        "Position": position,
        "Strategy_Return": strat_ret,
        "Equity": equity
    })
    if save_csv:
        os.makedirs("quant_a/data", exist_ok=True)
        out.to_csv(f"quant_a/data/MOM_{ticker.upper()}_L{lookback}.csv")
        print(f"✅ Momentum backtest saved to quant_a/data/MOM_{ticker.upper()}_L{lookback}.csv")
    return out


def backtest_arima(df, ticker, order=(5, 1, 0), initial_capital=1.0, save_csv=True):
    """
    Stratégie ARIMA basée sur Adj Close :

      - on ajuste ARIMA sur la série des prix ajustés
      - on prédit le prix de t+1
      - si Forecast(t+1) > Prix(t) -> position 1, sinon 0
    """
    prices = _check_prices(df).dropna().copy()

    # Met la série en fréquence jours ouvrés et propage les derniers prix
    prices = prices.asfreq("B", method="ffill")

    # Ajustement ARIMA
    model = ARIMA(prices, order=order)
    model_fit = model.fit()

    # Prévisions one-step-ahead : de l'observation 2 à la dernière
    raw_pred = model_fit.predict(start=1, end=len(prices) - 1, dynamic=False)

    # Met les prévisions sur le même index (sans NaN)
    pred_index = prices.index[1:]
    preds = pd.Series(raw_pred.values, index=pred_index)

    # Prix courants correspondants (t) pour comparer avec forecast(t+1)
    current_price = prices.iloc[:-1].set_axis(pred_index)

    # Signal : 1 si forecast > prix courant, 0 sinon
    position = (preds > current_price).astype(float)

    # Série complète des positions sur tout l'historique
    position_full = position.reindex(prices.index, fill_value=0.0)

    # Rendements journaliers
    daily_ret = prices.pct_change()

    # Rendement de la stratégie : position d'hier * rendement du jour
    strat_ret = (position_full.shift(1) * daily_ret).fillna(0.0)
    equity = initial_capital * (1.0 + strat_ret).cumprod()

    out = pd.DataFrame({
        "Adj Close": prices,
        "Forecast": preds.reindex(prices.index),
        "Position": position_full,
        "Strategy_Return": strat_ret,
        "Equity": equity
    })

    if save_csv:
        os.makedirs("quant_a/data", exist_ok=True)
        path = f"quant_a/data/ARIMA_STRAT_{ticker.upper()}.csv"
        out.to_csv(path)
        print(f"✅ ARIMA strategy backtest saved to {path}")

    return out
