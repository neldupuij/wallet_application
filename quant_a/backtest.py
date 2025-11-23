import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import os


# ------------------------------------------------------------
# 1. Buy & Hold
# ------------------------------------------------------------
def backtest_buy_and_hold(df, ticker, save_csv=False):
    df = df.copy()
    df["Return"] = df["Adj Close"].pct_change()
    df["Equity"] = (1 + df["Return"]).cumprod()
    if save_csv:
        os.makedirs("quant_a/data", exist_ok=True)
        df[["Adj Close", "Equity"]].to_csv(f"quant_a/data/BH_{ticker}.csv")
    return df


# ------------------------------------------------------------
# 2. Momentum Strategy
# ------------------------------------------------------------
def backtest_momentum(df, ticker, lookback=5, save_csv=False):
    df = df.copy()
    df["Return"] = df["Adj Close"].pct_change()
    df[f"Lookback_{lookback}_Return"] = df["Adj Close"].pct_change(lookback)
    df["Position"] = np.where(df[f"Lookback_{lookback}_Return"] > 0, 1, -1)
    df["Strategy_Return"] = df["Position"].shift(1) * df["Return"]
    df["Equity"] = (1 + df["Strategy_Return"].fillna(0)).cumprod()

    if save_csv:
        os.makedirs("quant_a/data", exist_ok=True)
        df.to_csv(f"quant_a/data/MOM_{ticker}_L{lookback}.csv")
    return df


# ------------------------------------------------------------
# 3. ARIMA Strategy
# ------------------------------------------------------------
def backtest_arima(df, ticker, order=(1, 0, 0), save_csv=False):
    df = df.copy()
    df["Return"] = df["Adj Close"].pct_change()
    df["Signal"] = 0.0
    preds = []

    for i in range(30, len(df)):
        train = df["Adj Close"].iloc[:i]
        try:
            model = ARIMA(train, order=order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)[0]
            preds.append(forecast)
        except Exception:
            preds.append(np.nan)

    df.loc[df.index[30:], "Forecast"] = preds
    df["Signal"] = np.sign(df["Forecast"] - df["Adj Close"])
    df["Strategy_Return"] = df["Signal"].shift(1) * df["Return"]
    df["Equity"] = (1 + df["Strategy_Return"].fillna(0)).cumprod()

    if save_csv:
        os.makedirs("quant_a/data", exist_ok=True)
        df.to_csv(f"quant_a/data/ARIMA_{ticker}.csv")
    return df


# ------------------------------------------------------------
# 4. Optimized Buy & Hold (by Sharpe)
# ------------------------------------------------------------
def optimized_buy_and_hold(df, min_window=252):
    best_date = None
    best_sharpe = -np.inf
    best_equity = None

    for start_idx in range(len(df) - min_window):
        sub = df.iloc[start_idx:].copy()
        sub["Return"] = sub["Adj Close"].pct_change()
        ret = sub["Return"].dropna()
        if ret.std() == 0:
            continue
        sharpe = ret.mean() / ret.std() * np.sqrt(252)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_date = sub.index[0]
            sub["Equity"] = (1 + ret).cumprod()
            best_equity = sub["Equity"]

    return best_date, best_sharpe, best_equity
