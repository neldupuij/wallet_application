import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import os


def backtest_arima(df, ticker, order=(1, 0, 0), step=10, save_csv=False):
    df = df.copy()
    df["Return"] = df["Adj Close"].pct_change()
    df["Forecast"] = np.nan

    # Limiter les itérations pour performance
    for i in range(30, len(df), step):
        train = df["Adj Close"].iloc[:i]
        try:
            model = ARIMA(train, order=order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=step)
            df.loc[df.index[i:i+step], "Forecast"] = forecast.values[:len(df.index[i:i+step])]
        except Exception:
            continue

    df["Signal"] = np.sign(df["Forecast"] - df["Adj Close"])
    df["Strategy_Return"] = df["Signal"].shift(1) * df["Return"]
    df["Equity"] = (1 + df["Strategy_Return"].fillna(0)).cumprod()

    if save_csv:
        os.makedirs("quant_a/data", exist_ok=True)
        df.to_csv(f"quant_a/data/ARIMA_{ticker}.csv")
    return df
