import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


# ----------------------
# BUY & HOLD (Base = 1)
# ----------------------
def backtest_buy_and_hold(df):
    df = df.copy()
    df["Equity"] = df["Adj Close"] / df["Adj Close"].iloc[0]
    return df


# ----------------------
# MOMENTUM (Base = 1)
# ----------------------
def backtest_momentum(df, lookback=20):
    df = df.copy()
    initial_price = df["Adj Close"].iloc[0]

    df["Return"] = df["Adj Close"].pct_change()
    df["Momentum"] = df["Adj Close"].pct_change(lookback)

    df["Position"] = np.where(df["Momentum"] > 0, 1, -1)
    df["Strategy_Return"] = df["Position"].shift(1) * df["Return"]

    df["Strategy_Return"] = df["Strategy_Return"].fillna(0)

    df["Equity"] = (1 + df["Strategy_Return"]).cumprod()

    return df


# ----------------------
# ARIMA FORECAST STRATEGY
# ----------------------
def backtest_arima(df, order=(1, 0, 1)):
    df = df.copy()
    initial_price = df["Adj Close"].iloc[0]

    df["LogPrice"] = np.log(df["Adj Close"])
    df["LogReturn"] = df["LogPrice"].diff()

    try:
        model = ARIMA(df["LogReturn"].dropna(), order=order)
        fitted = model.fit()

        forecast = fitted.predict()
        forecast = pd.Series(forecast, index=df.index[1:len(forecast)+1])
        df["Forecast"] = forecast
    except:
        df["Forecast"] = 0

    df["Forecast"] = df["Forecast"].fillna(0)

    df["Signal"] = np.where(df["Forecast"] > 0, 1, -1)

    df["Return"] = df["Adj Close"].pct_change()

    df["Strategy_Return"] = df["Signal"].shift(1) * df["Return"]
    df["Strategy_Return"] = df["Strategy_Return"].fillna(0)

    df["Equity"] = (1 + df["Strategy_Return"]).cumprod()

    return df
