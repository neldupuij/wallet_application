import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


# ----------------------
# BUY & HOLD (VALEUR RÉELLE)
# ----------------------
def backtest_buy_and_hold(df):
    df = df.copy()
    initial_price = df["Adj Close"].iloc[0]
    df["Equity"] = df["Adj Close"] / initial_price
    return df


# ----------------------
# MOMENTUM (VALEUR RÉELLE)
# ----------------------
def backtest_momentum(df, lookback=20):
    df = df.copy()
    initial = df["Adj Close"].iloc[0]

    df["Return"] = df["Adj Close"].pct_change()
    df["Momentum"] = df["Adj Close"].pct_change(lookback)

    df["Position"] = np.where(df["Momentum"] > 0, 1, -1)
    df["Strategy_Return"] = df["Position"].shift(1) * df["Return"]

    df["Equity"] = (1 + df["Strategy_Return"].fillna(0)).cumprod()
    df["Equity"] = df["Equity"] * initial  # SCALE IN REAL PRICE

    return df


def backtest_arima(df, order=(1, 0, 1)):
    df = df.copy()
    initial = df["Adj Close"].iloc[0]

    # Transform to log-returns (stationary)
    df["LogPrice"] = np.log(df["Adj Close"])
    df["LogReturn"] = df["LogPrice"].diff()

    try:
        model = ARIMA(df["LogReturn"].dropna(), order=order)
        fitted = model.fit()
        df.loc[df.index[1:], "Forecast"] = fitted.predict()
    except:
        df["Forecast"] = 0  # neutral signal

    # Convert forecasted returns to signals
    df["Signal"] = (df["Forecast"] > 0).astype(int)
    df["Signal"] = df["Signal"].replace(0, -1)

    # Strategy on ORIGINAL returns
    df["Return"] = df["Adj Close"].pct_change()
    df["Strategy_Return"] = df["Signal"].shift(1) * df["Return"]

    df["Equity"] = (1 + df["Strategy_Return"].fillna(0)).cumprod()
    df["Equity"] = df["Equity"] * initial

    return df
