import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


# ------------------------------------------------------------
# BUY & HOLD
# ------------------------------------------------------------
def backtest_buy_and_hold(df):
    """
    Simple buy & hold strategy.
    Returns normalized equity curve (starting at 1).
    """
    df = df.copy()
    initial = df["Adj Close"].iloc[0]
    df["Equity"] = df["Adj Close"] / initial
    return df


# ------------------------------------------------------------
# MOMENTUM STRATEGY
# ------------------------------------------------------------
def backtest_momentum(df, lookback=20):
    """
    Long if momentum>0, short otherwise.
    Returns real-value equity curve (scaled later by Streamlit).
    """
    df = df.copy()
    initial = df["Adj Close"].iloc[0]

    # compute returns
    df["Return"] = df["Adj Close"].pct_change()
    df["Momentum"] = df["Adj Close"].pct_change(lookback)

    # long / short signal
    df["Position"] = np.where(df["Momentum"] > 0, 1, -1)

    # strategy returns
    df["Strategy_Return"] = df["Position"].shift(1) * df["Return"]

    # cumulative performance
    df["Equity"] = (1 + df["Strategy_Return"].fillna(0)).cumprod()

    # scale to real amount (done in streamlit)
    df["Equity"] *= initial

    return df


# ------------------------------------------------------------
# ARIMA STRATEGY + FUTURE FORECAST
# ------------------------------------------------------------
def backtest_arima(df, order=(1, 0, 1), forecast_horizon=30):
    """
    1) Fit ARIMA on log-returns  
    2) Use sign of forecasted returns as trading signal  
    3) Build equity curve  
    4) Also returns 30-day future price forecast

    Returns:
        df_equity, df_future
    """
    df = df.copy()

    # --- LOG RETURNS ---
    df["LogPrice"] = np.log(df["Adj Close"])
    df["LogReturn"] = df["LogPrice"].diff()

    # --- FIT ARIMA ---
    try:
        model = ARIMA(df["LogReturn"].dropna(), order=order)
        fitted = model.fit()

        # In-sample forecast
        df.loc[df.index[1:], "Forecast"] = fitted.predict()
    except Exception:
        df["Forecast"] = 0.0
        fitted = None

    # --- SIGNAL ---
    df["Signal"] = (df["Forecast"] > 0).astype(int)
    df["Signal"] = df["Signal"].replace(0, -1)

    # --- STRATEGY RETURNS ---
    df["Return"] = df["Adj Close"].pct_change()
    df["Strategy_Return"] = df["Signal"].shift(1) * df["Return"]

    df["Equity"] = (1 + df["Strategy_Return"].fillna(0)).cumprod()

    # ------- FUTURE PRICE FORECAST -------
    future_df = None

    if fitted is not None:
        # forecast log returns into the future
        fc = fitted.forecast(steps=forecast_horizon)

        # convert cumulative log-returns → price forecast
        last_price = df["Adj Close"].iloc[-1]
        forecast_prices = last_price * np.exp(np.cumsum(fc))

        # future business days
        future_dates = pd.date_range(
            df.index[-1],
            periods=forecast_horizon + 1,
            freq="B"
        )[1:]

        future_df = pd.DataFrame(
            {"ForecastPrice": forecast_prices.values},
            index=future_dates
        )

    return df, future_df
