import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


# ------------------------------------------------------------
# BUY & HOLD
# ------------------------------------------------------------
def backtest_buy_and_hold(df: pd.DataFrame) -> pd.DataFrame:
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
def backtest_momentum(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Momentum strategy based on lookback-day price change.
    Returns normalized equity curve (starting at 1).
    """
    df = df.copy()

    # Daily returns
    df["Return"] = df["Adj Close"].pct_change()

    # Momentum signal
    df["Momentum"] = df["Adj Close"].pct_change(lookback)
    df["Position"] = np.where(df["Momentum"] > 0, 1, -1)

    # Strategy returns (signal shifted by 1 day to avoid look-ahead bias)
    df["Strategy_Return"] = df["Position"].shift(1) * df["Return"]

    # Normalized equity curve (start at 1)
    df["Equity"] = (1 + df["Strategy_Return"].fillna(0)).cumprod()

    return df


# ------------------------------------------------------------
# ARIMA STRATEGY + FUTURE FORECAST
# ------------------------------------------------------------
def backtest_arima(
    df: pd.DataFrame,
    order: tuple[int, int, int] = (1, 0, 1),
    forecast_horizon: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    1) Fit ARIMA on log-returns
    2) Use sign of forecasted returns as trading signal
    3) Build normalized equity curve (starting at 1)
    4) Also return a future price forecast over `forecast_horizon` days

    Returns:
        equity_df, future_forecast_df (or None if ARIMA failed)
    """
    df = df.copy()

    # --- LOG RETURNS ---
    df["LogPrice"] = np.log(df["Adj Close"])
    df["LogReturn"] = df["LogPrice"].diff()

    # --- FIT ARIMA ---
    try:
        model = ARIMA(df["LogReturn"].dropna(), order=order)
        fitted = model.fit()

        # In-sample forecast (log-returns)
        df.loc[df.index[1:], "Forecast"] = fitted.predict()
    except Exception:
        df["Forecast"] = 0.0
        fitted = None

    # --- SIGNAL ---
    df["Signal"] = (df["Forecast"] > 0).astype(int)
    df["Signal"] = df["Signal"].replace(0, -1)

    # --- STRATEGY RETURNS & EQUITY ---
    df["Return"] = df["Adj Close"].pct_change()
    df["Strategy_Return"] = df["Signal"].shift(1) * df["Return"]

    # Normalized equity (start at 1)
    df["Equity"] = (1 + df["Strategy_Return"].fillna(0)).cumprod()

    # ------- FUTURE PRICE FORECAST -------
    future_df: pd.DataFrame | None = None

    if fitted is not None:
        # Forecast future log-returns
        fc = fitted.forecast(steps=forecast_horizon)

        # Convert cumulative log-returns → price forecast
        last_price = df["Adj Close"].iloc[-1]
        forecast_prices = last_price * np.exp(np.cumsum(fc))

        # Future business days
        future_dates = pd.date_range(
            df.index[-1],
            periods=forecast_horizon + 1,
            freq="B",
        )[1:]

        future_df = pd.DataFrame(
            {"ForecastPrice": forecast_prices.values},
            index=future_dates,
        )

    return df, future_df
