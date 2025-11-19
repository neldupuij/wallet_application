import os
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def backtest_buy_and_hold(df: pd.DataFrame, ticker: str, initial_capital: float = 1.0, save_csv: bool = True) -> pd.DataFrame:
    if "Adj Close" not in df.columns:
        raise ValueError("DataFrame must contain an 'Adj Close' column.")

    prices = df["Adj Close"].copy()
    daily_ret = prices.pct_change().fillna(0.0)
    equity = initial_capital * (1.0 + daily_ret).cumprod()

    out = pd.DataFrame({"Adj Close": prices, "Return": daily_ret, "Equity": equity})

    if save_csv:
        save_dir = os.path.join("quant_a", "data")
        os.makedirs(save_dir, exist_ok=True)
        out.to_csv(os.path.join(save_dir, f"BH_{ticker.upper()}.csv"))

    return out


def backtest_momentum(df: pd.DataFrame, ticker: str, lookback: int = 5, initial_capital: float = 1.0, save_csv: bool = True) -> pd.DataFrame:
    if "Adj Close" not in df.columns:
        raise ValueError("DataFrame must contain an 'Adj Close' column.")

    prices = df["Adj Close"].copy()
    daily_ret = prices.pct_change()
    window_ret = prices / prices.shift(lookback) - 1.0
    position = (window_ret > 0).astype(float).shift(1).fillna(0.0)
    strat_ret = (position * daily_ret).fillna(0.0)
    equity = initial_capital * (1.0 + strat_ret).cumprod()

    out = pd.DataFrame(
        {
            "Adj Close": prices,
            "Return": daily_ret,
            f"Lookback_{lookback}_Return": window_ret,
            "Position": position,
            "Strategy_Return": strat_ret,
            "Equity": equity,
        }
    )

    if save_csv:
        save_dir = os.path.join("quant_a", "data")
        os.makedirs(save_dir, exist_ok=True)
        out.to_csv(os.path.join(save_dir, f"MOM_{ticker.upper()}_L{lookback}.csv"))

    return out


def backtest_arima(df: pd.DataFrame, ticker: str, order=(5, 1, 0), initial_capital: float = 1.0, save_csv: bool = True) -> pd.DataFrame:
    """
    ARIMA-based strategy:
      - Fit ARIMA model recursively on rolling window
      - If forecast for next day > current price -> long
      - Else -> cash
    """
    prices = df["Adj Close"].dropna()
    n = len(prices)
    preds = []
    positions = []

    # rolling ARIMA prediction
    for t in range(30, n - 1):  # start after 30 points
        train = prices.iloc[:t]
        try:
            model = ARIMA(train, order=order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)[0]
        except Exception:
            forecast = train.iloc[-1]
        preds.append(forecast)
        positions.append(1.0 if forecast > train.iloc[-1] else 0.0)

    # align results
    preds = pd.Series(preds, index=prices.index[30:-1])
    positions = pd.Series(positions, index=prices.index[30:-1])

    # daily returns
    daily_ret = prices.pct_change().reindex_like(prices)
    strat_ret = (positions.shift(1) * daily_ret).fillna(0.0)
    equity = initial_capital * (1.0 + strat_ret).cumprod()

    out = pd.DataFrame(
        {
            "Adj Close": prices,
            "Forecast": preds.reindex_like(prices),
            "Position": positions.reindex_like(prices),
            "Strategy_Return": strat_ret,
            "Equity": equity,
        }
    )

    if save_csv:
        save_dir = os.path.join("quant_a", "data")
        os.makedirs(save_dir, exist_ok=True)
        out.to_csv(os.path.join(save_dir, f"ARIMA_STRAT_{ticker.upper()}.csv"))
        print(f"✅ ARIMA strategy backtest saved to quant_a/data/ARIMA_STRAT_{ticker.upper()}.csv")

    return out
