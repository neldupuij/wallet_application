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
    prices = _check_prices(df).dropna().copy()
    # Force business-day frequency to avoid warnings
    if prices.index.freq is None:
        prices.index = pd.DatetimeIndex(prices.index).to_period("B").to_timestamp()
    model = ARIMA(prices, order=order)
    model_fit = model.fit()
    preds = model_fit.predict(start=1, end=len(prices), dynamic=False)
    preds = preds.reindex(prices.index)
    position = (preds > prices).astype(float)
    daily_ret = prices.pct_change()
    strat_ret = (position.shift(1) * daily_ret).fillna(0.0)
    equity = initial_capital * (1.0 + strat_ret).cumprod()
    out = pd.DataFrame({
        "Adj Close": prices,
        "Forecast": preds,
        "Position": position,
        "Strategy_Return": strat_ret,
        "Equity": equity
    })
    if save_csv:
        os.makedirs("quant_a/data", exist_ok=True)
        out.to_csv(f"quant_a/data/ARIMA_STRAT_{ticker.upper()}.csv")
        print(f"✅ ARIMA strategy backtest saved to quant_a/data/ARIMA_STRAT_{ticker.upper()}.csv")
    return out
