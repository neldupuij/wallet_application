import os
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def _check_prices(df: pd.DataFrame) -> pd.Series:
    if "Adj Close" not in df.columns:
        raise ValueError("DataFrame must contain an 'Adj Close' column.")
    return df["Adj Close"].copy()


def backtest_arima(
    df: pd.DataFrame,
    ticker: str,
    order=(5, 1, 0),
    initial_capital: float = 1.0,
    save_csv: bool = True,
) -> pd.DataFrame:
    """
    Stratégie basée sur ARIMA :
      - Ajuste un ARIMA(p,d,q) sur la série complète.
      - Si la prévision du jour suivant est supérieure au prix courant → achat (1), sinon cash (0).
    """

    prices = _check_prices(df).dropna().copy()

    # ⚙️ Forcer une fréquence BusinessDay (B) pour supprimer les warnings
    if prices.index.freq is None:
        prices.index = pd.DatetimeIndex(prices.index).to_period("B").to_timestamp()

    # Ajustement du modèle ARIMA
    model = ARIMA(prices, order=order)
    model_fit = model.fit()

    # Prévision one-step-ahead
    preds = model_fit.predict(start=1, end=len(prices), dynamic=False)
    preds = preds.reindex(prices.index)

    # Position binaire selon la tendance prévue
    position = (preds > prices).astype(float)

    # Rendements
    daily_ret = prices.pct_change()
    strat_ret = (position.shift(1) * daily_ret).fillna(0.0)
    equity = initial_capital * (1.0 + strat_ret).cumprod()

    out = pd.DataFrame(
        {
            "Adj Close": prices,
            "Forecast": preds,
            "Position": position,
            "Strategy_Return": strat_ret,
            "Equity": equity,
        }
    )

    if save_csv:
        save_dir = os.path.join("quant_a", "data")
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"ARIMA_STRAT_{ticker.upper()}.csv")
        out.to_csv(path)
        print(f"✅ ARIMA strategy backtest saved to {path}")

    return out
