import os
import numpy as np
import pandas as pd


def backtest_buy_and_hold(df: pd.DataFrame, ticker: str, initial_capital: float = 1.0, save_csv: bool = True) -> pd.DataFrame:
    """
    Buy & Hold: toujours investi à 100 % sur l'actif.
    Renvoie un DataFrame avec l'equity curve et sauve un CSV BH_<TICKER>.csv dans quant_a/data.
    """
    if "Adj Close" not in df.columns:
        raise ValueError("DataFrame must contain an 'Adj Close' column.")

    prices = df["Adj Close"].copy()
    daily_ret = prices.pct_change().fillna(0.0)

    equity = initial_capital * (1.0 + daily_ret).cumprod()

    out = pd.DataFrame(
        {
            "Adj Close": prices,
            "Return": daily_ret,
            "Equity": equity,
        }
    )

    if save_csv:
        save_dir = os.path.join("quant_a", "data")
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"BH_{ticker.upper()}.csv")
        out.to_csv(path)
        print(f"✅ Buy & Hold backtest saved to {path}")

    return out


def backtest_momentum(
    df: pd.DataFrame,
    ticker: str,
    lookback: int = 5,
    initial_capital: float = 1.0,
    save_csv: bool = True,
) -> pd.DataFrame:
    """
    Momentum simple:
      - on regarde le rendement sur 'lookback' jours
      - si rendement > 0 -> position 1 (investi), sinon 0 (cash)
      - signal décalé d'un jour pour éviter le look-ahead.
    Sauve un CSV MOM_<TICKER>_L<lookback>.csv dans quant_a/data.
    """
    if "Adj Close" not in df.columns:
        raise ValueError("DataFrame must contain an 'Adj Close' column.")

    prices = df["Adj Close"].copy()
    daily_ret = prices.pct_change()

    # rendement sur fenêtre lookback
    window_ret = prices / prices.shift(lookback) - 1.0

    # position basée sur ce rendement (signal décalé d'un jour)
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
        path = os.path.join(save_dir, f"MOM_{ticker.upper()}_L{lookback}.csv")
        out.to_csv(path)
        print(f"✅ Momentum backtest saved to {path}")

    return out
