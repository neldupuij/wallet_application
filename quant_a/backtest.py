import numpy as np
import pandas as pd

def backtest_buy_and_hold(df, ticker, save_csv=False):
    df = df.copy()
    df["Return"] = df["Adj Close"].pct_change()
    df["Equity"] = (1 + df["Return"]).cumprod()
    if save_csv:
        df[["Adj Close", "Equity"]].to_csv(f"quant_a/data/{ticker}_buyhold.csv")
    return df

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
