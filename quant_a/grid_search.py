import sys
import os
import numpy as np
import pandas as pd

from quant_a.download_data import download_ticker
from quant_a.backtest import (
    backtest_buy_and_hold,
    backtest_momentum,
    backtest_arima,
)


def compute_metrics(equity: pd.Series) -> dict:
    """
    Calcule quelques métriques standard à partir de la courbe d'equity.
    """
    equity = equity.dropna()
    if len(equity) < 2:
        return {
            "annual_return": np.nan,
            "annual_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }

    returns = equity.pct_change().dropna()
    n = len(returns)

    # On suppose 252 jours de bourse par an
    annual_return = (equity.iloc[-1] / equity.iloc[0]) ** (252.0 / n) - 1.0
    annual_vol = returns.std() * np.sqrt(252.0)
    sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = drawdown.min()

    return {
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def main():
    """
    Usage :
        python -m quant_a.grid_search AAPL 2020-01-01
    """

    if len(sys.argv) >= 3:
        ticker = sys.argv[1]
        start_date = sys.argv[2]
    else:
        print("No args provided, using default: AAPL 2020-01-01")
        ticker = "AAPL"
        start_date = "2020-01-01"

    print(f"\n=== Grid search for {ticker} starting {start_date} ===")

    # -----------------------------
    # 1) Données de prix
    # -----------------------------
    df = download_ticker(ticker, start_date)

    results = []

    # -----------------------------
    # 2) Buy & Hold de référence
    # -----------------------------
    print("\n=== Reference: Buy & Hold ===")
    try:
        bh_df = backtest_buy_and_hold(df, ticker, save_csv=False)
        bh_metrics = compute_metrics(bh_df["Equity"])
        bh_metrics.update({
            "strategy": "buy_hold",
            "lookback": np.nan,
            "p": np.nan,
            "d": np.nan,
            "q": np.nan,
        })
        results.append(bh_metrics)
        print(f"Buy & Hold: Sharpe={bh_metrics['sharpe']:.3f}, "
              f"AnnRet={bh_metrics['annual_return']:.3f}, "
              f"MaxDD={bh_metrics['max_drawdown']:.3f}")
    except Exception as e:
        print(f"⚠️ Buy & Hold failed: {e}")

    # -----------------------------
    # 3) Grid search Momentum
    # -----------------------------
    momentum_lookbacks = [3, 5, 10, 20, 40, 60, 90]

    print("\n=== Testing Momentum lookbacks ===")
    for lb in momentum_lookbacks:
        try:
            strat_df = backtest_momentum(df, ticker, lookback=lb, save_csv=False)
            metrics = compute_metrics(strat_df["Equity"])
            metrics.update({
                "strategy": "momentum",
                "lookback": lb,
                "p": np.nan,
                "d": np.nan,
                "q": np.nan,
            })
            results.append(metrics)
            print(f"Momentum L={lb:>3}: Sharpe={metrics['sharpe']:.3f}, "
                  f"AnnRet={metrics['annual_return']:.3f}, "
                  f"MaxDD={metrics['max_drawdown']:.3f}")
        except Exception as e:
            print(f"⚠️ Momentum L={lb} failed: {e}")

    # -----------------------------
    # 4) Grid search ARIMA
    # -----------------------------
    # Grille un peu plus large mais raisonnable
    arima_orders = []
    for p in [0, 1, 2, 3, 5]:
        for d in [0, 1]:
            for q in [0, 1, 2]:
                arima_orders.append((p, d, q))

    print("\n=== Testing ARIMA (p,d,q) ===")
    for order in arima_orders:
        p, d, q = order
        try:
            strat_df = backtest_arima(df, ticker, order=order, save_csv=False)
            metrics = compute_metrics(strat_df["Equity"])
            metrics.update({
                "strategy": "arima",
                "lookback": np.nan,
                "p": p,
                "d": d,
                "q": q,
            })
            results.append(metrics)
            print(f"ARIMA{order}: Sharpe={metrics['sharpe']:.3f}, "
                  f"AnnRet={metrics['annual_return']:.3f}, "
                  f"MaxDD={metrics['max_drawdown']:.3f}")
        except Exception as e:
            print(f"⚠️ ARIMA{order} failed: {e}")

    # -----------------------------
    # 5) Sauvegarde des résultats
    # -----------------------------
    if results:
        res_df = pd.DataFrame(results)

        # Tri par Sharpe décroissant
        res_df_sorted = res_df.sort_values(by="sharpe", ascending=False)

        save_dir = os.path.join("quant_a", "data")
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"GRIDSEARCH_{ticker.upper()}.csv")
        res_df_sorted.to_csv(out_path, index=False)
        print(f"\n✅ Grid search results saved to {out_path}")
        print("\n=== Top 10 configurations ===")
        print(res_df_sorted.head(10))
    else:
        print("No results computed (all configurations failed?).")


if __name__ == "__main__":
    main()
