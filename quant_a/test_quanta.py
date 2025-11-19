import sys
import os

import pandas as pd
import matplotlib.pyplot as plt

from quant_a.download_data import download_ticker
from quant_a.performance import PerformanceMetrics
from quant_a.predict import predict_arima
from quant_a.backtest import (
    backtest_buy_and_hold,
    backtest_momentum,
    backtest_arima,
)


def load_best_params(ticker: str):
    """
    Lit le fichier GRIDSEARCH_<TICKER>.csv et renvoie :
      - best_mom_lookback (int ou None)
      - best_arima_order  (tuple (p,d,q) ou None)
    """
    path = os.path.join("quant_a", "data", f"GRIDSEARCH_{ticker.upper()}.csv")
    if not os.path.exists(path):
        print(f"⚠️ No grid search file found at {path}. Using default params.")
        return None, None

    gs = pd.read_csv(path)

    # Momentum
    best_mom_lookback = None
    mom = gs[gs["strategy"] == "momentum"]
    if not mom.empty:
        mom_sorted = mom.sort_values("sharpe", ascending=False)
        best_mom_lookback = int(mom_sorted.iloc[0]["lookback"])

    # ARIMA
    best_arima_order = None
    ar = gs[gs["strategy"] == "arima"]
    if not ar.empty:
        ar_sorted = ar.sort_values("sharpe", ascending=False)
        row = ar_sorted.iloc[0]
        best_arima_order = (int(row["p"]), int(row["d"]), int(row["q"]))

    return best_mom_lookback, best_arima_order


def main():
    """
    Usage :
        python -m quant_a.test_quanta AAPL 2020-01-01 30
    """
    if len(sys.argv) >= 3:
        ticker = sys.argv[1]
        start_date = sys.argv[2]
    else:
        print("No arguments provided, using default: AAPL from 2020-01-01")
        ticker = "AAPL"
        start_date = "2020-01-01"

    if len(sys.argv) >= 4:
        steps_ahead = int(sys.argv[3])
    else:
        steps_ahead = 30

    print(f"\n=== Downloading data for {ticker} since {start_date} ===")
    df = download_ticker(ticker, start_date)

    # --- métriques globales sur l'actif ---
    perf = PerformanceMetrics(df, ticker)
    print("\n=== Annualized metrics on underlying ===")
    metrics = perf.annualized_metrics()
    for k, v in metrics.items():
        if isinstance(v, (float, int)):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    # --- récupérer les meilleurs paramètres Momentum / ARIMA ---
    best_mom_lookback, best_arima_order = load_best_params(ticker)

    # valeurs par défaut si pas de grid search dispo
    if best_mom_lookback is None:
        best_mom_lookback = 5
    if best_arima_order is None:
        best_arima_order = (5, 1, 0)

    print(f"\n✅ Using Momentum lookback = {best_mom_lookback}")
    print(f"✅ Using ARIMA order = {best_arima_order}")

    # --- Backtests ---
    print("\n=== Buy & Hold backtest (head) ===")
    bh_df = backtest_buy_and_hold(df, ticker)
    print(bh_df.head())

    print(f"\n=== Momentum backtest (best lookback={best_mom_lookback}) ===")
    mom_df = backtest_momentum(df, ticker, lookback=best_mom_lookback)
    print(mom_df.head())

    print(f"\n=== ARIMA strategy backtest (best order={best_arima_order}) ===")
    arima_df = backtest_arima(df, ticker, order=best_arima_order)
    print(arima_df.head())

    # --- Graphique : prix adj normalisé vs stratégies ---
    df = df.copy()
    df["Normalized Price"] = df["Adj Close"] / df["Adj Close"].iloc[0]

    print("\n=== Plotting normalized Adj Close vs best strategies ===")
    plt.figure(figsize=(12, 7))
    plt.plot(df.index, df["Normalized Price"], label="Normalized Price (Adj Close)", color="gray", alpha=0.7)
    plt.plot(bh_df.index, bh_df["Equity"], label="Buy & Hold", color="blue")
    plt.plot(mom_df.index, mom_df["Equity"], label=f"Momentum (L={best_mom_lookback})", color="orange")
    plt.plot(arima_df.index, arima_df["Equity"], label=f"ARIMA {best_arima_order}", color="green")

    plt.title(f"{ticker.upper()} — Price vs Best Strategies (Momentum & ARIMA)")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value (Start = 1)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Bonus : forecast ARIMA classique (sur CSV adj close) ---
    csv_path = os.path.join("quant_a", "data", f"{ticker.upper()}_adj_close.csv")
    print(f"\n=== ARIMA forecast on {csv_path} for {steps_ahead} business days ===")
    try:
        forecast_df = predict_arima(csv_path, steps_ahead)
        print("\n=== ARIMA forecast (head) ===")
        print(forecast_df.head())
    except Exception as e:
        print(f"⚠️ ARIMA forecast failed: {e}")


if __name__ == "__main__":
    main()
