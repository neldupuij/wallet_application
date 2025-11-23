import os
import sys
import warnings

warnings.filterwarnings("ignore")

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# ---- Rendre quant_a importable même si ce fichier est dans quant_a/ ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from quant_a.download_data import download_ticker
from quant_a.backtest import backtest_buy_and_hold, backtest_momentum, backtest_arima


st.set_page_config(
    page_title="Quant A – Single Asset Dashboard",
    layout="wide",
)


@st.cache_data
def load_prices(ticker: str, start_date: str) -> pd.DataFrame:
    """Télécharge les prix ajustés depuis start_date."""
    return download_ticker(ticker, start_date)


def compute_metrics_from_equity(equity: pd.Series) -> dict:
    """Calcule rendement annualisé, vol, Sharpe, max drawdown à partir d'une courbe d'equity."""
    equity = equity.dropna()
    if len(equity) < 2:
        return dict(annual_return=np.nan, annual_vol=np.nan,
                    sharpe=np.nan, max_drawdown=np.nan)

    returns = equity.pct_change().dropna()
    n = len(returns)
    annual_return = (equity.iloc[-1] / equity.iloc[0]) ** (252.0 / n) - 1.0
    annual_vol = returns.std() * np.sqrt(252.0)
    sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = drawdown.min()

    return dict(
        annual_return=annual_return,
        annual_vol=annual_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
    )


def filter_range(df: pd.DataFrame, range_label: str) -> pd.DataFrame:
    """Filtre df sur 5D / 1M / 3M / 6M / 1Y / Max."""
    if df.empty or range_label == "Max":
        return df
    end = df.index.max()
    if range_label == "5D":
        delta = timedelta(days=5)
    elif range_label == "1M":
        delta = timedelta(days=30)
    elif range_label == "3M":
        delta = timedelta(days=90)
    elif range_label == "6M":
        delta = timedelta(days=182)
    elif range_label == "1Y":
        delta = timedelta(days=365)
    else:
        return df
    start = end - delta
    return df[df.index >= start]


def main():
    st.title("Quant A – Single Asset Analysis (Streamlit)")

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("Parameters")

        ticker = st.text_input("Ticker", value="AAPL")
        years = st.slider("Years of history", 1, 15, 10)

        momentum_L = st.slider("Momentum lookback (days)", 3, 120, 20, step=1)

        st.markdown("**ARIMA parameters (p, d, q)**")
        arima_p = st.number_input("p", min_value=0, max_value=10, value=2, step=1)
        arima_d = st.number_input("d", min_value=0, max_value=2, value=0, step=1)
        arima_q = st.number_input("q", min_value=0, max_value=5, value=1, step=1)

        range_label = st.radio(
            "Display range",
            ["Max", "1Y", "6M", "3M", "1M", "5D"],
            index=0,
            horizontal=True,
        )

        chart_type = st.radio(
            "Chart type",
            ["Line", "Candlestick"],
            index=0,
            horizontal=True,
        )

        run_btn = st.button("Run backtests")

    if not run_btn:
        st.info("Choisis les paramètres dans la barre de gauche puis clique sur **Run backtests**.")
        return

    # ---------------- Data loading ----------------
    start_date = (datetime.today() - timedelta(days=years * 365)).date().isoformat()
    try:
        df = load_prices(ticker, start_date)
    except Exception as e:
        st.error(f"Erreur lors du téléchargement des données : {e}")
        return

    if df.empty:
        st.warning("Pas de données renvoyées pour ce ticker / cette période.")
        return

    st.subheader(f"Price data for {ticker} (last rows)")
    st.dataframe(df.tail())

    # ---------------- Strategies ----------------
    # Buy & Hold
    try:
        bh_df = backtest_buy_and_hold(df, ticker, save_csv=False)
    except TypeError:
        bh_df = backtest_buy_and_hold(df, ticker)

    # Momentum
    try:
        mom_df = backtest_momentum(df, ticker, lookback=momentum_L, save_csv=False)
    except TypeError:
        mom_df = backtest_momentum(df, ticker, lookback=momentum_L)

    # ARIMA
    arima_order = (int(arima_p), int(arima_d), int(arima_q))
    try:
        arima_df = backtest_arima(df, ticker, order=arima_order, save_csv=False)
    except TypeError:
        arima_df = backtest_arima(df, ticker, order=arima_order)
    except Exception as e:
        st.warning(f"ARIMA {arima_order} a échoué : {e}")
        arima_df = None

    # ---------------- Metrics table ----------------
    metrics_rows = []

    bh_metrics = compute_metrics_from_equity(bh_df["Equity"])
    bh_metrics["Strategy"] = "Buy & Hold"
    metrics_rows.append(bh_metrics)

    mom_metrics = compute_metrics_from_equity(mom_df["Equity"])
    mom_metrics["Strategy"] = f"Momentum (L={momentum_L})"
    metrics_rows.append(mom_metrics)

    if arima_df is not None:
        arima_metrics = compute_metrics_from_equity(arima_df["Equity"])
        arima_metrics["Strategy"] = f"ARIMA{arima_order}"
        metrics_rows.append(arima_metrics)

    metrics_df = pd.DataFrame(metrics_rows).set_index("Strategy")
    st.subheader("Performance metrics (annualized)")
    st.dataframe(metrics_df.style.format("{:.4f}"))

    # ---------------- Plot ----------------
    price_norm = df["Adj Close"] / df["Adj Close"].iloc[0]
    plot_df = pd.DataFrame(index=df.index)
    plot_df["Price"] = price_norm

    plot_df["Buy & Hold"] = bh_df["Equity"].reindex(plot_df.index)
    plot_df["Momentum"] = mom_df["Equity"].reindex(plot_df.index)
    if arima_df is not None:
        plot_df["ARIMA"] = arima_df["Equity"].reindex(plot_df.index)

    plot_df = filter_range(plot_df, range_label)

    st.subheader(f"{ticker} — Price vs Strategies ({range_label})")

    if chart_type == "Line":
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(plot_df.index, plot_df["Price"], label="Normalized Price (Adj Close)", color="0.5", alpha=0.7)
        ax.plot(plot_df.index, plot_df["Buy & Hold"], label="Buy & Hold", linewidth=1.5)
        ax.plot(plot_df.index, plot_df["Momentum"], label=f"Momentum (L={momentum_L})", linewidth=1.5)
        if "ARIMA" in plot_df.columns:
            ax.plot(plot_df.index, plot_df["ARIMA"], label=f"ARIMA{arima_order}", linewidth=1.5)

        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized value (start = 1)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        st.pyplot(fig)

    else:
        # Candlesticks avec Plotly + stratégies superposées
        raw = yf.download(ticker, start=start_date, interval="1d", auto_adjust=False, progress=False)
        if raw.empty:
            st.warning("Impossible de récupérer les données OHLC pour afficher les bougies.")
            return

        raw = raw[["Open", "High", "Low", "Close"]].dropna()
        raw = filter_range(raw, range_label)

        if raw.empty:
            st.warning("Pas de données OHLC après filtrage pour cette fenêtre de temps.")
            return

        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=raw.index,
                open=raw["Open"],
                high=raw["High"],
                low=raw["Low"],
                close=raw["Close"],
                name="Price (candles)",
            )
        )

        # Normaliser les equity pour qu'elles commencent au même niveau que le prix initial
        base_price = float(raw["Close"].iloc[0])

        def equity_on_candles(eq: pd.Series) -> pd.Series:
            eq = eq.dropna()
            if eq.empty:
                return pd.Series(index=raw.index, dtype=float)
            eq_norm = eq / eq.iloc[0] * base_price
            return eq_norm.reindex(raw.index).ffill()

        bh_equity = equity_on_candles(bh_df["Equity"])
        mom_equity = equity_on_candles(mom_df["Equity"])
        arima_equity = equity_on_candles(arima_df["Equity"]) if arima_df is not None else None

        fig.add_trace(
            go.Scatter(
                x=raw.index,
                y=bh_equity,
                mode="lines",
                name="Buy & Hold",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=raw.index,
                y=mom_equity,
                mode="lines",
                name=f"Momentum (L={momentum_L})",
            )
        )
        if arima_equity is not None:
            fig.add_trace(
                go.Scatter(
                    x=raw.index,
                    y=arima_equity,
                    mode="lines",
                    name=f"ARIMA{arima_order}",
                )
            )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price / Equity (aligned at start)",
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            height=600,
        )

        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
