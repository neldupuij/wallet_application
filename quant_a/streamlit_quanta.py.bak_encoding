import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# -------------------------------------------------------------------
# Path pour pouvoir faire "import quant_a.xxx" depuis la racine
# -------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from quant_a.download_data import download_ticker
from quant_a.backtest import backtest_buy_and_hold, backtest_momentum
from quant_a.backtest_arima_fast import backtest_arima
from quant_a.performance import compute_metrics_dict

# -------------------------------------------------------------------
# Univers de tickers
# -------------------------------------------------------------------
UNIVERSE = {
    "AAPL — Apple Inc.": "AAPL",
    "MSFT — Microsoft": "MSFT",
    "GOOGL — Alphabet Class A": "GOOGL",
    "META — Meta Platforms": "META",
    "NVDA — NVIDIA": "NVDA",
    "TSLA — Tesla": "TSLA",
    "ENGI.PA — Engie": "ENGI.PA",
    "MC.PA — LVMH": "MC.PA",
    "SPY — S&P 500 ETF": "SPY",
    "GLD — Gold": "GLD",
}


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def select_ticker_sidebar() -> str:
    st.sidebar.header("Parameters")

    mode = st.sidebar.radio("Ticker selection", ["From universe", "Manual"], index=0)

    if mode == "From universe":
        label = st.sidebar.selectbox("Universe (searchable)", sorted(UNIVERSE.keys()))
        ticker = UNIVERSE[label]
        override = st.sidebar.text_input("Override / custom ticker (optional)", "")
        if override.strip():
            ticker = override.strip().upper()
    else:
        ticker = st.sidebar.text_input("Ticker", "AAPL").strip().upper()

    return ticker


def load_prices(ticker: str, years_history: int) -> pd.DataFrame:
    """Télécharge les prix sur N années avec yfinance / notre fonction download_ticker."""
    end = datetime.today()
    start = end - timedelta(days=years_history * 365)

    # On passe par download_ticker pour rester cohérent avec le reste du projet
    df = download_ticker(ticker, start.strftime("%Y-%m-%d"))
    df = df.copy()
    df = df.dropna(subset=["Adj Close"])
    return df


def filter_range(df: pd.DataFrame, range_key: str) -> pd.DataFrame:
    if df.empty:
        return df
    if range_key == "Max":
        return df

    days_map = {
        "5Y": 252 * 5,
        "1Y": 252,
        "6M": 126,
        "3M": 63,
        "1M": 21,
        "5D": 5,
    }
    if range_key not in days_map:
        return df
    return df.iloc[-days_map[range_key] :]


def run_strategies(
    df: pd.DataFrame,
    ticker: str,
    lookback: int,
    arima_order: tuple[int, int, int],
):
    """Lance Buy&Hold, Momentum et ARIMA et renvoie equity + métriques."""

    # Buy & Hold et Momentum utilisent le DataFrame complet
    bh_df = backtest_buy_and_hold(df, ticker)
    mom_df = backtest_momentum(df, ticker, lookback=lookback)

    # ARIMA travaille sur la série de prix (Adj Close)
    prices = df["Adj Close"]
    arima_df = backtest_arima(prices, order=arima_order)

    # Calcul des métriques
    metrics_rows = []
    metrics_rows.append(
        {"strategy": "Buy & Hold", **compute_metrics_dict(bh_df["Equity"])}
    )
    metrics_rows.append(
        {
            "strategy": f"Momentum (L={lookback})",
            **compute_metrics_dict(mom_df["Equity"]),
        }
    )
    metrics_rows.append(
        {
            "strategy": f"ARIMA{arima_order}",
            **compute_metrics_dict(arima_df["Equity"]),
        }
    )

    metrics = pd.DataFrame(metrics_rows).set_index("strategy")
    return bh_df, mom_df, arima_df, metrics


def make_plot(
    price_df: pd.DataFrame,
    strategies: list[tuple[pd.DataFrame, str]],
    range_key: str,
    chart_type: str,
    title: str,
):
    """Construit le graphique Plotly.

    - Prix en échelle "réelle" (Adj Close)
    - Equity des stratégies re-scalées pour démarrer au même niveau que le prix
      sur la fenêtre affichée (comparaison visuelle simple).
    """

    price_plot = filter_range(price_df, range_key)

    fig = go.Figure()

    # Prix : chandeliers ou ligne
    if chart_type == "Candlestick":
        df_ohlc = price_plot.copy()
        # Si on n'a pas OHLC, on approxime avec Adj Close
        for col in ["Open", "High", "Low", "Close"]:
            if col not in df_ohlc.columns:
                df_ohlc[col] = df_ohlc["Adj Close"]

        fig.add_trace(
            go.Candlestick(
                x=df_ohlc.index,
                open=df_ohlc["Open"],
                high=df_ohlc["High"],
                low=df_ohlc["Low"],
                close=df_ohlc["Close"],
                name="Price (Adj Close)",
                yaxis="y1",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=price_plot.index,
                y=price_plot["Adj Close"],
                mode="lines",
                name="Price (Adj Close)",
                line=dict(width=1.4),
                yaxis="y1",
            )
        )

    if not price_plot.empty:
        start_price = float(price_plot["Adj Close"].iloc[0])
    else:
        start_price = 1.0

    # Stratégies : on rescale l'equity pour coller à l'échelle du prix
    for strat_df, label in strategies:
        strat_plot = filter_range(strat_df, range_key)
        if strat_plot.empty:
            continue
        equity = strat_plot["Equity"]
        scaled = equity / equity.iloc[0] * start_price

        fig.add_trace(
            go.Scatter(
                x=strat_plot.index,
                y=scaled,
                mode="lines",
                name=label,
                line=dict(width=1.4),
                yaxis="y1",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price / Equity (rescaled)",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=700,
        margin=dict(l=40, r=20, t=60, b=40),
    )

    return fig


# -------------------------------------------------------------------
# Streamlit app
# -------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Quant A — Single Asset Analysis",
        layout="wide",
    )

    ticker = select_ticker_sidebar()

    years_history = st.sidebar.slider("Years of history", 1, 20, 10)
    lookback = st.sidebar.slider("Momentum lookback (days)", 3, 120, 20)

    st.sidebar.subheader("ARIMA parameters (p, d, q)")
    p = st.sidebar.number_input("p", min_value=0, max_value=5, value=2, step=1)
    d = st.sidebar.number_input("d", min_value=0, max_value=2, value=0, step=1)
    q = st.sidebar.number_input("q", min_value=0, max_value=5, value=1, step=1)
    arima_order = (int(p), int(d), int(q))

    st.sidebar.subheader("Display range")
    range_key = st.sidebar.radio(
        "",
        ["Max", "5Y", "1Y", "6M", "3M", "1M", "5D"],
        index=0,
    )

    st.sidebar.subheader("Chart type")
    chart_type = st.sidebar.radio("", ["Line", "Candlestick"], index=0)

    run_button = st.sidebar.button("Run backtests", type="primary")

    st.title("Quant A — Single Asset Analysis")

    if not ticker:
        st.warning("Please select or type a ticker.")
        return

    if not run_button:
        st.info("Set your parameters in the sidebar, then click **Run backtests**.")
        return

    with st.spinner("Running backtests..."):
        df = load_prices(ticker, years_history)
        if df.empty:
            st.error("No data downloaded for this ticker / period.")
            return

        bh_df, mom_df, arima_df, metrics = run_strategies(
            df, ticker, lookback, arima_order
        )

    # --- Tableau de métriques ---
    st.subheader("Strategy performance metrics")
    metrics_display = metrics.copy()
    for col in ["annual_return", "annual_vol", "sharpe", "max_drawdown"]:
        if col in metrics_display.columns:
            metrics_display[col] = metrics_display[col].astype(float)

    st.dataframe(
        metrics_display.style.format(
            {
                "annual_return": "{:.2%}",
                "annual_vol": "{:.2%}",
                "sharpe": "{:.3f}",
                "max_drawdown": "{:.2%}",
            }
        ),
        use_container_width=True,
    )

    # --- Graphique principal ---
    st.subheader(f"{ticker} — Price vs Strategies")

    strategies = [
        (bh_df, "Buy & Hold"),
        (mom_df, f"Momentum (L={lookback})"),
        (arima_df, f"ARIMA{arima_order}"),
    ]

    fig = make_plot(
        price_df=df,
        strategies=strategies,
        range_key=range_key,
        chart_type=chart_type,
        title=f"{ticker} — Price vs Strategies",
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
