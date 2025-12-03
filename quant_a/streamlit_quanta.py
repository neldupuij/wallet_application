import os
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from download_data import download_ticker
from backtest import (
    backtest_buy_and_hold,
    backtest_momentum,
    backtest_arima,
)

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------
def compute_metrics(equity: pd.Series):
    """Compute Return, Vol, Sharpe, MaxDD from an equity curve."""
    equity = equity.astype(float)
    returns = equity.pct_change().dropna()

    if returns.empty or equity.iloc[0] == 0:
        return {"Return": 0.0, "Vol": 0.0, "Sharpe": 0.0, "MaxDD": 0.0}

    ann_return = (equity.iloc[-1] / equity.iloc[0]) ** (252 / len(equity)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    max_dd = (equity / equity.cummax() - 1).min()

    return {
        "Return": ann_return,
        "Vol": ann_vol,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
    }


# ------------------------------------------------------------
# STREAMLIT APP
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="Quant A", layout="wide")

    st.title("📈 Quant A – Single Asset Backtesting")
    st.caption("Buy & Hold • Momentum • ARIMA • Grid Search • Forecast • Stable UI")

    # --------------------------------------------------------
    # Session state initialisation
    # --------------------------------------------------------
    if "results" not in st.session_state:
        st.session_state.results = None
    if "gs_results" not in st.session_state:
        st.session_state.gs_results = None

    # --------------------------------------------------------
    # Sidebar
    # --------------------------------------------------------
    st.sidebar.header("Parameters")

    tickers = ["AAPL", "MSFT", "NVDA", "GOOG", "TSLA", "SPY", "EURUSD=X", "BTC-USD"]
    choice = st.sidebar.selectbox("Choose asset", ["Custom ticker"] + tickers)

    ticker = st.sidebar.text_input(
        "Enter ticker",
        choice if choice != "Custom ticker" else "AAPL"
    ).upper()

    start_date = st.sidebar.date_input(
        "Start date",
        value=datetime(2015, 1, 1),
        format="YYYY/MM/DD"
    ).strftime("%Y-%m-%d")

    initial_equity = st.sidebar.number_input("Initial investment", value=10000, step=100)

    st.sidebar.subheader("Momentum")
    momentum_lb = st.sidebar.slider("Lookback", 2, 120, 20)

    st.sidebar.subheader("ARIMA (p,d,q)")
    p = st.sidebar.slider("p", 0, 5, 1)
    d = st.sidebar.slider("d", 0, 2, 0)
    q = st.sidebar.slider("q", 0, 5, 1)

    st.sidebar.subheader("Grid Search")
    gs_mom = st.sidebar.checkbox("Momentum Grid Search")
    gs_ari = st.sidebar.checkbox("ARIMA Grid Search")

    st.sidebar.subheader("View range")
    view_range = st.sidebar.selectbox(
        "Display period",
        ["1W", "1M", "3M", "6M", "1Y", "5Y", "MAX"],
        index=6,
    )

    run_bt = st.sidebar.button("🚀 Run Backtests")

    # --------------------------------------------------------
    # RUN BACKTESTS ONLY WHEN BUTTON IS CLICKED
    # --------------------------------------------------------
    if run_bt:
        try:
            df = download_ticker(ticker, start_date)
            df = df[~df.index.duplicated()].dropna()
        except Exception as e:
            st.error(f"Error downloading data: {e}")
            return

        # Run strategies
        bh = backtest_buy_and_hold(df)
        mom = backtest_momentum(df, momentum_lb)
        ari, ari_future = backtest_arima(df, (p, d, q), forecast_horizon=30)

        # Scale equity
        bh["Equity"] *= initial_equity
        mom["Equity"] *= initial_equity
        ari["Equity"] *= initial_equity

        # Save results
        st.session_state.results = {
            "df": df,
            "bh": bh,
            "mom": mom,
            "ari": ari,
            "ari_future": ari_future,
            "momentum_lb": momentum_lb,
            "order": (p, d, q),
            "initial_equity": initial_equity,
            "view_range": view_range
        }

        # ---------------- GRID SEARCH (only once per run_bt)
        gs_output = {}

        # Momentum GS
        if gs_mom:
            mom_list = []
            best_L = None
            best_sharpe = -999

            for L in range(2, 121):
                eq = backtest_momentum(df, L)["Equity"]
                s = compute_metrics(eq)["Sharpe"]
                mom_list.append((L, s))
                if s > best_sharpe:
                    best_L, best_sharpe = L, s

            gs_output["mom"] = (mom_list, best_L, best_sharpe)

        # ARIMA GS
        if gs_ari:
            ari_list = []
            best_tuple = None
            best_sharpe = -999

            for P in range(0, 6):
                for D in range(0, 2):
                    for Q in range(0, 6):
                        eq_df, _ = backtest_arima(df, (P, D, Q))
                        s = compute_metrics(eq_df["Equity"])["Sharpe"]
                        ari_list.append((P, D, Q, s))
                        if s > best_sharpe:
                            best_tuple, best_sharpe = (P, D, Q), s

            gs_output["ari"] = (ari_list, best_tuple, best_sharpe)

        st.session_state.gs_results = gs_output

    # --------------------------------------------------------
    # IF NO BACKTEST DATA YET
    # --------------------------------------------------------
    if st.session_state.results is None:
        st.info("Configure parameters then click Run.")
        return

    # --------------------------------------------------------
    # LOAD RESULTS
    # --------------------------------------------------------
    df = st.session_state.results["df"]
    bh = st.session_state.results["bh"]
    mom = st.session_state.results["mom"]
    ari = st.session_state.results["ari"]
    ari_future = st.session_state.results["ari_future"]
    momentum_lb = st.session_state.results["momentum_lb"]
    p, d, q = st.session_state.results["order"]
    initial_equity = st.session_state.results["initial_equity"]
    view_range = st.session_state.results["view_range"]

    # --------------------------------------------------------
    # PERFORMANCE METRICS
    # --------------------------------------------------------
    st.subheader("📊 Performance Metrics")

    metrics = pd.DataFrame(
        {
            "Buy & Hold": compute_metrics(bh["Equity"]),
            f"Momentum (L={momentum_lb})": compute_metrics(mom["Equity"]),
            f"ARIMA({p},{d},{q})": compute_metrics(ari["Equity"]),
        }
    ).T

    st.dataframe(metrics.style.format({
        "Return": "{:.2%}", "Vol": "{:.2%}",
        "Sharpe": "{:.2f}", "MaxDD": "{:.2%}",
    }), use_container_width=True)

    # --------------------------------------------------------
    # VIEW RANGE FILTER
    # --------------------------------------------------------
    def filter_range(df_in):
        if view_range == "MAX":
            return df_in
        days = {"1W":7,"1M":30,"3M":90,"6M":180,"1Y":365,"5Y":365*5}[view_range]
        return df_in.iloc[-days:]

    df_v = filter_range(df)
    bh_v = filter_range(bh)
    mom_v = filter_range(mom)
    ari_v = filter_range(ari)

    # --------------------------------------------------------
    # PRICE CHART
    # --------------------------------------------------------
    st.subheader("📉 Price Chart")

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df_v.index, y=df_v["Adj Close"], name=f"{ticker} Price"))

    if ari_future is not None:
        fig_price.add_trace(go.Scatter(
            x=ari_future.index, y=ari_future["ForecastPrice"],
            name="ARIMA Forecast", line=dict(dash="dot")
        ))

    fig_price.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_price, use_container_width=True)

    # --------------------------------------------------------
    # PORTFOLIO VALUE
    # --------------------------------------------------------
    st.subheader(f"💰 Portfolio Value (Initial = ${initial_equity:,})")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bh_v.index, y=bh_v["Equity"], name="Buy & Hold"))
    fig.add_trace(go.Scatter(x=mom_v.index, y=mom_v["Equity"], name="Momentum"))
    fig.add_trace(go.Scatter(x=ari_v.index, y=ari_v["Equity"], name="ARIMA"))

    fig.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # HISTOGRAMS (NO REFRESH)
    # --------------------------------------------------------
    st.subheader("📊 Returns Distribution")

    show_bh = st.checkbox("Buy & Hold", True)
    show_mom = st.checkbox("Momentum", True)
    show_ari = st.checkbox("ARIMA", True)

    ret = pd.DataFrame({
        "BH": bh["Equity"].pct_change(),
        "Momentum": mom["Equity"].pct_change(),
        "ARIMA": ari["Equity"].pct_change(),
    }).dropna()

    fig_hist = go.Figure()
    if show_bh: fig_hist.add_trace(go.Histogram(x=ret["BH"], name="BH", opacity=0.6))
    if show_mom: fig_hist.add_trace(go.Histogram(x=ret["Momentum"], name="Momentum", opacity=0.6))
    if show_ari: fig_hist.add_trace(go.Histogram(x=ret["ARIMA"], name="ARIMA", opacity=0.6))

    fig_hist.update_layout(template="plotly_dark", barmode="overlay", height=320)
    st.plotly_chart(fig_hist, use_container_width=True)

    # --------------------------------------------------------
    # GRID SEARCH RESULTS (stored)
    # --------------------------------------------------------
    st.subheader("🏆 Grid Search Results")

    if st.session_state.gs_results is not None:

        # Momentum GS
        if "mom" in st.session_state.gs_results:
            mom_list, best_L, best_sharpe = st.session_state.gs_results["mom"]
            df_mom = pd.DataFrame(mom_list, columns=["L", "Sharpe"])
            st.write(f"⭐ Best Momentum L = {best_L} (Sharpe={best_sharpe:.3f})")
            st.line_chart(df_mom.set_index("L"))

        # ARIMA GS
        if "ari" in st.session_state.gs_results:
            ari_list, best_tuple, best_sharpe = st.session_state.gs_results["ari"]
            df_ari = pd.DataFrame(ari_list, columns=["p","d","q","Sharpe"])
            st.write(f"⭐ Best ARIMA = {best_tuple} (Sharpe={best_sharpe:.3f})")
            st.dataframe(df_ari.sort_values("Sharpe", ascending=False))

if __name__ == "__main__":
    main()
