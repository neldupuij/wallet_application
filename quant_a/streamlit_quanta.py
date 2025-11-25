import os
from datetime import datetime, timedelta
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

warnings.filterwarnings("ignore")   # <<< remove ARIMA warnings


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------
def compute_metrics(equity: pd.Series):
    equity = equity.squeeze()
    returns = equity.pct_change().dropna()

    if returns.empty or len(equity) < 2:
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
    st.set_page_config(
        page_title="Quant A – Backtesting Suite (Dark)",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📈 Quant A – Single Asset Backtesting")
    st.caption("Buy & Hold • Momentum • ARIMA • Grid Search • Optimal Parameters • Price View Selector")

    # ---------------- Sidebar ----------------
    st.sidebar.header("Parameters")

    base_tickers = ["AAPL", "MSFT", "NVDA", "GOOG", "TSLA", "SPY", "EURUSD=X", "BTC-USD"]
    choice = st.sidebar.selectbox("Choose asset", ["Custom ticker"] + base_tickers)

    if choice == "Custom ticker":
        ticker = st.sidebar.text_input("Enter ticker", "AAPL").upper()
    else:
        ticker = choice

    start_date = st.sidebar.date_input(
        "Start date",
        value=datetime(2015, 1, 1),
        format="YYYY/MM/DD",
    ).strftime("%Y-%m-%d")

    initial_equity = st.sidebar.number_input(
        "Initial investment",
        min_value=100,
        max_value=1_000_000_000,
        value=10_000,
        step=100
    )

    st.sidebar.subheader("Momentum")
    momentum_lb = st.sidebar.slider("Lookback (days)", 2, 120, 20)

    st.sidebar.subheader("ARIMA order (p,d,q)")
    p = st.sidebar.slider("p", 0, 5, 1)
    d = st.sidebar.slider("d", 0, 2, 0)
    q = st.sidebar.slider("q", 0, 5, 1)

    st.sidebar.subheader("Grid Search")
    gs_mom = st.sidebar.checkbox("Momentum Grid Search")
    gs_ari = st.sidebar.checkbox("ARIMA Grid Search")

    st.sidebar.subheader("View range")
    view_range = st.sidebar.selectbox(
        "Display period", ["1W", "1M", "3M", "6M", "1Y", "5Y", "MAX"], index=6
    )

    run_bt = st.sidebar.button("🚀 Run Backtests")

    if not run_bt:
        st.info("Configure parameters in the sidebar and click **Run Backtests**.")
        return

    # ---------------- Data loading ----------------
    try:
        df = download_ticker(ticker, start_date).asfreq("B")
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return

    if df.empty:
        st.error("Downloaded data is empty.")
        return

    # ---------------- Backtests ----------------
    bh = backtest_buy_and_hold(df).copy()
    mom = backtest_momentum(df, momentum_lb).copy()
    ari = backtest_arima(df, (p, d, q)).copy()

    # Scale to real money
    bh["Equity"] *= initial_equity
    mom["Equity"] *= initial_equity
    ari["Equity"] *= initial_equity

    # ----------- Time range filter -------------
    def filter_range(df):
        if view_range == "MAX":
            return df
        days = {
            "1W": 7,
            "1M": 30,
            "3M": 90,
            "6M": 180,
            "1Y": 365,
            "5Y": 365 * 5
        }[view_range]
        return df.iloc[-days:]

    df_v = filter_range(df)
    bh_v = filter_range(bh)
    mom_v = filter_range(mom)
    ari_v = filter_range(ari)

    # ---------------- Price Chart ----------------
    st.subheader("📉 Price Chart")

    fig_price = go.Figure()
    fig_price.add_trace(
        go.Scatter(
            x=df_v.index,
            y=df_v["Adj Close"],
            name=f"{ticker} Price",
            line=dict(color="#4c78ff", width=2),
        )
    )
    fig_price.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
        yaxis_title="Price",
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # ---------------- Portfolio Value Chart ----------------
    st.subheader(f"💰 Portfolio Value (Initial = ${initial_equity:,})")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bh_v.index, y=bh_v["Equity"], name="Buy & Hold", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=mom_v.index, y=mom_v["Equity"], name=f"Momentum L={momentum_lb}"))
    fig.add_trace(go.Scatter(x=ari_v.index, y=ari_v["Equity"], name=f"ARIMA({p},{d},{q})"))

    fig.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=40, r=20, t=40, b=40),
        yaxis_title="Portfolio Value ($)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------
    # MOMENTUM GRID SEARCH
    # ------------------------------------------------------------
    best_L = None
    best_L_sharpe = -999

    if gs_mom:
        st.subheader("🔥 Momentum Grid Search")

        results = []
        for L in range(2, 121):
            try:
                eq = backtest_momentum(df, L)["Equity"] * initial_equity
                sharpe = compute_metrics(eq)["Sharpe"]
                results.append((L, sharpe))
                if sharpe > best_L_sharpe:
                    best_L = L
                    best_L_sharpe = sharpe
            except:
                results.append((L, np.nan))

        mom_df = pd.DataFrame(results, columns=["Lookback", "Sharpe"]).set_index("Lookback")
        st.line_chart(mom_df)

    # ------------------------------------------------------------
    # ARIMA GRID SEARCH
    # ------------------------------------------------------------
    best_arima = None
    best_arima_sharpe = -999

    if gs_ari:
        st.subheader("🔥 ARIMA Grid Search (Sharpe)")

        results = []
        for P in range(0, 6):
            for D in range(0, 2):
                for Q in range(0, 6):
                    try:
                        eq = backtest_arima(df, (P, D, Q))["Equity"] * initial_equity
                        sharpe = compute_metrics(eq)["Sharpe"]
                        results.append((P, D, Q, sharpe))
                        if sharpe > best_arima_sharpe:
                            best_arima = (P, D, Q)
                            best_arima_sharpe = sharpe
                    except:
                        results.append((P, D, Q, np.nan))

        ari_df = pd.DataFrame(results, columns=["p", "d", "q", "Sharpe"])
        st.dataframe(ari_df.sort_values("Sharpe", ascending=False), use_container_width=True)

    # ------------------------------------------------------------
    # BEST PARAMETERS SUMMARY
    # ------------------------------------------------------------
    st.subheader("🏆 Optimal Parameters Summary")

    summary = []
    if best_L is not None:
        summary.append(["Momentum", f"L = {best_L}", best_L_sharpe])
    if best_arima is not None:
        summary.append(["ARIMA", str(best_arima), best_arima_sharpe])

    if summary:
        df_sum = pd.DataFrame(summary, columns=["Strategy", "Optimal Parameters", "Sharpe"])
        st.dataframe(df_sum.style.format({"Sharpe": "{:.4f}"}), use_container_width=True)
    else:
        st.info("No optimal parameters found.")

    # ---------------- CSV Export ----------------
    st.subheader("⬇ Download Results")

    out = pd.DataFrame(
        {
            "Price": df["Adj Close"],
            "BH": bh["Equity"],
            "Momentum": mom["Equity"],
            "ARIMA": ari["Equity"],
        }
    )

    st.download_button(
        "Download CSV",
        out.to_csv().encode(),
        file_name=f"{ticker}_portfolio_value.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
