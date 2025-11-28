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
    st.caption("Buy & Hold • Momentum • ARIMA • Grid Search • Forecast")

    # Sidebar ----------------
    st.sidebar.header("Parameters")

    tickers = ["AAPL", "MSFT", "NVDA", "GOOG", "TSLA", "SPY", "EURUSD=X", "BTC-USD"]
    choice = st.sidebar.selectbox("Choose asset", ["Custom ticker"] + tickers)

    ticker = st.sidebar.text_input(
        "Enter ticker",
        choice if choice != "Custom ticker" else "AAPL",
    ).upper()

    start_date = st.sidebar.date_input(
        "Start date",
        value=datetime(2015, 1, 1),
        format="YYYY/MM/DD",
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

    if not run_bt:
        st.info("Configure parameters then click Run.")
        return

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    try:
        df = download_ticker(ticker, start_date)
        df = df[~df.index.duplicated()].dropna()
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return

    if df.empty:
        st.error("Downloaded data is empty.")
        return

    # --------------------------------------------------------
    # Run Backtests
    # --------------------------------------------------------
    bh = backtest_buy_and_hold(df)
    mom = backtest_momentum(df, momentum_lb)
    ari, ari_future = backtest_arima(df, (p, d, q), forecast_horizon=30)

    # Scale portfolios
    bh["Equity"] *= initial_equity
    mom["Equity"] *= initial_equity
    ari["Equity"] *= initial_equity

    # --------------------------------------------------------
    # Performance Metrics
    # --------------------------------------------------------
    st.subheader("📊 Performance Metrics")

    metrics = pd.DataFrame(
        {
            "Buy & Hold": compute_metrics(bh["Equity"]),
            f"Momentum (L={momentum_lb})": compute_metrics(mom["Equity"]),
            f"ARIMA({p},{d},{q})": compute_metrics(ari["Equity"]),
        }
    ).T

    st.dataframe(
        metrics.style.format(
            {
                "Return": "{:.2%}",
                "Vol": "{:.2%}",
                "Sharpe": "{:.2f}",
                "MaxDD": "{:.2%}",
            }
        ),
        use_container_width=True,
    )

    # --------------------------------------------------------
    # View range filtering
    # --------------------------------------------------------
    def filter_range(df_in: pd.DataFrame) -> pd.DataFrame:
        if view_range == "MAX":
            return df_in

        days = {
            "1W": 7, "1M": 30, "3M": 90, "6M": 180,
            "1Y": 365, "5Y": 365 * 5,
        }[view_range]

        return df_in.iloc[-days:]

    df_v = filter_range(df)
    bh_v = filter_range(bh)
    mom_v = filter_range(mom)
    ari_v = filter_range(ari)

    # --------------------------------------------------------
    # Price chart
    # --------------------------------------------------------
    st.subheader("📉 Price Chart")

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=df_v.index,
        y=df_v["Adj Close"],
        name=f"{ticker} Price",
        line=dict(color="#4c78ff", width=2),
    ))

    if ari_future is not None:
        fig_price.add_trace(go.Scatter(
            x=ari_future.index,
            y=ari_future["ForecastPrice"],
            name="ARIMA Forecast 30d",
            mode="lines",
            line=dict(color="orange", width=2, dash="dot"),
        ))

    fig_price.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Date",
        yaxis_title="Price",
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # --------------------------------------------------------
    # Portfolio Value Chart
    # --------------------------------------------------------
    st.subheader(f"💰 Portfolio Value (Initial = ${initial_equity:,})")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bh_v.index, y=bh_v["Equity"], name="Buy & Hold"))
    fig.add_trace(go.Scatter(x=mom_v.index, y=mom_v["Equity"], name=f"Momentum L={momentum_lb}"))
    fig.add_trace(go.Scatter(x=ari_v.index, y=ari_v["Equity"], name=f"ARIMA({p},{d},{q})"))

    fig.update_layout(
        template="plotly_dark",
        height=450,
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # NEW: DAILY RETURNS GRAPH
    # --------------------------------------------------------
    st.subheader("📉 Daily Returns Comparison")

    returns_df = pd.DataFrame({
        "BH_Returns": bh["Equity"].pct_change(),
        f"Momentum_Returns_L={momentum_lb}": mom["Equity"].pct_change(),
        f"ARIMA_Returns_{p}_{d}_{q}": ari["Equity"].pct_change(),
    }).dropna()

    fig_ret = go.Figure()

    for col in returns_df.columns:
        fig_ret.add_trace(
            go.Scatter(
                x=returns_df.index,
                y=returns_df[col],
                name=col,
                mode="lines",
                line=dict(width=1)
            )
        )

    fig_ret.update_layout(
        template="plotly_dark",
        height=350,
        xaxis_title="Date",
        yaxis_title="Daily Returns",
    )

    st.plotly_chart(fig_ret, use_container_width=True)

    # --------------------------------------------------------
    # Grid Search
    # --------------------------------------------------------
    st.subheader("🏆 Optimal Parameters Summary")

    best_L = None
    best_L_sharpe = -999
    best_arima = None
    best_arima_sharpe = -999

    # Momentum Grid Search
    if gs_mom:
        st.subheader("🔥 Momentum Grid Search")
        results = []

        for L in range(2, 121):
            eq = backtest_momentum(df, L)["Equity"]
            sharpe = compute_metrics(eq)["Sharpe"]
            results.append((L, sharpe))
            if sharpe > best_L_sharpe:
                best_L = L
                best_L_sharpe = sharpe

        st.line_chart(pd.DataFrame(results, columns=["L", "Sharpe"]).set_index("L"))

    # ARIMA Grid Search
    if gs_ari:
        st.subheader("🔥 ARIMA Grid Search")
        results = []

        for P in range(0, 6):
            for D in range(0, 2):
                for Q in range(0, 6):
                    eq_df, _ = backtest_arima(df, (P, D, Q))
                    sharpe = compute_metrics(eq_df["Equity"])["Sharpe"]
                    results.append((P, D, Q, sharpe))
                    if sharpe > best_arima_sharpe:
                        best_arima = (P, D, Q)
                        best_arima_sharpe = sharpe

        st.dataframe(
            pd.DataFrame(results, columns=["p", "d", "q", "Sharpe"]).sort_values(
                "Sharpe", ascending=False
            ),
            use_container_width=True,
        )

    # Summary Table
    summary = []
    if best_L is not None:
        summary.append(["Momentum", f"L={best_L}", best_L_sharpe])
    if best_arima is not None:
        summary.append(["ARIMA", str(best_arima), best_arima_sharpe])

    if summary:
        st.dataframe(
            pd.DataFrame(summary, columns=["Strategy", "Optimal Parameters", "Sharpe"]),
            use_container_width=True,
        )
    else:
        st.info("No optimal parameters found.")

    # --------------------------------------------------------
    # CSV Export
    # --------------------------------------------------------
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
