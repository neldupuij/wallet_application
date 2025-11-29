import datetime as dt

import pandas as pd
import streamlit as st
import os
import sys

# Add project root (wallet_application) to sys.path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from quant_B.data_loader import download_prices
from quant_B.weights import asset_metrics, equal_weight, normalize_weights
from quant_B.simulation import simulate_portfolio
from quant_B.reporting import portfolio_metrics, build_correlation_matrix


def portfolio_page():
    st.header("Quant B – Multi-Asset Portfolio Simulation")

    # ---------- User inputs ----------
    default_tickers = "AAPL, MSFT, GOOG"
    tickers_str = st.text_input(
        "Asset tickers (comma-separated)",
        value=default_tickers,
        help="Example: AAPL, MSFT, GOOG, AMZN, META",
    )
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=dt.date(2020, 1, 1))
    with col2:
        end_date = st.date_input("End date", value=dt.date.today())

    if len(tickers) < 3:
        st.warning("Please enter at least 3 different assets.")
        return

    # ---------- Download data ----------
    with st.spinner("Downloading price data..."):
        prices = download_prices(
            tickers,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
        )

    if prices.empty:
        st.error("No data downloaded. Check tickers or dates.")
        return

    st.subheader("Raw Prices (last 10 rows)")
    st.dataframe(prices.tail(10))

    # ---------- Strategy parameters ----------
    st.subheader("Portfolio Strategy Parameters")

    colw1, colw2 = st.columns(2)
    with colw1:
        allocation_mode = st.radio(
            "Allocation rule",
            options=["Equal weight", "Custom weights"],
            index=0,
        )
    with colw2:
        freq_label, rebalance_freq = st.selectbox(
            "Rebalancing frequency",
            options=[("Daily", "D"), ("Weekly", "W"), ("Monthly", "M")],
            format_func=lambda x: x[0],
        )

    if allocation_mode == "Equal weight":
        weights = equal_weight(tickers)
    else:
        st.markdown("### Custom weights")
        raw_weights = {}
        for t in tickers:
            raw_weights[t] = st.slider(
                f"Weight for {t}",
                min_value=0.0,
                max_value=1.0,
                value=1.0 / len(tickers),
                step=0.01,
            )

        try:
            weights = normalize_weights(raw_weights)
        except ValueError as e:
            st.error(str(e))
            return

    st.write("**Final normalized weights:**")
    st.write(weights)

    # ---------- Simulation ----------
    port_rets, port_equity = simulate_portfolio(prices, weights, rebalance_freq)

    # ---------- Metrics ----------
    st.subheader("Portfolio & Asset Metrics")

    asset_stats = asset_metrics(prices)
    port_stats = portfolio_metrics(port_rets, port_equity, asset_stats, weights)

    colm1, colm2 = st.columns(2)
    with colm1:
        st.markdown("**Asset Metrics**")
        st.dataframe(asset_stats)

    with colm2:
        st.markdown("**Portfolio Metrics**")
        st.json({k: float(v) for k, v in port_stats.items()})

    # Correlation matrix
    st.markdown("### Correlation Matrix")
    corr = build_correlation_matrix(prices)
    st.dataframe(corr.style.background_gradient(cmap="RdBu_r"))

    # ---------- Main chart: assets vs portfolio ----------
    st.subheader("Price vs Portfolio Value (normalized)")

    norm_prices = prices / prices.iloc[0]
    norm_port = port_equity / port_equity.iloc[0]
    norm_port.name = "Portfolio"

    chart_df = pd.concat([norm_prices, norm_port], axis=1)
    st.line_chart(chart_df)

    st.caption(
        "All asset prices and the portfolio value are normalized to 1 at the start date."
    )

if __name__ == "__main__":
    portfolio_page()
