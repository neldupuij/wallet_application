import os
import sys
import subprocess
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

from quant_a.download_data import download_ticker
from quant_a.backtest import (
    backtest_buy_and_hold,
    backtest_momentum,
    backtest_arima,
)


def compute_strategy_metrics(equity: pd.Series) -> dict:
    """Compute basic performance metrics from an equity curve (start = 1)."""
    equity = equity.dropna()
    if equity.empty:
        return {
            "final_equity": np.nan,
            "annual_return": np.nan,
            "annual_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown_pct": np.nan,
        }

    rets = equity.pct_change().dropna()
    if rets.empty:
        return {
            "final_equity": float(equity.iloc[-1]),
            "annual_return": 0.0,
            "annual_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown_pct": 0.0,
        }

    mean_ret = rets.mean()
    vol = rets.std()

    ann_ret = (1 + mean_ret) ** 252 - 1
    ann_vol = vol * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    max_dd = dd.min()

    return {
        "final_equity": float(equity.iloc[-1]),
        "annual_return": float(ann_ret),
        "annual_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown_pct": float(max_dd * 100.0),
    }


def restrict_range(df: pd.DataFrame, range_key: str) -> pd.DataFrame:
    """Coupe la série selon Max / 1Y / 6M / 3M / 1M / 5D."""
    if df.empty:
        return df
    end = df.index.max()
    if range_key == "Max":
        return df
    mapping_days = {
        "1Y": 252,
        "6M": 126,
        "3M": 63,
        "1M": 21,
        "5D": 5,
    }
    days = mapping_days.get(range_key, None)
    if days is None:
        return df
    start = end - pd.Timedelta(days=days * 1.5)
    return df.loc[df.index >= start]


def get_ohlc(ticker: str, start_date: str) -> pd.DataFrame:
    """Télécharge OHLC pour les chandeliers."""
    data = yf.download(ticker, start=start_date, interval="1d", auto_adjust=False)
    if data.empty:
        return data
    return data[["Open", "High", "Low", "Close"]]


def get_data_dir() -> str:
    """Retourne le chemin du dossier quant_a/data/."""
    here = os.path.dirname(__file__)
    data_dir = os.path.join(here, "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def load_universe() -> pd.DataFrame:
    """
    Charge la liste de tickers depuis quant_a/data/universe.csv.
    Si le fichier n'existe pas, renvoie un petit univers par défaut.
    """
    data_dir = get_data_dir()
    csv_path = os.path.join(data_dir, "universe.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if "ticker" in df.columns:
                if "name" not in df.columns:
                    df["name"] = ""
                return df[["ticker", "name"]]
        except Exception:
            pass

    data = [
        ("AAPL", "Apple Inc."),
        ("MSFT", "Microsoft Corp."),
        ("GOOGL", "Alphabet Class A"),
        ("AMZN", "Amazon.com Inc."),
        ("META", "Meta Platforms"),
        ("NVDA", "NVIDIA Corp."),
        ("TSLA", "Tesla Inc."),
        ("SPY", "SPDR S&P 500 ETF"),
        ("VOO", "Vanguard S&P 500 ETF"),
        ("^GSPC", "S&P 500 Index"),
        ("QQQ", "Invesco QQQ"),
        ("IWM", "Russell 2000 ETF"),
        ("GLD", "SPDR Gold Shares"),
        ("GC=F", "Gold Futures"),
        ("SI=F", "Silver Futures"),
        ("CL=F", "Crude Oil WTI"),
        ("EURUSD=X", "EUR/USD"),
        ("GBPUSD=X", "GBP/USD"),
        ("USDJPY=X", "USD/JPY"),
    ]
    return pd.DataFrame(data, columns=["ticker", "name"])


def main():
    st.set_page_config(
        page_title="Quant A – Single Asset Backtester",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Quant A – Single Asset Analysis")

    if "momentum_L" not in st.session_state:
        st.session_state["momentum_L"] = 20
    if "arima_p" not in st.session_state:
        st.session_state["arima_p"] = 2
    if "arima_d" not in st.session_state:
        st.session_state["arima_d"] = 0
    if "arima_q" not in st.session_state:
        st.session_state["arima_q"] = 1
    if "best_grid_df" not in st.session_state:
        st.session_state["best_grid_df"] = None
    if "best_momentum_L" not in st.session_state:
        st.session_state["best_momentum_L"] = None
    if "best_arima" not in st.session_state:
        st.session_state["best_arima"] = None

    with st.sidebar:
        st.header("Parameters")

        universe_df = load_universe()
        universe_df["label"] = universe_df["ticker"] + " — " + universe_df["name"].fillna("")

        mode = st.radio(
            "Ticker selection",
            options=["From universe", "Manual"],
            index=0,
        )

        if mode == "From universe":
            selected_label = st.selectbox(
                "Universe (searchable)",
                options=universe_df["label"].tolist(),
            )
            ticker = universe_df.loc[
                universe_df["label"] == selected_label, "ticker"
            ].iloc[0]
            manual_override = st.text_input(
                "Override / custom ticker (optional)", value=""
            ).upper()
            if manual_override:
                ticker = manual_override
        else:
            ticker = st.text_input("Ticker", value="AAPL").upper()

        years_history = st.slider("Years of history", 1, 20, 10)

        st.markdown("Momentum lookback (days)")
        L = st.slider(
            "Momentum lookback (days)",
            min_value=3,
            max_value=120,
            value=st.session_state["momentum_L"],
            step=1,
            key="momentum_L",
            label_visibility="collapsed",
        )

        st.markdown("ARIMA parameters (p, d, q)")
        p = st.number_input(
            "p", min_value=0, max_value=5, value=st.session_state["arima_p"], step=1, key="arima_p"
        )
        d = st.number_input(
            "d", min_value=0, max_value=2, value=st.session_state["arima_d"], step=1, key="arima_d"
        )
        q = st.number_input(
            "q", min_value=0, max_value=5, value=st.session_state["arima_q"], step=1, key="arima_q"
        )

        st.markdown("Display range")
        display_range = st.radio(
            "",
            options=["Max", "1Y", "6M", "3M", "1M", "5D"],
            index=0,
            horizontal=True,
            key="display_range",
        )

        chart_type = st.radio(
            "Chart type",
            options=["Line", "Candlestick"],
            index=0,
        )

        run_backtests = st.button("Run backtests", type="primary")

        st.markdown("---")
        optimize_clicked = st.button("Optimize parameters (grid search)")

    if not ticker:
        st.warning("Please enter a ticker.")
        return

    start_date = (date.today() - timedelta(days=years_history * 365)).isoformat()

    with st.spinner(f"Downloading data for {ticker}…"):
        try:
            df_price = download_ticker(ticker, start_date)
        except Exception as e:
            st.error(f"Error while downloading data for {ticker}: {e}")
            return

    if df_price.empty:
        st.error("No data downloaded for this ticker / period.")
        return

    ohlc = get_ohlc(ticker, start_date)

    if run_backtests or "bh_df" not in st.session_state:
        with st.spinner("Running backtests…"):
            bh_df = backtest_buy_and_hold(df_price, ticker)
            mom_df = backtest_momentum(df_price, ticker, lookback=int(L))
            arima_df = backtest_arima(df_price, ticker, order=(int(p), int(d), int(q)))

        st.session_state["bh_df"] = bh_df
        st.session_state["mom_df"] = mom_df
        st.session_state["arima_df"] = arima_df
    else:
        bh_df = st.session_state["bh_df"]
        mom_df = st.session_state["mom_df"]
        arima_df = st.session_state["arima_df"]

    metrics_rows = []

    bh_equity = bh_df["Equity"]
    m_bh = compute_strategy_metrics(bh_equity)
    m_bh["strategy"] = "Buy & Hold"
    metrics_rows.append(m_bh)

    mom_equity = mom_df["Equity"]
    m_mom = compute_strategy_metrics(mom_equity)
    m_mom["strategy"] = f"Momentum (L={int(L)})"
    metrics_rows.append(m_mom)

    arima_equity = arima_df["Equity"]
    m_arima = compute_strategy_metrics(arima_equity)
    m_arima["strategy"] = f"ARIMA({int(p)}, {int(d)}, {int(q)})"
    metrics_rows.append(m_arima)

    metrics_df = pd.DataFrame(metrics_rows)[
        [
            "strategy",
            "final_equity",
            "annual_return",
            "annual_vol",
            "sharpe",
            "max_drawdown_pct",
        ]
    ]

    st.subheader("Strategy metrics")

    st.dataframe(
        metrics_df.style.format(
            {
                "final_equity": "{:.2f}",
                "annual_return": "{:.2%}",
                "annual_vol": "{:.2%}",
                "sharpe": "{:.3f}",
                "max_drawdown_pct": "{:.1f}%",
            },
            na_rep="–",
        ),
        use_container_width=True,
    )

    if optimize_clicked:
        with st.spinner("Running grid search (this can take some time)…"):
            try:
                cmd = [sys.executable, "-m", "quant_a.grid_search", ticker, start_date]
                subprocess.run(cmd, check=True)

                data_dir = get_data_dir()
                grid_path = os.path.join(data_dir, f"GRIDSEARCH_{ticker}.csv")
                if not os.path.exists(grid_path):
                    st.error(f"Grid search CSV not found at {grid_path}")
                else:
                    grid_df = pd.read_csv(grid_path)
                    st.session_state["best_grid_df"] = grid_df

                    mom_mask = grid_df["strategy"] == "momentum"
                    if mom_mask.any():
                        best_mom = grid_df.loc[mom_mask].sort_values(
                            "sharpe", ascending=False
                        ).iloc[0]
                        st.session_state["best_momentum_L"] = int(best_mom["lookback"])

                    ar_mask = grid_df["strategy"] == "arima"
                    if ar_mask.any():
                        best_ar = grid_df.loc[ar_mask].sort_values(
                            "sharpe", ascending=False
                        ).iloc[0]
                        st.session_state["best_arima"] = (
                            int(best_ar["p"]),
                            int(best_ar["d"]),
                            int(best_ar["q"]),
                        )
            except subprocess.CalledProcessError as e:
                st.error(f"Error while running grid_search: {e}")

    if st.session_state.get("best_grid_df") is not None:
        st.subheader("Grid search results (top by Sharpe)")
        grid_df = st.session_state["best_grid_df"].copy()
        grid_df_sorted = grid_df.sort_values("sharpe", ascending=False).head(15)
        st.dataframe(grid_df_sorted, use_container_width=True)

        if st.session_state.get("best_momentum_L") is not None:
            st.markdown(
                f"**Best Momentum:** L = {st.session_state['best_momentum_L']}"
            )
        if st.session_state.get("best_arima") is not None:
            bp, bd, bq = st.session_state["best_arima"]
            st.markdown(f"**Best ARIMA:** ({bp}, {bd}, {bq})")

        if st.button("Apply best parameters to controls"):
            if st.session_state.get("best_momentum_L") is not None:
                st.session_state["momentum_L"] = int(
                    st.session_state["best_momentum_L"]
                )
            if st.session_state.get("best_arima") is not None:
                bp, bd, bq = st.session_state["best_arima"]
                st.session_state["arima_p"] = int(bp)
                st.session_state["arima_d"] = int(bd)
                st.session_state["arima_q"] = int(bq)
            st.experimental_rerun()

    st.subheader(f"{ticker} — Price vs Strategies")

    price = df_price["Adj Close"]
    price_norm = price / price.iloc[0]

    bh_plot = restrict_range(bh_df.copy(), display_range)
    mom_plot = restrict_range(mom_df.copy(), display_range)
    arima_plot = restrict_range(arima_df.copy(), display_range)
    price_plot = restrict_range(price_norm.to_frame("Price_norm"), display_range)

    fig = go.Figure()

    if chart_type == "Candlestick" and not ohlc.empty:
        ohlc_plot = restrict_range(ohlc.copy(), display_range)

        fig.add_trace(
            go.Candlestick(
                x=ohlc_plot.index,
                open=ohlc_plot["Open"],
                high=ohlc_plot["High"],
                low=ohlc_plot["Low"],
                close=ohlc_plot["Close"],
                name="Price (candles)",
                opacity=0.8,
            )
        )

        base_price = ohlc_plot["Close"].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=bh_plot.index,
                y=bh_plot["Equity"] * base_price,
                name="Buy & Hold",
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=mom_plot.index,
                y=mom_plot["Equity"] * base_price,
                name=f"Momentum (L={int(L)})",
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=arima_plot.index,
                y=arima_plot["Equity"] * base_price,
                name=f"ARIMA({int(p)}, {int(d)}, {int(q)})",
                mode="lines",
            )
        )

        fig.update_yaxes(title="Price / Equity")
    else:
        fig.add_trace(
            go.Scatter(
                x=price_plot.index,
                y=price_plot["Price_norm"],
                name="Normalized Price (Adj Close)",
                mode="lines",
                line=dict(color="lightgray"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=bh_plot.index,
                y=bh_plot["Equity"],
                name="Buy & Hold",
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=mom_plot.index,
                y=mom_plot["Equity"],
                name=f"Momentum (L={int(L)})",
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=arima_plot.index,
                y=arima_plot["Equity"],
                name=f"ARIMA({int(p)}, {int(d)}, {int(q)})",
                mode="lines",
            )
        )

        fig.update_yaxes(title="Normalized value (start = 1)")

    fig.update_layout(
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=650,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Export results")

    equity_df = pd.DataFrame(
        {
            "BH_Equity": bh_equity,
            "Momentum_Equity": mom_equity,
            "ARIMA_Equity": arima_equity,
        }
    )
    equity_csv = equity_df.to_csv(index=True).encode("utf-8")
    metrics_csv = metrics_df.to_csv(index=False).encode("utf-8")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download equity curves (CSV)",
            data=equity_csv,
            file_name=f"{ticker}_equity_curves.csv",
            mime="text/csv",
        )
    with col2:
        st.download_button(
            "Download strategy metrics (CSV)",
            data=metrics_csv,
            file_name=f"{ticker}_strategy_metrics.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
