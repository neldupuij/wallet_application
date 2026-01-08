import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

# -------------------------------------------------------------------
# Make sure the project root (wallet_application) is in sys.path
# -------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# -------------------------------------------------------------------
# Import the two Streamlit apps
# -------------------------------------------------------------------
from streamlit_quanta import main as quant_a_app
from ui import portfolio_page as quant_b_app

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Wallet Application",
    layout="wide",
    page_icon="📊",
)

# -------------------------------------------------------------------
# Simple router using session_state
# -------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"


def go_home():
    st.session_state.page = "home"


def go_quant_a():
    st.session_state.page = "quant_a"


def go_quant_b():
    st.session_state.page = "quant_b"


# -------------------------------------------------------------------
# Sidebar navigation
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    st.button("🏠 Home", use_container_width=True, on_click=go_home)
    st.button("🧠 Quant A — Single Asset", use_container_width=True, on_click=go_quant_a)
    st.button("🧳 Quant B — Portfolio", use_container_width=True, on_click=go_quant_b)

    st.markdown("---")
    st.caption("Wallet Application\nQuant Research & Portfolio Lab")


# -------------------------------------------------------------------
# Home helpers
# -------------------------------------------------------------------
def _ensure_series(x, name: str) -> pd.Series:
    """Ensure x is a 1D Series (yfinance can return DF with MultiIndex columns)."""
    if isinstance(x, pd.Series):
        s = x.copy()
        s.name = name
        return s

    if isinstance(x, pd.DataFrame) and not x.empty:
        # Flatten MultiIndex columns if needed
        if isinstance(x.columns, pd.MultiIndex):
            x2 = x.copy()
            x2.columns = [" ".join([str(i) for i in tup if i is not None]).strip() for tup in x.columns]
        else:
            x2 = x

        # Prefer price-like columns
        for pref in ["Adj Close", "Close"]:
            if pref in x2.columns and pd.api.types.is_numeric_dtype(x2[pref]):
                s = x2[pref].dropna().astype(float)
                s.name = name
                return s

        # Otherwise first numeric column
        num_cols = [c for c in x2.columns if pd.api.types.is_numeric_dtype(x2[c])]
        if num_cols:
            s = x2[num_cols[0]].dropna().astype(float)
            s.name = name
            return s

    return pd.Series(dtype=float, name=name)


@st.cache_data(show_spinner=False)
def _intraday_last_pct(symbol: str) -> float:
    """
    Return today's intraday % move based on yfinance:
    - intraday 5m over 1d (preferred)
    - fallback: daily last 2 closes (1d over 5d)
    Returns NaN if no data.
    """
    try:
        import yfinance as yf

        df = yf.download(symbol, period="1d", interval="5m", progress=False, auto_adjust=False)
        s = _ensure_series(df, symbol)
        if s is not None and not s.empty:
            first = float(s.iloc[0])
            last = float(s.iloc[-1])
            if first != 0:
                return (last / first) - 1.0
    except Exception:
        pass

    # fallback daily: last 2 closes
    try:
        import yfinance as yf

        df = yf.download(symbol, period="5d", interval="1d", progress=False, auto_adjust=False)
        s = _ensure_series(df, symbol)
        if s is not None and len(s) >= 2:
            prev = float(s.iloc[-2])
            last = float(s.iloc[-1])
            if prev != 0:
                return (last / prev) - 1.0
    except Exception:
        pass

    return float("nan")


def _market_clocks_block():
    now_utc = datetime.utcnow()

    def fmt(dt_obj):
        return dt_obj.strftime("%H:%M")

    clocks = [
        ("New York (NYSE)", -5),
        ("London (LSE)", 0),
        ("Paris (Euronext)", +1),
        ("Tokyo (TSE)", +9),
        ("Hong Kong (HKEX)", +8),
    ]

    cols = st.columns(len(clocks))
    for col, (name, offset) in zip(cols, clocks):
        local = now_utc + timedelta(hours=offset)
        col.metric(name, fmt(local))


def _delta_label(pct: float) -> str:
    """Arrow + label for Streamlit metric delta."""
    if pd.isna(pct):
        return ""
    return f"{'▼' if pct < 0 else '▲'} {pct:+.2%}"


# -------------------------------------------------------------------
# Pages
# -------------------------------------------------------------------
def render_home():
    st.markdown(
        """
        <h1 style="text-align:center;">📊 Wallet Application</h1>
        <h3 style="text-align:center; color: grey;">
        Quant Research & Portfolio Construction Lab
        </h3>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("## 🧠 Quant A")
        st.markdown(
            """
            **Single Asset Research Lab**

            - Analyse d’un actif unique  
            - Momentum & signaux  
            - Backtesting de stratégies  
            - Prévisions ARIMA  
            - Diagnostics statistiques  
            """
        )
        st.write("")
        if st.button("🚀 Open Quant A", use_container_width=True):
            st.session_state.page = "quant_a"

    with c2:
        st.markdown("## 🧳 Quant B")
        st.markdown(
            """
            **Portfolio Construction Engine**

            - Construction multi-actifs  
            - Allocation & pondérations  
            - Analyse du risque  
            - Performance globale  
            - Comparaison benchmarks  
            """
        )
        st.write("")
        if st.button("🚀 Open Quant B", use_container_width=True):
            st.session_state.page = "quant_b"

    st.markdown("---")

    # -------------------------
    # Market Snapshot (Intraday) - NO CHARTS + red down / green up
    # -------------------------
    st.subheader("📈 Market Snapshot (Intraday)")
    st.caption("Variation journalière (intraday 5m si dispo, sinon daily).")

    symbols = ["SPY", "AAPL", "MSFT", "NVDA", "TSLA"]
    cols = st.columns(len(symbols), gap="large")

    pct_map = {sym: _intraday_last_pct(sym) for sym in symbols}

    for col, sym in zip(cols, symbols):
        pct = pct_map.get(sym, float("nan"))
        if pd.isna(pct):
            col.metric(sym, "N/A", "No data")
        else:
            # value = absolute % (no sign), delta = signed with arrow
            value = f"{abs(pct):.2%}"
            delta = _delta_label(pct)
            # delta_color does exactly what you want: negative -> red (inverse) + down arrow in delta string
            col.metric(sym, value=value, delta=delta, delta_color="inverse")

    if all(pd.isna(pct_map[s]) for s in symbols):
        st.info("No market data available. Install yfinance (pip install yfinance) and check internet access.")

    st.markdown("---")

    # -------------------------
    # Market clocks
    # -------------------------
    st.subheader("🕒 Market Clocks")
    _market_clocks_block()
    st.caption("Times are approximate (UTC offsets, DST not handled).")


def render_quant_a():
    st.markdown("## 🧠 Quant A — Single Asset Backtesting")
    st.caption("Analyse, signaux, backtests et prévisions sur un actif unique.")
    st.markdown("---")
    quant_a_app()


def render_quant_b():
    st.markdown("## 🧳 Quant B — Multi-Asset Portfolio")
    st.caption("Construction de portefeuille, allocation, risque et performance.")
    st.markdown("---")
    quant_b_app()


# -------------------------------------------------------------------
# Router
# -------------------------------------------------------------------
if st.session_state.page == "home":
    render_home()
elif st.session_state.page == "quant_a":
    render_quant_a()
elif st.session_state.page == "quant_b":
    render_quant_b()
