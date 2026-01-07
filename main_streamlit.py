import os
import sys
import time
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
# Hub Streamlit
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Wallet Application", layout="wide")

    # --- AUTO-REFRESH LOGIC (Exigence CDC: Refresh every 5 min) ---
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()

    # Si plus de 300 secondes (5 minutes) se sont écoulées
    if time.time() - st.session_state.last_refresh > 300:
        st.session_state.last_refresh = time.time()
        st.rerun()
    # --------------------------------------------------------------

    st.sidebar.title("🔀 Application Selector")
    choice = st.sidebar.radio(
        "Choose an application:",
        ("Quant A – Single Asset Backtesting", "Quant B – Multi-Asset Portfolio")
    )

    st.title("📊 Wallet Application – Streamlit Hub")

    if choice.startswith("Quant A"):
        st.header("Quant A – Single Asset Backtesting")
        quant_a_app()

    elif choice.startswith("Quant B"):
        st.header("Quant B – Multi-Asset Portfolio Simulation")
        quant_b_app()


if __name__ == "__main__":
    main()