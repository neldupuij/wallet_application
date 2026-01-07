import os
import datetime
import pandas as pd
from download_data import download_ticker
from metrics import annualized_return, annualized_volatility, max_drawdown

WATCHLIST = ["AAPL", "BTC-USD", "EURUSD=X", "MSFT"]
REPORT_DIR = "reports"

def generate_daily_report():
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)

    today_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = [
        "========================================",
        f"DAILY FINANCIAL REPORT - {today_str}",
        "========================================",
        ""
    ]

    for ticker in WATCHLIST:
        print(f"Processing {ticker}...")
        try:
            # Télécharge les données 
            df = download_ticker(ticker, "2023-01-01") 
            if df.empty:
                report_lines.append(f"[{ticker}] : NO DATA FOUND")
                continue
                
            prices = df["Adj Close"]
            last_price = prices.iloc[-1]
            # Pour l'Open, on prend l'avant-dernier jour ou l'open du jour si dispo
            # Ici on simplifie avec le prix de la veille
            prev_price = prices.iloc[-2] if len(prices) > 1 else last_price
            
            # Métriques
            ann_ret = annualized_return(prices)
            ann_vol = annualized_volatility(prices)
            # MaxDD sur la période téléchargée
            equity = prices / prices.iloc[0]
            max_dd = max_drawdown(equity)

            report_lines.append(f"Asset: {ticker}")
            report_lines.append(f"  > Price:       ${last_price:.2f} (Prev: ${prev_price:.2f})")
            report_lines.append(f"  > Return (Ann): {ann_ret:.2%}")
            report_lines.append(f"  > Volatility:   {ann_vol:.2%}")
            report_lines.append(f"  > Max Drawdown: {max_dd:.2%}")
            report_lines.append("-" * 30)

        except Exception as e:
            report_lines.append(f"[{ticker}] Error: {str(e)}")

    report_lines.append("")
    report_lines.append("End of Report")

    filename = f"report_{datetime.datetime.now().strftime('%Y%m%d')}.txt"
    filepath = os.path.join(REPORT_DIR, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"Rapport généré : {filepath}")

if __name__ == "__main__":
    generate_daily_report()