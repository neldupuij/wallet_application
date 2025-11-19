import pandas as pd
import numpy as np
import os

class PerformanceMetrics:
    """
    Compute and export basic performance metrics for a single asset.
    """

    def __init__(self, df: pd.DataFrame, ticker: str = "TICKER"):
        if "Adj Close" not in df.columns:
            raise ValueError("DataFrame must contain an 'Adj Close' column.")
        self.ticker = ticker.upper()
        self.df = df.copy()
        self.df["Return"] = self.df["Adj Close"].pct_change()
        self.df.dropna(inplace=True)

    def cumulative_return(self):
        """Compute cumulative return of a buy-and-hold strategy."""
        self.df["Cumulative"] = (1 + self.df["Return"]).cumprod()
        return self.df

    def annualized_metrics(self, risk_free_rate=0.0, save_csv=True):
        """Compute and optionally save annualized metrics."""
        mean_daily = self.df["Return"].mean()
        std_daily = self.df["Return"].std()

        ann_return = (1 + mean_daily) ** 252 - 1
        ann_vol = std_daily * np.sqrt(252)
        sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol != 0 else np.nan

        cum_returns = (1 + self.df["Return"]).cumprod()
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_dd = drawdown.min()

        metrics = {
            "Ticker": self.ticker,
            "Annual Return": ann_return,
            "Annual Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd,
        }

        metrics_df = pd.DataFrame([metrics])

        if save_csv:
            save_dir = os.path.join("quant_a", "data")
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.join(save_dir, f"METRICS_{self.ticker}.csv")
            metrics_df.to_csv(filename, index=False)
            print(f"✅ Metrics saved to {filename}")

        return metrics
