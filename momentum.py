import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


class MomentumStrategy:
    # Initialize the strategy configuration and placeholders for results.
    def __init__(
        self,ticker,start,end,bias=False,tf="1d",MA=20,fees=0.005,leverage=1.0,
    ):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.bias = bias
        self.tf = tf
        self.ma = MA
        self.fees = fees
        self.leverage = leverage

        self.data = pd.DataFrame()
        self.summary = pd.DataFrame()

    # Download historical market data from Yahoo Finance and normalize column names.
    def fetch_data(self):
        data = yf.download(
            self.ticker,
            start=self.start,
            end=self.end,
            interval=self.tf,
            auto_adjust=True,
            progress=True,
        )

        if data.empty:
            raise ValueError(f"No Yahoo Finance data returned for {self.ticker}.")

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.rename(columns=str.lower)
        data.index = pd.to_datetime(data.index)
        self.data = data
        return self.data

    @staticmethod
    # Estimate the number of observations per year from the timestamp spacing.
    def _periods_per_year(index):
        if len(index) < 2:
            return np.nan

        deltas = index.to_series().diff().dropna().dt.total_seconds()
        if deltas.empty:
            return np.nan

        median_seconds = deltas.median()
        if median_seconds <= 0:
            return np.nan

        return (365.25 * 24 * 60 * 60) / median_seconds

    @staticmethod
    # Compute rolling annualized volatility on a time window such as 365D or 30D.
    def _rolling_annualized_vol(log_returns, window, min_periods):
        rolling_std = log_returns.rolling(window=window, min_periods=min_periods).std()
        rolling_count = log_returns.rolling(window=window, min_periods=min_periods).count()

        timestamp_seconds = pd.Series(
            log_returns.index.view("int64") / 1_000_000_000,
            index=log_returns.index,
        )
        rolling_start_seconds = timestamp_seconds.rolling(
            window=window, min_periods=min_periods
        ).min()
        elapsed_days = ((timestamp_seconds - rolling_start_seconds) / (24 * 60 * 60)).clip(
            lower=1 / 24
        )

        periods_per_year = rolling_count / (elapsed_days / 365.25)
        return rolling_std * np.sqrt(periods_per_year)

    # Build the core dataset: returns, moving average, signal, volatility, and position size.
    def prepare_data(self):
        if self.data.empty:
            self.fetch_data()

        df = self.data.copy()
        df["return"] = df["close"].pct_change()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["ma"] = df["close"].rolling(self.ma).mean()

        df["signal"] = np.where(df["close"] > df["ma"], 1.0, -1.0)
        df.loc[df["ma"].isna(), "signal"] = np.nan

        if self.bias:
            df["signal"] = df["signal"].replace(-1.0, 0.0)

        df["annual_vol"] = self._rolling_annualized_vol(
            df["log_return"],
            window="365D",
            min_periods=200,
        )
        df["recent_vol"] = self._rolling_annualized_vol(
            df["log_return"],
            window="30D",
            min_periods=5,
        )

        df["position"] = df["signal"] * (df["annual_vol"] / df["recent_vol"])
        df.loc[~np.isfinite(df["position"]), "position"] = np.nan
        df["position"] = df["position"] * self.leverage

        return self._compute_strategy(df)

    # Convert positions into gross and net strategy returns, then compute summary statistics.
    def _compute_strategy(self, df):
        periods_per_year = self._periods_per_year(df.index)
        if not np.isfinite(periods_per_year):
            raise ValueError("Not enough data points to estimate annualization.")

        df["position_prev"] = df["position"].shift(1).fillna(0.0)
        df["asset_simple_return"] = np.expm1(df["log_return"]).fillna(0.0)
        df["gross_strategy_return"] = df["position_prev"] * df["asset_simple_return"]

        df["turnover"] = (df["position"].fillna(0.0) - df["position_prev"]).abs()
        df["fee_cost"] = df["turnover"] * self.fees
        df["net_strategy_return"] = df["gross_strategy_return"] - df["fee_cost"]
        df["net_strategy_return"] = df["net_strategy_return"].clip(lower=-0.999999)
        df["net_log_return"] = np.log1p(df["net_strategy_return"])

        df["wealth"] = np.exp(df["net_log_return"].fillna(0.0).cumsum())
        df["cum_fees"] = df["fee_cost"].fillna(0.0).cumsum()

        active_mask = df["position_prev"] != 0
        active_returns = df.loc[active_mask, "net_strategy_return"].dropna()
        active_log_returns = df.loc[active_mask, "net_log_return"].dropna()

        if active_returns.empty:
            win_rate = np.nan
        else:
            win_rate = (active_returns > 0).mean()

        if active_log_returns.empty:
            avg_return_factor = np.nan
        else:
            avg_return_factor = np.exp(active_log_returns.mean())

        if active_log_returns.empty:
            sharpe = np.nan
        else:
            ret_std = active_log_returns.std(ddof=1)
            sharpe = (
                (active_log_returns.mean() / ret_std) * np.sqrt(periods_per_year)
                if pd.notna(ret_std) and ret_std > 0
                else np.nan
            )

        summary = pd.DataFrame(
            {
                "ticker": [self.ticker],
                "start": [pd.to_datetime(self.start)],
                "end": [pd.to_datetime(self.end)],
                "bias": [self.bias],
                "tf": [self.tf],
                "ma": [self.ma],
                "fees": [self.fees],
                "leverage": [self.leverage],
                "total_pnl": [df["wealth"].iloc[-1] - 1 if not df.empty else np.nan],
                "total_fees": [df["fee_cost"].sum()],
                "winrate": [win_rate],
                "average_return_factor": [avg_return_factor],
                "sharpe_ratio_annualized": [sharpe],
            }
        )

        self.data = df
        self.summary = summary
        return self.data

    # Run the full strategy pipeline from data preparation to final results.
    def run(self):
        return self.prepare_data()

    # Plot the strategy wealth curve on a log scale using a seaborn-style chart.
    def plot_wealth(self):
        if self.data.empty:
            self.run()

        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.data.index, self.data["wealth"], label=f"{self.ticker} wealth")
        ax.set_yscale("log")
        ax.set_title(f"{self.ticker} Momentum Strategy Wealth")
        ax.set_xlabel("Date")
        ax.set_ylabel("Wealth (log scale)")
        ax.legend()
        plt.tight_layout()
        plt.show()
        return fig, ax
    

# Quick start:
# 1. Create and activate a local environment.
# 2. Install the libraries from requirements.txt.
# 3. Run the summary command below.
# 4. Run the plotting command below if you want to export a chart.
#
# Example usage:
# source .venv/bin/activate
# python -c "from momentum import MomentumStrategy; print('import ok')"
# python -c "from momentum import MomentumStrategy; s = MomentumStrategy(ticker='SPY', start='2020-01-01', end='2025-01-01', bias=True, tf='1d', MA=50, fees=0.0005, leverage=1.0); s.run(); print(s.summary.to_string(index=False))"
# python -c "from momentum import MomentumStrategy; import matplotlib.pyplot as plt; s = MomentumStrategy(ticker='SPY', start='2020-01-01', end='2025-01-01', bias=True, tf='1d', MA=50, fees=0.0005, leverage=1.0); s.run(); s.plot_wealth(); plt.savefig('wealth.png'); print('saved wealth.png')"