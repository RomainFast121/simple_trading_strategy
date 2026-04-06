import numpy as np
import pandas as pd

from utils import (
    calculate_monte_carlo_performance,
    calculate_performance,
    fetch_data,
    log_return,
    plot_monte_carlo_wealth,
    plot_wealth,
    rolling_annualized_vol,
)


class MomentumStrategy:
    # Initialize the strategy inputs and prepare placeholders for raw data and results.
    def __init__(
        self,
        ticker,
        start,
        end,
        bias=False,
        tf="1d",
        MA=20,
        fees=0.005,
        leverage=1.0,
    ):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.bias = bias
        self.tf = tf
        self.ma = MA
        self.fees = fees
        self.leverage = leverage

        self.raw_data = pd.DataFrame()
        self.data = pd.DataFrame()
        self.summary = pd.DataFrame()
        self.monte_carlo_paths = pd.DataFrame()
        self.monte_carlo_wealth = pd.DataFrame()
        self.monte_carlo_path_summaries = pd.DataFrame()
        self.monte_carlo_summary = pd.DataFrame()

    # Download and store the raw Yahoo Finance data used by the strategy.
    def fetch_data(self):
        self.raw_data = fetch_data(
            ticker=self.ticker,
            start=self.start,
            end=self.end,
            interval=self.tf,
            auto_adjust=True,
            progress=False,
        )
        return self.raw_data

    # Build the momentum-specific signal, volatility, and position sizing columns from a close series.
    def _build_strategy_frame(self, close):
        close = pd.Series(close, copy=False).astype(float)

        df = pd.DataFrame(index=close.index)
        df["close"] = close
        df["return"] = df["close"].pct_change()
        df["log_return"] = log_return(df["close"])
        df["ma"] = df["close"].rolling(self.ma).mean()

        df["signal"] = np.where(df["close"] > df["ma"], 1.0, -1.0)
        df.loc[df["ma"].isna(), "signal"] = np.nan

        if self.bias:
            df["signal"] = df["signal"].replace(-1.0, 0.0)

        df["annual_vol"] = rolling_annualized_vol(
            df["log_return"],
            window="365D",
            min_periods=200,
        )
        df["recent_vol"] = rolling_annualized_vol(
            df["log_return"],
            window="30D",
            min_periods=5,
        )

        df["position"] = df["signal"] * (df["annual_vol"] / df["recent_vol"])
        df.loc[~np.isfinite(df["position"]), "position"] = np.nan
        df["position"] = df["position"] * self.leverage
        return df

    # Evaluate one close path by building the strategy data and passing returns and positions to utils.
    def _evaluate_close_series(self, close):
        df = self._build_strategy_frame(close)
        performance_data, summary = calculate_performance(
            returns=df["log_return"],
            positions=df["position"],
            fees=self.fees,
            log_return=True,
            summary_meta={
                "ticker": self.ticker,
                "start": pd.to_datetime(self.start),
                "end": pd.to_datetime(self.end),
                "bias": self.bias,
                "tf": self.tf,
                "ma": self.ma,
                "fees": self.fees,
                "leverage": self.leverage,
            },
        )

        df = df.join(
            performance_data[
                [
                    "position_prev",
                    "asset_simple_return",
                    "asset_log_return",
                    "gross_strategy_return",
                    "turnover",
                    "fee_cost",
                    "net_strategy_return",
                    "net_log_return",
                    "wealth",
                    "cum_fees",
                    "running_peak",
                    "drawdown",
                ]
            ]
        )
        return df, summary

    # Run the backtest on the real observed close prices.
    def run(self):
        if self.raw_data.empty:
            self.fetch_data()

        self.data, self.summary = self._evaluate_close_series(self.raw_data["close"])
        return self.data

    # Run a bootstrap Monte Carlo on historical returns and summarize average metrics with confidence intervals.
    def run_monte_carlo(self, n_paths=250, seed=None, confidence=0.95):
        if self.raw_data.empty:
            self.fetch_data()

        monte_carlo_results = calculate_monte_carlo_performance(
            close=self.raw_data["close"],
            evaluator=self._evaluate_close_series,
            metric_columns=[
                "total_pnl",
                "total_fees",
                "max_drawdown",
                "winrate",
                "average_return_factor",
                "sharpe_ratio_annualized",
            ],
            n_paths=n_paths,
            seed=seed,
            confidence=confidence,
            summary_meta={
                "ticker": self.ticker,
                "start": pd.to_datetime(self.start),
                "end": pd.to_datetime(self.end),
                "bias": self.bias,
                "tf": self.tf,
                "ma": self.ma,
                "fees": self.fees,
                "leverage": self.leverage,
            },
        )

        self.monte_carlo_paths = monte_carlo_results["paths"]
        self.monte_carlo_wealth = monte_carlo_results["wealth_paths"]
        self.monte_carlo_path_summaries = monte_carlo_results["path_summaries"]
        self.monte_carlo_summary = monte_carlo_results["summary"]
        return self.monte_carlo_summary

    # Plot the wealth curve of the real backtest.
    def plot_wealth(self):
        if self.data.empty:
            self.run()

        return plot_wealth(
            self.data["wealth"],
            title=f"{self.ticker} Momentum Strategy Wealth",
            log_scale=True,
        )

    # Plot the spread of Monte Carlo wealth paths together with the mean path and 95% envelope.
    def plot_monte_carlo(self):
        if self.monte_carlo_wealth.empty:
            self.run_monte_carlo()

        return plot_monte_carlo_wealth(
            self.monte_carlo_wealth,
            title=f"{self.ticker} Momentum Strategy Monte Carlo Wealth",
            log_scale=True,
        )


# Quick start:
# 1. Create and activate a local environment.
# 2. Install the libraries from requirements.txt.
# 3. Run the real backtest summary command below.
# 4. Run the Monte Carlo summary command below if you want average path statistics and 95% confidence intervals.
# 5. Run one of the plotting commands below if you want to export a chart.
#
# Example usage:
# source .venv/bin/activate
# python -c "from momentum import MomentumStrategy; print('import ok')"
# python -c "from momentum import MomentumStrategy; s = MomentumStrategy(ticker='SPY', start='2020-01-01', end='2025-01-01', bias=True, tf='1d', MA=200, fees=0.0005, leverage=1.0); s.run(); print(s.summary)"
# python -c "from momentum import MomentumStrategy; s = MomentumStrategy(ticker='SPY', start='2020-01-01', end='2025-01-01', bias=True, tf='1d', MA=200, fees=0.0005, leverage=1.0); s.run_monte_carlo(n_paths=250, seed=42); print(s.monte_carlo_summary)"
# python -c "from momentum import MomentumStrategy; s = MomentumStrategy(ticker='SPY', start='2020-01-01', end='2025-01-01', bias=True, tf='1d', MA=200, fees=0.0005, leverage=1.0); s.run_monte_carlo(n_paths=250, seed=42); s.monte_carlo_summary.to_csv('monte_carlo_summary.csv', index=False); print('saved monte_carlo_summary.csv')"
# python -c "from momentum import MomentumStrategy; s = MomentumStrategy(ticker='SPY', start='2020-01-01', end='2025-01-01', bias=True, tf='1d', MA=50, fees=0.0005, leverage=1.0); s.run(); s.plot_wealth()"
# python -c "from momentum import MomentumStrategy; s = MomentumStrategy(ticker='SPY', start='2020-01-01', end='2025-01-01', bias=True, tf='1d', MA=50, fees=0.0005, leverage=1.0); s.run_monte_carlo(n_paths=250, seed=42); s.plot_monte_carlo()"
