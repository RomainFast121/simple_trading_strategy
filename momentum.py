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
    summarize_returns,
)


class MomentumStrategy:
    # Initialize the strategy inputs and prepare placeholders for raw data and results.
    def __init__(
        self,
        ticker,
        start,
        end,
        bias,
        tf,
        MA,
        fees,
        target_vol,
        vol_window,
        init_amount,
    ):
        self.ticker = ticker
        self.tickers = [ticker] if isinstance(ticker, str) else list(ticker)
        self.ticker_label = ",".join(self.tickers)
        self.start = start
        self.end = end
        self.bias = bias
        self.tf = tf
        self.ma = MA
        self.fees = fees
        self.target_vol = target_vol
        self.vol_window = vol_window
        self.init_amount = init_amount

        self.raw_data = pd.DataFrame()
        self.data = pd.DataFrame()
        self.summary = pd.DataFrame()
        self.monte_carlo_paths = pd.DataFrame()
        self.monte_carlo_wealth = pd.DataFrame()
        self.monte_carlo_path_summaries = pd.DataFrame()
        self.monte_carlo_summary = pd.DataFrame()

    # Download and store the raw Yahoo Finance data used by the strategy.
    def fetch_data(self):
        tickers = self.tickers[0] if len(self.tickers) == 1 else self.tickers
        self.raw_data = fetch_data(
            ticker=tickers,
            start=self.start,
            end=self.end,
            interval=self.tf,
            auto_adjust=True,
            progress=False,
        )
        return self.raw_data

    # Extract close prices as a consistent dataframe for both single and multiple tickers.
    def _close_frame(self, close_source=None):
        source = self.raw_data if close_source is None else close_source

        if isinstance(source, pd.Series):
            return source.to_frame(name=self.tickers[0]).astype(float)

        if not isinstance(source, pd.DataFrame):
            raise ValueError("Close source must be a pandas Series or DataFrame.")

        if isinstance(source.columns, pd.MultiIndex):
            close_frame = source["close"].copy()
            if isinstance(close_frame, pd.Series):
                close_frame = close_frame.to_frame(name=self.tickers[0])
        elif "close" in source.columns:
            close_frame = source[["close"]].copy()
            close_frame.columns = [self.tickers[0]]
        else:
            close_frame = source.copy()

        close_frame = close_frame.astype(float)
        close_frame = close_frame.reindex(columns=self.tickers)
        return close_frame

    # Build the momentum-specific signal, recent volatility, and target-vol position sizing columns for one ticker.
    def _build_single_ticker_frame(self, close):
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

        df["recent_vol"] = rolling_annualized_vol(
            df["log_return"],
            window=self.vol_window,
            min_periods=self.vol_window,
        )

        df["position"] = df["signal"] * (self.target_vol / df["recent_vol"])
        df.loc[~np.isfinite(df["position"]), "position"] = np.nan
        return df

    # Evaluate one ticker path by building strategy columns and delegating performance calculations to utils.
    def _evaluate_single_ticker(self, close, ticker_name):
        df = self._build_single_ticker_frame(close)
        performance_data, summary = calculate_performance(
            init_amount=self.init_amount,
            returns=df["log_return"],
            positions=df["position"],
            fees=self.fees,
            log_return=True,
            summary_meta={
                "ticker": ticker_name,
                "start": pd.to_datetime(self.start),
                "end": pd.to_datetime(self.end),
                "bias": self.bias,
                "tf": self.tf,
                "ma": self.ma,
                "fees": self.fees,
                "target_vol": self.target_vol,
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
                    "drawdown%",
                ]
            ]
        )
        return df, summary

    # Combine multiple ticker sleeves into an equal-weight portfolio and summarize from total wealth.
    def _evaluate_multi_ticker(self, close_frame):
        ticker_frames = {}
        ticker_net_returns = {}
        ticker_fee_costs = {}

        for ticker_name in close_frame.columns:
            frame, _ = self._evaluate_single_ticker(close_frame[ticker_name], ticker_name)
            ticker_frames[ticker_name] = frame
            ticker_net_returns[ticker_name] = frame["net_strategy_return"].fillna(0.0)
            ticker_fee_costs[ticker_name] = frame["fee_cost"].fillna(0.0)

        ticker_net_returns = pd.DataFrame(ticker_net_returns, index=close_frame.index)
        ticker_fee_costs = pd.DataFrame(ticker_fee_costs, index=close_frame.index)

        portfolio_returns = ticker_net_returns.mean(axis=1)
        portfolio_fees = ticker_fee_costs.mean(axis=1)

        portfolio_data, summary = summarize_returns(
            init_amount=self.init_amount,
            strategy_returns=portfolio_returns,
            fee_cost=portfolio_fees,
            summary_meta={
                "ticker": self.ticker_label,
                "start": pd.to_datetime(self.start),
                "end": pd.to_datetime(self.end),
                "bias": self.bias,
                "tf": self.tf,
                "ma": self.ma,
                "fees": self.fees,
                "target_vol": self.target_vol,
            },
            active_mask=portfolio_returns != 0,
        )

        portfolio_frame = pd.DataFrame(index=close_frame.index)
        portfolio_frame["weight"] = 1.0 / len(close_frame.columns)
        portfolio_frame["net_strategy_return"] = portfolio_data["net_strategy_return"]
        portfolio_frame["net_log_return"] = portfolio_data["net_log_return"]
        portfolio_frame["fee_cost"] = portfolio_data["fee_cost"]
        portfolio_frame["cum_fees"] = portfolio_data["cum_fees"]
        portfolio_frame["wealth"] = portfolio_data["wealth"]
        portfolio_frame["running_peak"] = portfolio_data["running_peak"]
        portfolio_frame["drawdown%"] = portfolio_data["drawdown%"]
        portfolio_frame["active_tickers"] = (ticker_net_returns != 0).sum(axis=1)

        ticker_frames["portfolio"] = portfolio_frame
        combined_data = pd.concat(ticker_frames, axis=1)
        return combined_data, summary

    # Evaluate either a single ticker or a basket of tickers through the same interface.
    def _evaluate_close_series(self, close_source):
        close_frame = self._close_frame(close_source)

        if len(close_frame.columns) == 1:
            ticker_name = close_frame.columns[0]
            return self._evaluate_single_ticker(close_frame.iloc[:, 0], ticker_name)

        return self._evaluate_multi_ticker(close_frame)

    # Run the backtest on the real observed close prices.
    def run(self):
        if self.raw_data.empty:
            self.fetch_data()

        self.data, self.summary = self._evaluate_close_series(self.raw_data)
        return self.data

    # Run a bootstrap Monte Carlo on historical returns and summarize average metrics with confidence intervals.
    def run_monte_carlo(self, n_paths=250, seed=None, confidence=0.95):
        if self.raw_data.empty:
            self.fetch_data()

        close_input = self._close_frame()
        if len(self.tickers) == 1:
            close_input = close_input.iloc[:, 0]

        monte_carlo_results = calculate_monte_carlo_performance(
            close=close_input,
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
                "ticker": self.ticker_label,
                "start": pd.to_datetime(self.start),
                "end": pd.to_datetime(self.end),
                "bias": self.bias,
                "tf": self.tf,
                "ma": self.ma,
                "fees": self.fees,
                "target_vol": self.target_vol,
            },
        )

        self.monte_carlo_paths = monte_carlo_results["paths"]
        self.monte_carlo_wealth = monte_carlo_results["wealth_paths"]
        self.monte_carlo_path_summaries = monte_carlo_results["path_summaries"]
        self.monte_carlo_summary = monte_carlo_results["summary"]
        return self.monte_carlo_summary

    # Plot the wealth curve of the real backtest using total portfolio wealth when several tickers are used.
    def plot_wealth(self):
        if self.data.empty:
            self.run()

        wealth = self.data["portfolio"]["wealth"] if isinstance(self.data.columns, pd.MultiIndex) else self.data["wealth"]
        return plot_wealth(
            wealth,
            title=f"{self.ticker_label} Momentum Strategy Wealth",
            log_scale=True,
        )

    # Plot the spread of Monte Carlo wealth paths together with the mean path and 95% envelope.
    def plot_monte_carlo(self):
        if self.monte_carlo_wealth.empty:
            self.run_monte_carlo()

        return plot_monte_carlo_wealth(
            self.monte_carlo_wealth,
            title=f"{self.ticker_label} Momentum Strategy Monte Carlo Wealth",
            log_scale=True,
        )
