import numpy as np
import pandas as pd

from utils import (
    calculate_monte_carlo_performance,
    calculate_performance,
    combine_sleeve_frames,
    fetch_data,
    log_return,
    normalize_symbol_input,
    plot_monte_carlo_wealth,
    plot_wealth,
    rolling_annualized_vol,
)


class MomentumStrategy:
    # Initialize the strategy inputs and prepare placeholders for raw data and results.
    def __init__(
        self,
        ticker=None,
        crypto=None,
        *,
        start,
        end,
        bias,
        tf,
        MA,
        fees,
        target_vol,
        vol_window,
        init_amount,
        hour=None,
        hour_timezone="UTC",
    ):
        self.ticker = ticker
        self.crypto = crypto
        self.tickers = normalize_symbol_input(ticker)
        self.crypto_tickers = normalize_symbol_input(crypto)
        self.symbols = self.tickers + self.crypto_tickers

        if not self.symbols:
            raise ValueError("select at least one ticker")

        self.ticker_label = ",".join(self.symbols)
        self.start = start
        self.end = end
        self.bias = bias
        self.tf = tf
        self.ma = MA
        self.fees = fees
        self.target_vol = target_vol
        self.vol_window = vol_window
        self.init_amount = init_amount
        self.hour = hour
        self.hour_timezone = hour_timezone

        self.raw_data = {}
        self.data = pd.DataFrame()
        self.summary = pd.DataFrame()
        self.monte_carlo_paths = {}
        self.monte_carlo_wealth = pd.DataFrame()
        self.monte_carlo_path_summaries = pd.DataFrame()
        self.monte_carlo_summary = pd.DataFrame()

    # Download and store one native raw dataframe per selected sleeve.
    def fetch_data(self):
        self.raw_data = fetch_data(
            ticker=self.tickers,
            crypto=self.crypto_tickers,
            start=self.start,
            end=self.end,
            interval=self.tf,
            auto_adjust=True,
            progress=False,
            hour=self.hour,
            hour_timezone=self.hour_timezone,
        )
        return self.raw_data

    # Extract one close series per sleeve from raw data or Monte Carlo inputs.
    def _close_map(self, close_source=None):
        source = self.raw_data if close_source is None else close_source

        if isinstance(source, pd.Series):
            return {self.symbols[0]: source.astype(float)}

        if isinstance(source, dict):
            close_map = {}
            for symbol, value in source.items():
                if isinstance(value, pd.Series):
                    close_map[symbol] = value.astype(float)
                elif isinstance(value, pd.DataFrame) and "close" in value.columns:
                    close_map[symbol] = value["close"].astype(float)
                else:
                    raise ValueError("Each sleeve must be a Series or a DataFrame with a 'close' column.")
            return close_map

        if isinstance(source, pd.DataFrame):
            if isinstance(source.columns, pd.MultiIndex) and "close" in source.columns.get_level_values(0):
                close_frame = source["close"]
                return {symbol: close_frame[symbol].astype(float) for symbol in close_frame.columns}
            if "close" in source.columns:
                return {self.symbols[0]: source["close"].astype(float)}
            return {symbol: source[symbol].astype(float) for symbol in source.columns}

        raise ValueError("Close source must be a Series, DataFrame, or mapping.")

    # Build the momentum-specific signal, recent volatility, and target-vol position sizing columns for one sleeve.
    def _build_single_ticker_frame(self, close):
        if isinstance(close, pd.DataFrame):
            if close.shape[1] != 1:
                raise ValueError("Each sleeve close input must be one-dimensional.")
            close = close.iloc[:, 0]

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

    # Evaluate one sleeve on its own native calendar before any cross-source merge happens.
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
                "hour": self.hour,
                "hour_timezone": self.hour_timezone,
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

    # Combine multiple fully-built sleeves only after each sleeve finished local indicator construction.
    def _evaluate_multi_ticker(self, close_map):
        sleeve_frames = {}
        for ticker_name, close_series in close_map.items():
            frame, _ = self._evaluate_single_ticker(close_series, ticker_name)
            sleeve_frames[ticker_name] = frame

        return combine_sleeve_frames(
            sleeve_frames=sleeve_frames,
            init_amount=self.init_amount,
            fees=self.fees,
            summary_meta={
                "ticker": self.ticker_label,
                "start": pd.to_datetime(self.start),
                "end": pd.to_datetime(self.end),
                "bias": self.bias,
                "tf": self.tf,
                "ma": self.ma,
                "fees": self.fees,
                "target_vol": self.target_vol,
                "hour": self.hour,
                "hour_timezone": self.hour_timezone,
            },
        )

    # Evaluate either a single sleeve or a basket of sleeves through the same interface.
    def _evaluate_close_series(self, close_source):
        close_map = self._close_map(close_source)

        if len(close_map) == 1:
            ticker_name = next(iter(close_map))
            return self._evaluate_single_ticker(close_map[ticker_name], ticker_name)

        return self._evaluate_multi_ticker(close_map)

    # Run the backtest on the real observed close prices.
    def run(self):
        if not self.raw_data:
            self.fetch_data()

        self.data, self.summary = self._evaluate_close_series(self.raw_data)
        return self.data

    # Run a bootstrap Monte Carlo on historical returns and summarize average metrics with confidence intervals.
    def run_monte_carlo(self, n_paths=250, seed=None, confidence=0.95):
        if not self.raw_data:
            self.fetch_data()

        close_input = self._close_map()
        if len(close_input) == 1:
            close_input = close_input[next(iter(close_input))]

        monte_carlo_results = calculate_monte_carlo_performance(
            close=close_input,
            evaluator=self._evaluate_close_series,
            metric_columns=[
                "yearly_factor",
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
                "hour": self.hour,
                "hour_timezone": self.hour_timezone,
            },
        )

        self.monte_carlo_paths = monte_carlo_results["paths"]
        self.monte_carlo_wealth = monte_carlo_results["wealth_paths"]
        self.monte_carlo_path_summaries = monte_carlo_results["path_summaries"]
        self.monte_carlo_summary = monte_carlo_results["summary"]
        return self.monte_carlo_summary

    # Plot the wealth curve of the real backtest using total portfolio wealth when several sleeves are used.
    def plot_wealth(self):
        if self.data.empty:
            self.run()

        wealth = (
            self.data["portfolio"]["wealth"]
            if isinstance(self.data.columns, pd.MultiIndex)
            else self.data["wealth"]
        )
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
