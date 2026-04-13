import numpy as np
import pandas as pd

from utils import (
    calculate_buy_and_hold_baseline,
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
    # Initialize the data-source inputs and prepare placeholders for fetched data and results.
    def __init__(
        self,
        ticker=None,
        crypto=None,
        *,
        start,
        end,
        tf,
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
        self.tf = tf
        self.hour = hour
        self.hour_timezone = hour_timezone

        self.bias = None
        self.ma = None
        self.fees = None
        self.target_vol = None
        self.vol_window = None
        self.init_amount = None

        self.raw_data = {}
        self.data = pd.DataFrame()
        self.summary = pd.DataFrame()
        self.buy_and_hold_data = pd.DataFrame()
        self.buy_and_hold_summary = {}
        self.monte_carlo_paths = {}
        self.monte_carlo_wealth = pd.DataFrame()
        self.monte_carlo_path_summaries = pd.DataFrame()
        self.monte_carlo_summary = pd.DataFrame()
        self.monte_carlo_calibration = pd.DataFrame()

    # Store the strategy parameters that can be reused across repeated runs on the same fetched data.
    def _set_strategy_params(
        self,
        *,
        bias=None,
        MA=None,
        fees=None,
        target_vol=None,
        vol_window=None,
        init_amount=None,
    ):
        updates = {
            "bias": bias,
            "ma": MA,
            "fees": fees,
            "target_vol": target_vol,
            "vol_window": vol_window,
            "init_amount": init_amount,
        }

        for attribute, value in updates.items():
            if value is not None:
                setattr(self, attribute, value)

        missing = [
            name
            for name in ["bias", "ma", "fees", "target_vol", "vol_window", "init_amount"]
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(
                "Missing strategy parameters: " + ", ".join(missing)
            )

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

    # Build the Monte Carlo input payload from the currently fetched close data.
    def _monte_carlo_close_input(self):
        close_input = self._close_map()
        if len(close_input) == 1:
            return close_input[next(iter(close_input))]
        return close_input

    # Compute and store the historical buy-and-hold baseline on the same close inputs used by the strategy.
    def _update_buy_and_hold_baseline(self):
        baseline_data, baseline_summary = calculate_buy_and_hold_baseline(
            close_source=self._monte_carlo_close_input(),
            init_amount=self.init_amount,
            target_vol=self.target_vol,
            vol_window=self.vol_window,
            fees=self.fees,
            summary_meta={
                "ticker": self.ticker_label,
                "start": pd.to_datetime(self.start),
                "end": pd.to_datetime(self.end),
                "tf": self.tf,
                "fees": self.fees,
                "target_vol": self.target_vol,
                "hour": self.hour,
                "hour_timezone": self.hour_timezone,
            },
        )
        self.buy_and_hold_data = baseline_data
        self.buy_and_hold_summary = baseline_summary
        return baseline_data, baseline_summary

    # Run the shared Monte Carlo engine for one specific block length.
    def _run_monte_carlo_with_block_length(self, n_paths, seed, confidence, block_length):
        return calculate_monte_carlo_performance(
            close=self._monte_carlo_close_input(),
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
            block_length=block_length,
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

    # Score how well one Monte Carlo summary contains the realized historical metrics inside its confidence intervals.
    def _score_monte_carlo_calibration(self, monte_carlo_summary, calibration_metrics):
        historical = self.summary.iloc[0]
        monte_carlo = monte_carlo_summary.iloc[0]

        inside_count = 0
        metric_distances = {}

        for metric in calibration_metrics:
            historical_value = historical.get(metric, np.nan)
            lower = monte_carlo.get(f"{metric}_ci_lower", np.nan)
            upper = monte_carlo.get(f"{metric}_ci_upper", np.nan)

            if pd.isna(historical_value) or pd.isna(lower) or pd.isna(upper) or upper <= lower:
                continue

            interval_width = max(upper - lower, 1e-12)
            if lower <= historical_value <= upper:
                normalized_distance = 0.0
                inside_count += 1
            elif historical_value < lower:
                normalized_distance = (lower - historical_value) / interval_width
            else:
                normalized_distance = (historical_value - upper) / interval_width

            metric_distances[metric] = float(normalized_distance)

        if not metric_distances:
            return {
                "inside_count": 0,
                "mean_interval_distance": np.inf,
                "score": np.inf,
                "all_inside": False,
            }

        mean_interval_distance = float(np.mean(list(metric_distances.values())))
        result = {
            "inside_count": inside_count,
            "mean_interval_distance": mean_interval_distance,
            "score": mean_interval_distance,
            "all_inside": inside_count == len(calibration_metrics),
        }
        for metric in calibration_metrics:
            result[f"{metric}_interval_distance"] = metric_distances.get(metric, np.nan)
        return result

    # Search a small number of block lengths and keep the one that best matches the realized metrics.
    def calibrate_monte_carlo_block_length(
        self,
        n_paths=250,
        search_n_paths=100,
        seed=None,
        confidence=0.95,
        *,
        bias=None,
        MA=None,
        fees=None,
        target_vol=None,
        vol_window=None,
        init_amount=None,
        min_block_length=5,
        max_block_length=365,
        step=5,
        max_evaluations=9,
        calibration_metrics=None,
    ):
        self._set_strategy_params(
            bias=bias,
            MA=MA,
            fees=fees,
            target_vol=target_vol,
            vol_window=vol_window,
            init_amount=init_amount,
        )

        if not self.raw_data:
            self.fetch_data()

        self.data, self.summary = self._evaluate_close_series(self.raw_data)

        calibration_metrics = calibration_metrics or [
            "yearly_factor",
            "max_drawdown",
            "winrate",
            "sharpe_ratio_annualized",
        ]

        low = int(min_block_length)
        high = int(max_block_length)
        if step <= 0:
            raise ValueError("step must be positive.")
        if low >= high:
            raise ValueError("min_block_length must be smaller than max_block_length.")

        def round_to_step(value):
            rounded = int(step * round(float(value) / step))
            return max(low, min(high, rounded))

        evaluated = {}

        def evaluate_length(block_length_value):
            block_length_value = round_to_step(block_length_value)
            if block_length_value in evaluated or len(evaluated) >= max_evaluations:
                return block_length_value

            monte_carlo_results = self._run_monte_carlo_with_block_length(
                n_paths=search_n_paths,
                seed=seed,
                confidence=confidence,
                block_length=block_length_value,
            )
            calibration_score = self._score_monte_carlo_calibration(
                monte_carlo_results["summary"],
                calibration_metrics=calibration_metrics,
            )

            record = {
                "block_length": block_length_value,
                **calibration_score,
            }
            for metric in calibration_metrics:
                record[f"historical_{metric}"] = self.summary.iloc[0].get(metric, np.nan)
                record[f"monte_carlo_{metric}"] = monte_carlo_results["summary"].iloc[0].get(metric, np.nan)
                record[f"monte_carlo_{metric}_ci_lower"] = monte_carlo_results["summary"].iloc[0].get(
                    f"{metric}_ci_lower", np.nan
                )
                record[f"monte_carlo_{metric}_ci_upper"] = monte_carlo_results["summary"].iloc[0].get(
                    f"{metric}_ci_upper", np.nan
                )

            evaluated[block_length_value] = {
                "results": monte_carlo_results,
                "record": record,
            }
            return block_length_value

        def get_record(block_length_value):
            return evaluated[block_length_value]["record"]

        def rank_key(block_length_value):
            record = get_record(block_length_value)
            return (
                record["all_inside"],
                -record["score"],
                record["inside_count"],
                -abs(block_length_value),
            )

        for candidate in [low, high]:
            evaluate_length(candidate)
            if get_record(candidate)["all_inside"]:
                break

        left_bound = low
        right_bound = high

        while len(evaluated) < max_evaluations and (right_bound - left_bound) > step:
            midpoint = round_to_step((left_bound + right_bound) / 2)
            if midpoint in (left_bound, right_bound):
                break

            evaluate_length(midpoint)
            if get_record(midpoint)["all_inside"]:
                break

            left_probe = round_to_step((left_bound + midpoint) / 2)
            right_probe = round_to_step((midpoint + right_bound) / 2)

            for candidate in [left_probe, right_probe]:
                if len(evaluated) >= max_evaluations:
                    break
                if candidate in (left_bound, midpoint, right_bound):
                    continue
                evaluate_length(candidate)
                if get_record(candidate)["all_inside"]:
                    break

            if any(record["record"]["all_inside"] for record in evaluated.values()):
                break

            interval_candidates = [candidate for candidate in [left_probe, midpoint, right_probe] if candidate in evaluated]
            best_interval_candidate = min(
                interval_candidates,
                key=lambda candidate: (
                    get_record(candidate)["score"],
                    -get_record(candidate)["inside_count"],
                    abs(candidate - midpoint),
                ),
            )

            if best_interval_candidate == left_probe:
                right_bound = midpoint
            elif best_interval_candidate == right_probe:
                left_bound = midpoint
            else:
                left_bound = left_probe
                right_bound = right_probe

        inside_candidates = [
            length for length, payload in evaluated.items() if payload["record"]["all_inside"]
        ]

        if inside_candidates:
            best_length = min(
                inside_candidates,
                key=lambda candidate: (
                    get_record(candidate)["score"],
                    -get_record(candidate)["inside_count"],
                    candidate,
                ),
            )
        else:
            best_length = min(
                evaluated,
                key=lambda candidate: (
                    get_record(candidate)["score"],
                    -get_record(candidate)["inside_count"],
                    candidate,
                ),
            )

        for candidate in [best_length - step, best_length + step]:
            if len(evaluated) >= max_evaluations:
                break
            if low <= candidate <= high and candidate not in evaluated:
                evaluate_length(candidate)

        def local_status(block_length_value):
            left_neighbor = block_length_value - step
            right_neighbor = block_length_value + step
            left_complete = left_neighbor in evaluated
            right_complete = right_neighbor in evaluated
            complete = left_complete and right_complete

            if not complete:
                return {
                    "neighbor_check_complete": False,
                    "is_local_optimum": False,
                }

            current_record = get_record(block_length_value)
            left_record = get_record(left_neighbor)
            right_record = get_record(right_neighbor)
            current_tuple = (current_record["all_inside"], -current_record["score"], current_record["inside_count"])
            left_tuple = (left_record["all_inside"], -left_record["score"], left_record["inside_count"])
            right_tuple = (right_record["all_inside"], -right_record["score"], right_record["inside_count"])

            return {
                "neighbor_check_complete": True,
                "is_local_optimum": current_tuple > left_tuple and current_tuple > right_tuple,
            }

        calibration_records = [evaluated[length]["record"] for length in sorted(evaluated)]
        calibration_frame = pd.DataFrame(calibration_records)
        local_checks = calibration_frame["block_length"].apply(lambda length: local_status(int(length)))
        calibration_frame["neighbor_check_complete"] = local_checks.apply(
            lambda status: status["neighbor_check_complete"]
        )
        calibration_frame["is_local_optimum"] = local_checks.apply(
            lambda status: status["is_local_optimum"]
        )

        self.monte_carlo_calibration = calibration_frame.sort_values(
            by=["all_inside", "is_local_optimum", "neighbor_check_complete", "score", "inside_count", "block_length"],
            ascending=[False, False, False, True, False, True],
        ).reset_index(drop=True)

        best_row = self.monte_carlo_calibration.iloc[0]
        best_length = int(best_row["block_length"])
        best_results = self._run_monte_carlo_with_block_length(
            n_paths=n_paths,
            seed=seed,
            confidence=confidence,
            block_length=best_length,
        )

        self.monte_carlo_paths = best_results["paths"]
        self.monte_carlo_wealth = best_results["wealth_paths"]
        self.monte_carlo_path_summaries = best_results["path_summaries"]
        self.monte_carlo_summary = best_results["summary"].copy()
        _, baseline_summary = self._update_buy_and_hold_baseline()
        self.monte_carlo_summary["B&H_yearly_factor"] = baseline_summary["yearly_factor"]
        self.monte_carlo_summary["B&H_max_drawdown"] = baseline_summary["max_drawdown"]
        self.monte_carlo_summary["selected_block_length"] = best_length
        self.monte_carlo_summary["search_n_paths"] = search_n_paths

        return best_length

    # Run the backtest on the real observed close prices.
    def run(
        self,
        *,
        bias=None,
        MA=None,
        fees=None,
        target_vol=None,
        vol_window=None,
        init_amount=None,
    ):
        self._set_strategy_params(
            bias=bias,
            MA=MA,
            fees=fees,
            target_vol=target_vol,
            vol_window=vol_window,
            init_amount=init_amount,
        )

        if not self.raw_data:
            self.fetch_data()

        self.data, self.summary = self._evaluate_close_series(self.raw_data)
        _, baseline_summary = self._update_buy_and_hold_baseline()
        self.summary["B&H_yearly_factor"] = baseline_summary["yearly_factor"]
        self.summary["B&H_max_drawdown"] = baseline_summary["max_drawdown"]
        return self.data

    # Run a bootstrap Monte Carlo on historical returns and summarize average metrics with confidence intervals.
    def run_monte_carlo(
        self,
        n_paths=250,
        seed=None,
        confidence=0.95,
        block_length=None,
        *,
        bias=None,
        MA=None,
        fees=None,
        target_vol=None,
        vol_window=None,
        init_amount=None,
    ):
        self._set_strategy_params(
            bias=bias,
            MA=MA,
            fees=fees,
            target_vol=target_vol,
            vol_window=vol_window,
            init_amount=init_amount,
        )

        if not self.raw_data:
            self.fetch_data()
        monte_carlo_results = self._run_monte_carlo_with_block_length(
            n_paths=n_paths,
            seed=seed,
            confidence=confidence,
            block_length=block_length,
        )

        self.monte_carlo_paths = monte_carlo_results["paths"]
        self.monte_carlo_wealth = monte_carlo_results["wealth_paths"]
        self.monte_carlo_path_summaries = monte_carlo_results["path_summaries"]
        self.monte_carlo_summary = monte_carlo_results["summary"]
        _, baseline_summary = self._update_buy_and_hold_baseline()
        self.monte_carlo_summary["B&H_yearly_factor"] = baseline_summary["yearly_factor"]
        self.monte_carlo_summary["B&H_max_drawdown"] = baseline_summary["max_drawdown"]
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
        benchmark_wealth = (
            self.buy_and_hold_data["portfolio"]["wealth"]
            if isinstance(self.buy_and_hold_data, pd.DataFrame)
            and isinstance(self.buy_and_hold_data.columns, pd.MultiIndex)
            else self.buy_and_hold_data.get("wealth")
        )
        return plot_wealth(
            wealth,
            title=f"{self.ticker_label} Momentum Strategy Wealth",
            log_scale=True,
            benchmark_wealth=benchmark_wealth,
            benchmark_label="B&H",
        )

    # Plot the spread of Monte Carlo wealth paths together with the mean path and 95% envelope.
    def plot_monte_carlo(self):
        if self.monte_carlo_wealth.empty:
            self.run_monte_carlo()

        benchmark_wealth = (
            self.buy_and_hold_data["portfolio"]["wealth"]
            if isinstance(self.buy_and_hold_data, pd.DataFrame)
            and isinstance(self.buy_and_hold_data.columns, pd.MultiIndex)
            else self.buy_and_hold_data.get("wealth")
        )
        return plot_monte_carlo_wealth(
            self.monte_carlo_wealth,
            title=f"{self.ticker_label} Momentum Strategy Monte Carlo Wealth",
            log_scale=True,
            benchmark_wealth=benchmark_wealth,
            benchmark_label="B&H",
        )
