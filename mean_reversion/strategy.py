from itertools import product

import numpy as np
import pandas as pd

from utils import (
    attach_buy_and_hold_metrics,
    calculate_buy_and_hold_baseline,
    calculate_performance,
    combine_sleeve_frames,
    extract_close_map,
    fetch_data,
    log_return,
    make_train_test_walkforward_windows,
    normalize_symbol_input,
    plot_wealth,
    portfolio_column,
    require_attributes,
    summarize_returns,
    update_provided_attributes,
)


class MeanReversionStrategy:
    # Initialize data-source inputs and placeholders for fetched data and walk-forward results.
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

        self.return_window = None
        self.signal_cap = None
        self.min_std_distance = None
        self.position_scale = None
        self.fees = None
        self.init_amount = None

        self.raw_data = {}
        self.data = pd.DataFrame()
        self.summary = pd.DataFrame()
        self.buy_and_hold_data = pd.DataFrame()
        self.buy_and_hold_summary = {}
        self.walkforward_schedule = pd.DataFrame()
        self.walkforward_grid_results = pd.DataFrame()

    def _set_strategy_params(
        self,
        *,
        return_window=None,
        signal_cap=None,
        min_std_distance=None,
        position_scale=None,
        fees=None,
        init_amount=None,
    ):
        updates = {
            "return_window": return_window,
            "signal_cap": signal_cap,
            "min_std_distance": min_std_distance,
            "position_scale": position_scale,
            "fees": fees,
            "init_amount": init_amount,
        }
        update_provided_attributes(self, updates)
        self._require_strategy_params("Missing mean-reversion strategy parameters: ")
        self.return_window = int(self.return_window)
        self.signal_cap = float(self.signal_cap)
        self.min_std_distance = float(self.min_std_distance)
        self.position_scale = float(self.position_scale)

    def _require_strategy_params(self, message):
        require_attributes(
            self,
            [
                "return_window",
                "signal_cap",
                "min_std_distance",
                "position_scale",
                "fees",
                "init_amount",
            ],
            message,
        )

    def _summary_meta(
        self,
        *,
        start,
        end,
        return_window=None,
        signal_cap=None,
        min_std_distance=None,
        position_scale=None,
    ):
        return {
            "ticker": self.ticker_label,
            "start": start,
            "end": end,
            "tf": self.tf,
            "return_window": self.return_window if return_window is None else return_window,
            "signal_cap": self.signal_cap if signal_cap is None else signal_cap,
            "min_std_distance": self.min_std_distance if min_std_distance is None else min_std_distance,
            "position_scale": self.position_scale if position_scale is None else position_scale,
            "fees": self.fees,
            "hour": self.hour,
            "hour_timezone": self.hour_timezone,
        }

    def fetch_data(self, force=False):
        if self.raw_data and not force:
            return self.raw_data

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

    def _close_map(self, close_source=None):
        source = self.raw_data if close_source is None else close_source
        return extract_close_map(source, symbols=self.symbols)

    # Build a no-lookahead mean-reversion signal from the latest return versus the prior return window.
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

        prior_returns = df["log_return"].shift(1)
        df["return_mean"] = prior_returns.rolling(
            self.return_window,
            min_periods=self.return_window,
        ).mean()
        df["return_std"] = prior_returns.rolling(
            self.return_window,
            min_periods=self.return_window,
        ).std()
        df["return_deviation"] = df["log_return"] - df["return_mean"]
        df["z_score"] = df["return_deviation"] / df["return_std"].replace(0.0, np.nan)
        edge = -df["z_score"]
        df["signal"] = (
            np.sign(edge) * (edge.abs() - self.min_std_distance).clip(lower=0.0)
        ).clip(
            lower=-self.signal_cap,
            upper=self.signal_cap,
        )

        invalid_signal = (
            df["log_return"].isna()
            | df["return_mean"].isna()
            | df["return_std"].isna()
            | (df["return_std"] <= 0.0)
            | df["z_score"].isna()
        )
        df.loc[invalid_signal, "signal"] = np.nan

        df["position"] = df["signal"] * self.position_scale
        df.loc[~np.isfinite(df["position"]), "position"] = np.nan
        return df

    def _evaluate_single_ticker(self, close, ticker_name):
        df = self._build_single_ticker_frame(close)
        valid_position = df["position"].notna()
        if not valid_position.any():
            raise ValueError(f"No valid mean-reversion position was generated for {ticker_name}.")

        evaluation_start = valid_position.loc[valid_position].index[0]
        df = df.loc[evaluation_start:].copy()
        if len(df) < 2:
            raise ValueError(f"Not enough valid mean-reversion rows to evaluate {ticker_name}.")
        df = df.iloc[:-1].copy()

        performance_data, summary = calculate_performance(
            init_amount=self.init_amount,
            returns=df["log_return"],
            positions=df["position"],
            fees=self.fees,
            log_return=True,
            summary_meta={
                "ticker": ticker_name,
                "start": df.index[0],
                "end": df.index[-1],
                "tf": self.tf,
                "return_window": self.return_window,
                "signal_cap": self.signal_cap,
                "min_std_distance": self.min_std_distance,
                "position_scale": self.position_scale,
                "fees": self.fees,
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

    def _evaluate_multi_ticker(self, close_map):
        sleeve_frames = {}
        for ticker_name, close_series in close_map.items():
            try:
                frame, _ = self._evaluate_single_ticker(close_series, ticker_name)
            except ValueError:
                continue
            sleeve_frames[ticker_name] = frame

        if not sleeve_frames:
            raise ValueError("No valid mean-reversion rows were generated for any ticker.")

        return combine_sleeve_frames(
            sleeve_frames=sleeve_frames,
            init_amount=self.init_amount,
            fees=self.fees,
            summary_meta={
                "ticker": self.ticker_label,
                "start": min(frame.index[0] for frame in sleeve_frames.values()),
                "end": max(frame.index[-1] for frame in sleeve_frames.values()),
                "tf": self.tf,
                "return_window": self.return_window,
                "signal_cap": self.signal_cap,
                "min_std_distance": self.min_std_distance,
                "position_scale": self.position_scale,
                "fees": self.fees,
                "hour": self.hour,
                "hour_timezone": self.hour_timezone,
            },
        )

    def _evaluate_close_series(self, close_source):
        close_map = self._close_map(close_source)

        if len(close_map) == 1:
            ticker_name = next(iter(close_map))
            return self._evaluate_single_ticker(close_map[ticker_name], ticker_name)

        return self._evaluate_multi_ticker(close_map)

    def _slice_native_frame(self, frame, start, end):
        index = pd.to_datetime(frame.index)
        start_day = pd.Timestamp(start).normalize()
        end_day = pd.Timestamp(end).normalize()
        index_days = pd.Series(index, index=frame.index).dt.normalize()
        mask = (index_days >= start_day) & (index_days <= end_day)
        return frame.loc[mask].copy()

    def _evaluate_single_ticker_window(self, close, ticker_name, start, end, summary_meta=None):
        frame = self._build_single_ticker_frame(close)
        frame = frame.loc[frame["position"].notna()].copy()
        if len(frame) < 2:
            raise ValueError(f"No valid mean-reversion rows were generated for {ticker_name}.")
        frame = frame.iloc[:-1].copy()
        frame = self._slice_native_frame(frame, start=start, end=end)
        if len(frame) < 2:
            raise ValueError(f"Not enough mean-reversion rows for {ticker_name} in this walk-forward window.")

        performance_data, summary = calculate_performance(
            init_amount=self.init_amount,
            returns=frame["log_return"],
            positions=frame["position"],
            fees=self.fees,
            log_return=True,
            summary_meta=summary_meta
            or self._summary_meta(start=frame.index[0], end=frame.index[-1]),
        )
        frame = frame.join(
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
        return frame, summary

    def _evaluate_close_series_window(self, close_source, start, end, summary_meta=None):
        close_map = self._close_map(close_source)
        sleeve_frames = {}
        single_summary = None

        for ticker_name, close_series in close_map.items():
            try:
                frame, summary = self._evaluate_single_ticker_window(
                    close_series,
                    ticker_name,
                    start=start,
                    end=end,
                    summary_meta=summary_meta,
                )
            except ValueError:
                continue
            sleeve_frames[ticker_name] = frame
            single_summary = summary

        if not sleeve_frames:
            raise ValueError("No valid mean-reversion rows were generated for this walk-forward window.")

        if len(sleeve_frames) == 1 and len(self.symbols) == 1:
            ticker_name = next(iter(sleeve_frames))
            return sleeve_frames[ticker_name], single_summary

        return combine_sleeve_frames(
            sleeve_frames=sleeve_frames,
            init_amount=self.init_amount,
            fees=self.fees,
            summary_meta=summary_meta
            or self._summary_meta(
                start=min(frame.index[0] for frame in sleeve_frames.values()),
                end=max(frame.index[-1] for frame in sleeve_frames.values()),
            ),
        )

    def _walkforward_days(self):
        close_map = self._close_map()
        all_days = sorted(
            set().union(
                *[
                    set(pd.to_datetime(series.index).normalize())
                    for series in close_map.values()
                    if len(series) > 0
                ]
            )
        )
        return pd.Index(all_days)

    def _grid_ready_days_by_symbol(self, return_window_values):
        previous_return_window = self.return_window
        previous_min_std_distance = self.min_std_distance
        ready_days = {}

        try:
            self.min_std_distance = 0.0
            for ticker_name, close_series in self._close_map().items():
                candidate_ready_days = []
                for return_window in return_window_values:
                    self.return_window = int(return_window)
                    frame = self._build_single_ticker_frame(close_series)
                    frame = frame.loc[frame["position"].notna()].copy()
                    if len(frame) < 2:
                        candidate_ready_days = []
                        break
                    frame = frame.iloc[:-1].copy()
                    if not frame.empty:
                        candidate_ready_days.append(pd.Timestamp(frame.index[0]).normalize())
                if candidate_ready_days:
                    ready_days[ticker_name] = max(candidate_ready_days)
        finally:
            self.return_window = previous_return_window
            self.min_std_distance = previous_min_std_distance

        if not ready_days:
            raise ValueError("No return_window candidate produced a valid mean-reversion position.")
        return ready_days

    def _close_source_ready_for_grid(self, ready_days_by_symbol, start):
        start = pd.Timestamp(start).normalize()
        close_map = self._close_map()
        return {
            ticker_name: close_map[ticker_name]
            for ticker_name, ready_day in ready_days_by_symbol.items()
            if ticker_name in close_map and pd.Timestamp(ready_day).normalize() <= start
        }

    def _native_test_row_counts(self, symbols, start, end):
        close_map = self._close_map()
        counts = {}
        for symbol in symbols:
            if symbol not in close_map:
                counts[symbol] = 0
                continue
            frame = pd.DataFrame(index=close_map[symbol].index)
            counts[symbol] = len(self._slice_native_frame(frame, start=start, end=end))
        return counts

    def _score_candidate_summary(self, summary):
        row = summary.iloc[0]
        return (
            -float(row.get("sharpe_ratio_annualized", float("-inf"))),
            -float(row.get("yearly_factor", float("-inf"))),
            float(row.get("total_fees", float("inf"))),
        )

    def _rebuild_walkforward_summary(self, data, summary_meta):
        if isinstance(data.columns, pd.MultiIndex):
            portfolio = data["portfolio"].copy()
            active_mask = portfolio["active_assets"].fillna(0) > 0
            portfolio_summary, summary = summarize_returns(
                init_amount=self.init_amount,
                strategy_returns=portfolio["net_strategy_return"],
                fee_cost=portfolio["fee_cost"],
                summary_meta=summary_meta,
                active_mask=active_mask,
            )
            for column in ["net_log_return", "wealth", "cum_fees", "running_peak", "drawdown%"]:
                data[("portfolio", column)] = portfolio_summary[column]
            data[("portfolio", "turnover")] = portfolio.get("turnover", pd.Series(index=data.index, dtype=float))
            summary["turnover"] = portfolio.get("turnover", pd.Series(dtype=float)).fillna(0.0).mean()
            return data, summary

        active_mask = data["position_prev"].fillna(0.0) != 0.0
        performance_summary, summary = summarize_returns(
            init_amount=self.init_amount,
            strategy_returns=data["net_strategy_return"],
            fee_cost=data["fee_cost"],
            summary_meta=summary_meta,
            active_mask=active_mask,
        )
        for column in ["net_log_return", "wealth", "cum_fees", "running_peak", "drawdown%"]:
            data[column] = performance_summary[column]
        summary["turnover"] = data["turnover"].fillna(0.0).mean()
        return data, summary

    def _evaluate_walkforward_candidate(
        self,
        return_window,
        min_std_distance,
        start,
        end,
        split_id,
        close_source=None,
    ):
        previous_return_window = self.return_window
        previous_min_std_distance = self.min_std_distance
        self.return_window = int(return_window)
        self.min_std_distance = float(min_std_distance)
        try:
            return self._evaluate_close_series_window(
                self.raw_data if close_source is None else close_source,
                start=start,
                end=end,
                summary_meta=self._summary_meta(
                    start=pd.Timestamp(start),
                    end=pd.Timestamp(end),
                    return_window=int(return_window),
                    min_std_distance=float(min_std_distance),
                )
                | {"split_id": split_id},
            )
        finally:
            self.return_window = previous_return_window
            self.min_std_distance = previous_min_std_distance

    def run_walkforward(
        self,
        *,
        fees,
        init_amount,
        return_window_values,
        signal_cap,
        min_std_distance_values,
        position_scale,
        train_days,
        test_days,
        step_days,
    ):
        self.fees = fees
        self.init_amount = init_amount
        self.signal_cap = float(signal_cap)
        self.position_scale = float(position_scale)

        return_window_values = [int(value) for value in return_window_values]
        min_std_distance_values = [float(value) for value in min_std_distance_values]
        if not return_window_values:
            raise ValueError("Walk-forward parameter value lists must be non-empty.")
        if any(value <= 0 for value in return_window_values):
            raise ValueError("return_window_values must contain positive integers.")
        if not min_std_distance_values:
            raise ValueError("min_std_distance_values must be non-empty.")
        if any(value < 0.0 for value in min_std_distance_values):
            raise ValueError("min_std_distance_values must contain non-negative values.")
        if self.signal_cap <= 0.0:
            raise ValueError("signal_cap must be positive.")
        if self.position_scale <= 0.0:
            raise ValueError("position_scale must be positive.")

        if not self.raw_data:
            self.fetch_data()

        split_records = []
        grid_records = []
        test_frames = []
        ready_days_by_symbol = self._grid_ready_days_by_symbol(return_window_values)
        first_ready_day = min(ready_days_by_symbol.values())
        walkforward_days = self._walkforward_days()
        walkforward_days = walkforward_days[walkforward_days >= first_ready_day]
        splits = make_train_test_walkforward_windows(
            walkforward_days,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
            label="mean-reversion",
        )

        for split in splits:
            training_close_source = self._close_source_ready_for_grid(
                ready_days_by_symbol,
                split["train_start"],
            )
            if not training_close_source:
                continue

            candidates = []
            for return_window, min_std_distance in product(
                return_window_values,
                min_std_distance_values,
            ):
                try:
                    _, candidate_summary = self._evaluate_walkforward_candidate(
                        return_window=return_window,
                        min_std_distance=min_std_distance,
                        start=split["train_start"],
                        end=split["train_end"],
                        split_id=split["split_id"],
                        close_source=training_close_source,
                    )
                except ValueError:
                    continue

                candidate_row = candidate_summary.iloc[0].to_dict()
                candidate_row.update(
                    {
                        "split_id": split["split_id"],
                        "selection_window_start": split["train_start"],
                        "selection_window_end": split["train_end"],
                        "return_window": int(return_window),
                        "signal_cap": self.signal_cap,
                        "min_std_distance": float(min_std_distance),
                        "position_scale": self.position_scale,
                    }
                )
                grid_records.append(candidate_row)
                candidates.append(
                    (
                        self._score_candidate_summary(candidate_summary),
                        return_window,
                        min_std_distance,
                        candidate_summary,
                    )
                )

            if not candidates:
                raise ValueError(f"No valid mean-reversion candidate for split {split['split_id']}.")

            (
                _,
                selected_return_window,
                selected_min_std_distance,
                selected_summary,
            ) = min(candidates, key=lambda item: item[0])

            try:
                selected_data, _ = self._evaluate_walkforward_candidate(
                    return_window=selected_return_window,
                    min_std_distance=selected_min_std_distance,
                    start=split["test_start"],
                    end=split["test_end"],
                    split_id=split["split_id"],
                    close_source=training_close_source,
                )
            except ValueError as exc:
                native_test_counts = self._native_test_row_counts(
                    training_close_source,
                    start=split["test_start"],
                    end=split["test_end"],
                )
                if native_test_counts and max(native_test_counts.values()) < 2:
                    split_records.append(
                        {
                            "split_id": split["split_id"],
                            "selection_window_start": split["train_start"],
                            "selection_window_end": split["train_end"],
                            "test_window_start": split["test_start"],
                            "test_window_end": split["test_end"],
                            "selected_return_window": int(selected_return_window),
                            "selected_min_std_distance": float(selected_min_std_distance),
                            "signal_cap": self.signal_cap,
                            "position_scale": self.position_scale,
                            "eligible_symbols": ",".join(training_close_source),
                            "status": "live_selection_only",
                        }
                    )
                    continue
                raise ValueError(
                    f"Selected mean-reversion parameters failed on test split {split['split_id']}. "
                    f"test_window={pd.Timestamp(split['test_start'])} -> {pd.Timestamp(split['test_end'])}; "
                    f"selected_return_window={selected_return_window}; "
                    f"selected_min_std_distance={selected_min_std_distance}; "
                    f"eligible_symbols={list(training_close_source)}; "
                    f"native_test_counts={native_test_counts}; error={exc}"
                ) from exc
            selected_data = selected_data.copy()
            if isinstance(selected_data.columns, pd.MultiIndex):
                selected_data[("portfolio", "split_id")] = split["split_id"]
                selected_data[("portfolio", "selected_return_window")] = int(selected_return_window)
                selected_data[("portfolio", "signal_cap")] = self.signal_cap
                selected_data[("portfolio", "selected_min_std_distance")] = float(selected_min_std_distance)
                selected_data[("portfolio", "position_scale")] = self.position_scale
            else:
                selected_data["split_id"] = split["split_id"]
                selected_data["selected_return_window"] = int(selected_return_window)
                selected_data["signal_cap"] = self.signal_cap
                selected_data["selected_min_std_distance"] = float(selected_min_std_distance)
                selected_data["position_scale"] = self.position_scale
            test_frames.append(selected_data)

            selected_row = selected_summary.iloc[0].to_dict()
            selected_row.update(
                {
                    "split_id": split["split_id"],
                    "selection_window_start": split["train_start"],
                    "selection_window_end": split["train_end"],
                    "test_window_start": split["test_start"],
                    "test_window_end": split["test_end"],
                    "selected_return_window": int(selected_return_window),
                    "selected_min_std_distance": float(selected_min_std_distance),
                    "signal_cap": self.signal_cap,
                    "position_scale": self.position_scale,
                    "eligible_symbols": ",".join(training_close_source),
                    "status": "backtested",
                }
            )
            split_records.append(selected_row)

        if not test_frames:
            raise ValueError("No walk-forward test frames were generated.")

        self.data = pd.concat(test_frames, axis=0).sort_index()
        self.data = self.data.loc[~self.data.index.duplicated(keep="first")].copy()
        self.walkforward_schedule = pd.DataFrame(split_records)
        self.walkforward_grid_results = pd.DataFrame(grid_records)

        last_selection = self.walkforward_schedule.iloc[-1]
        self.return_window = int(last_selection["selected_return_window"])
        self.min_std_distance = float(last_selection["selected_min_std_distance"])

        summary_meta = self._summary_meta(
            start=self.data.index[0],
            end=self.data.index[-1],
            return_window="walkforward",
            min_std_distance="walkforward",
        ) | {
            "walkforward_train_days": int(train_days),
            "walkforward_test_days": int(test_days),
            "walkforward_step_days": int(step_days),
        }
        self.data, self.summary = self._rebuild_walkforward_summary(self.data, summary_meta)
        _, baseline_summary = self._update_buy_and_hold_baseline()
        attach_buy_and_hold_metrics(self.summary, baseline_summary)
        self.summary["selected_return_window"] = self.return_window
        self.summary["selected_min_std_distance"] = self.min_std_distance
        return self.data

    def run(self, **kwargs):
        return self.run_walkforward(**kwargs)

    def _monte_carlo_close_input(self):
        close_input = self._close_map()
        if len(close_input) == 1:
            return close_input[next(iter(close_input))]
        return close_input

    def _update_buy_and_hold_baseline(self):
        evaluation_index = self.data.index if not self.data.empty else None

        baseline_data, baseline_summary = calculate_buy_and_hold_baseline(
            close_source=self._monte_carlo_close_input(),
            init_amount=self.init_amount,
            fees=self.fees,
            evaluation_index=evaluation_index,
            summary_meta={
                "ticker": self.ticker_label,
                "start": evaluation_index[0] if evaluation_index is not None else pd.to_datetime(self.start),
                "end": evaluation_index[-1] if evaluation_index is not None else pd.to_datetime(self.end),
                "benchmark": "equal_weight_rebalanced",
                "tf": self.tf,
                "fees": self.fees,
                "hour": self.hour,
                "hour_timezone": self.hour_timezone,
            },
        )
        if evaluation_index is not None:
            baseline_data = baseline_data.reindex(evaluation_index)
        self.buy_and_hold_data = baseline_data
        self.buy_and_hold_summary = baseline_summary
        return baseline_data, baseline_summary

    def latest_positions(self):
        self._require_strategy_params("Missing strategy parameters. Run run_walkforward(...) first: ")

        if not self.raw_data:
            self.fetch_data()

        close_map = self._close_map()
        frames = {}
        valid_indexes = []
        for symbol, close_series in close_map.items():
            frame = self._build_single_ticker_frame(close_series)
            frame = frame.loc[frame["position"].notna()].copy()
            if frame.empty:
                continue
            frames[symbol] = frame
            valid_indexes.append(frame.index)

        if not frames:
            raise ValueError("No latest mean-reversion positions are available.")

        common_index = valid_indexes[0]
        for index in valid_indexes[1:]:
            common_index = common_index.intersection(index)
        if common_index.empty:
            raise ValueError("No common latest mean-reversion timestamp is available.")

        timestamp = common_index.max()
        rows = []
        raw_positions = {
            symbol: float(frame.loc[timestamp, "position"])
            for symbol, frame in frames.items()
        }
        active_count = max(sum(position != 0.0 for position in raw_positions.values()), 1)

        for symbol, frame in frames.items():
            raw_position = raw_positions[symbol]
            portfolio_weight = (1.0 / active_count) if raw_position != 0.0 else 0.0
            rows.append(
                {
                    "available_at": timestamp,
                    "applies_to": "next_bar",
                    "SYMBOL": symbol,
                    "close": frame.loc[timestamp, "close"],
                    "log_return": frame.loc[timestamp, "log_return"],
                    "return_mean": frame.loc[timestamp, "return_mean"],
                    "z_score": frame.loc[timestamp, "z_score"],
                    "signal": frame.loc[timestamp, "signal"],
                    "signal_side": "long" if raw_position > 0 else "short" if raw_position < 0 else "flat",
                    "raw_position": raw_position,
                    "portfolio_weighted_position": raw_position * portfolio_weight,
                }
            )

        return pd.DataFrame(rows).sort_values("SYMBOL").reset_index(drop=True)

    def plot_wealth(self):
        if self.data.empty:
            raise ValueError("Run run_walkforward(...) before plotting wealth.")

        wealth = portfolio_column(self.data, "wealth")
        benchmark_wealth = portfolio_column(self.buy_and_hold_data, "wealth")
        benchmark_wealth = benchmark_wealth.reindex(wealth.index).dropna()
        if benchmark_wealth.empty:
            raise ValueError("B&H benchmark could not be aligned to the current strategy index.")
        return plot_wealth(
            wealth,
            title=f"{self.ticker_label} Mean-Reversion Strategy Wealth",
            log_scale=True,
            benchmark_wealth=benchmark_wealth,
            benchmark_label="B&H",
        )

    def plot_walkforward_parameters(self):
        if self.walkforward_schedule.empty:
            raise ValueError("Run run_walkforward(...) before plotting selected parameters.")

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError as exc:  # pragma: no cover - plotting dependency
            raise ImportError("matplotlib and seaborn are required for plotting.") from exc

        schedule = self.walkforward_schedule.copy()
        schedule["test_window_start"] = pd.to_datetime(schedule["test_window_start"])
        sns.set_theme(style="whitegrid")

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        parameter_columns = [
            ("selected_return_window", "Selected Return Window"),
            ("selected_min_std_distance", "Selected Min Std Distance"),
        ]

        for axis, (column, label) in zip(axes, parameter_columns):
            axis.step(schedule["test_window_start"], schedule[column], where="post", linewidth=2)
            axis.scatter(schedule["test_window_start"], schedule[column], s=28)
            axis.set_ylabel(label)

        axes[0].set_title("Mean-Reversion Walk-Forward Selected Parameters")
        axes[-1].set_xlabel("Test Window Start")

        fig.autofmt_xdate()
        fig.tight_layout()
        return fig, axes
