import numpy as np
import pandas as pd
from itertools import product

from utils import (
    attach_buy_and_hold_metrics,
    calculate_buy_and_hold_baseline,
    calculate_monte_carlo_performance,
    calculate_performance,
    combine_sleeve_frames,
    extract_close_map,
    fetch_data,
    log_return,
    normalize_symbol_input,
    plot_monte_carlo_wealth,
    plot_wealth,
    portfolio_column,
    require_attributes,
    rolling_annualized_vol,
    summarize_returns,
    update_provided_attributes,
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
        self.selection_mode = None

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
        self.walkforward_schedule = pd.DataFrame()
        self.walkforward_grid_results = pd.DataFrame()
        self.walkforward_selected_by_symbol = {}

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

        update_provided_attributes(self, updates)
        self._require_strategy_params("Missing strategy parameters: ")

    def _require_strategy_params(self, message):
        require_attributes(
            self,
            ["bias", "ma", "fees", "target_vol", "vol_window", "init_amount"],
            message,
        )

    def _summary_meta(self, *, start, end, ma=None, vol_window=None):
        return {
            "ticker": self.ticker_label,
            "start": start,
            "end": end,
            "bias": self.bias,
            "tf": self.tf,
            "ma": self.ma if ma is None else ma,
            "fees": self.fees,
            "target_vol": self.target_vol,
            "vol_window": self.vol_window if vol_window is None else vol_window,
            "selection_mode": self.selection_mode,
            "hour": self.hour,
            "hour_timezone": self.hour_timezone,
        }

    # Download and store one native raw dataframe per selected sleeve.
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

    # Extract one close series per sleeve from raw data or Monte Carlo inputs.
    def _close_map(self, close_source=None):
        source = self.raw_data if close_source is None else close_source
        return extract_close_map(source, symbols=self.symbols)

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
        valid_position = df["position"].notna()
        if not valid_position.any():
            raise ValueError(f"No valid momentum position was generated for {ticker_name}.")

        evaluation_start = valid_position.loc[valid_position].index[0]
        df = df.loc[evaluation_start:].copy()
        if len(df) < 2:
            raise ValueError(f"Not enough valid momentum rows to evaluate {ticker_name}.")
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
                "bias": self.bias,
                "tf": self.tf,
                "ma": self.ma,
                "fees": self.fees,
                "target_vol": self.target_vol,
                "vol_window": self.vol_window,
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
            try:
                frame, _ = self._evaluate_single_ticker(close_series, ticker_name)
            except ValueError:
                continue
            sleeve_frames[ticker_name] = frame

        if not sleeve_frames:
            raise ValueError("No valid momentum rows were generated for any ticker.")

        return combine_sleeve_frames(
            sleeve_frames=sleeve_frames,
            init_amount=self.init_amount,
            fees=self.fees,
            summary_meta={
                "ticker": self.ticker_label,
                "start": min(frame.index[0] for frame in sleeve_frames.values()),
                "end": max(frame.index[-1] for frame in sleeve_frames.values()),
                "bias": self.bias,
                "tf": self.tf,
                "ma": self.ma,
                "fees": self.fees,
                "target_vol": self.target_vol,
                "vol_window": self.vol_window,
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

    def _slice_native_frame(self, frame, start, end):
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        index = pd.to_datetime(frame.index)
        mask = (index >= start) & (index <= end)
        return frame.loc[mask].copy()

    def _evaluate_single_ticker_window(self, close, ticker_name, start, end, summary_meta=None):
        frame = self._build_single_ticker_frame(close)
        frame = frame.loc[frame["position"].notna()].copy()
        if len(frame) < 2:
            raise ValueError(f"No valid momentum rows were generated for {ticker_name}.")
        frame = frame.iloc[:-1].copy()
        frame = self._slice_native_frame(frame, start=start, end=end)
        if len(frame) < 2:
            raise ValueError(f"Not enough momentum rows for {ticker_name} in this walk-forward window.")

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

    # Evaluate a candidate over one date window while preserving indicators built from the full native history.
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
            raise ValueError("No valid momentum rows were generated for this walk-forward window.")

        if len(sleeve_frames) == 1 and len(close_map) == 1:
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
        if len(all_days) < 3:
            raise ValueError("Not enough dates to create momentum walk-forward splits.")
        return pd.Index(all_days)

    def _grid_ready_days_by_symbol(self, ma_values, vol_window_values):
        previous_ma = self.ma
        previous_vol_window = self.vol_window
        ready_days = {}

        try:
            for ticker_name, close_series in self._close_map().items():
                candidate_ready_days = []
                for MA, vol_window in product(ma_values, vol_window_values):
                    self.ma = int(MA)
                    self.vol_window = int(vol_window)
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
            self.ma = previous_ma
            self.vol_window = previous_vol_window

        if not ready_days:
            raise ValueError("No MA / vol_window candidate produced a valid momentum position.")
        return ready_days

    def _first_walkforward_ready_day(self, ma_values, vol_window_values):
        ready_days = self._grid_ready_days_by_symbol(ma_values, vol_window_values)
        return min(ready_days)

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

    def _make_momentum_walkforward_splits(self, train_days, test_days, step_days, start_day=None):
        days = self._walkforward_days()
        if start_day is not None:
            start_day = pd.Timestamp(start_day).normalize()
            days = days[days >= start_day]
        train_days = int(train_days)
        test_days = int(test_days)
        step_days = int(step_days)
        if min(train_days, test_days, step_days) <= 0:
            raise ValueError("train_days, test_days, and step_days must be positive integers.")
        if len(days) < 3:
            raise ValueError("Not enough valid dates to create momentum walk-forward splits.")
        if train_days + test_days > len(days):
            raise ValueError("Walk-forward windows are larger than the available sample.")

        splits = []
        split_id = 1
        train_start = 0
        while True:
            train_end = train_start + train_days
            test_start = train_end
            test_end = test_start + test_days
            if test_start >= len(days):
                break
            if test_end > len(days):
                test_end = len(days)

            splits.append(
                {
                    "split_id": split_id,
                    "train_start": days[train_start],
                    "train_end": days[train_end - 1],
                    "test_start": days[test_start],
                    "test_end": days[test_end - 1],
                }
            )

            split_id += 1
            train_start += step_days
            if train_start + train_days >= len(days):
                break

        if not splits:
            raise ValueError("No momentum walk-forward splits could be created.")
        return splits

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

    def _evaluate_walkforward_candidate(self, MA, vol_window, start, end, split_id, close_source=None):
        previous_ma = self.ma
        previous_vol_window = self.vol_window
        self.ma = int(MA)
        self.vol_window = int(vol_window)
        try:
            return self._evaluate_close_series_window(
                self.raw_data if close_source is None else close_source,
                start=start,
                end=end,
                summary_meta=self._summary_meta(
                    start=pd.Timestamp(start),
                    end=pd.Timestamp(end),
                    ma=int(MA),
                    vol_window=int(vol_window),
                )
                | {"split_id": split_id},
            )
        finally:
            self.ma = previous_ma
            self.vol_window = previous_vol_window

    def _evaluate_walkforward_per_asset_test(self, selected_by_symbol, start, end, split_id):
        previous_ma = self.ma
        previous_vol_window = self.vol_window
        sleeve_frames = {}
        diagnostics = {}

        try:
            close_map = self._close_map()
            for ticker_name, selection in selected_by_symbol.items():
                if ticker_name not in close_map:
                    continue
                self.ma = int(selection["MA"])
                self.vol_window = int(selection["vol_window"])
                try:
                    frame, _ = self._evaluate_single_ticker_window(
                        close_map[ticker_name],
                        ticker_name,
                        start=start,
                        end=end,
                        summary_meta=self._summary_meta(
                            start=pd.Timestamp(start),
                            end=pd.Timestamp(end),
                            ma=int(selection["MA"]),
                            vol_window=int(selection["vol_window"]),
                        )
                        | {
                            "split_id": split_id,
                            "selection_mode": "per_asset",
                            "selected_symbol": ticker_name,
                        },
                    )
                except ValueError as exc:
                    diagnostics[ticker_name] = {
                        "MA": int(selection["MA"]),
                        "vol_window": int(selection["vol_window"]),
                        "error": str(exc),
                        "native_rows_in_test": len(
                            self._slice_native_frame(
                                pd.DataFrame(index=close_map[ticker_name].index),
                                start=start,
                                end=end,
                            )
                        ),
                    }
                    continue
                frame["selected_ma"] = int(selection["MA"])
                frame["selected_vol_window"] = int(selection["vol_window"])
                sleeve_frames[ticker_name] = frame
        finally:
            self.ma = previous_ma
            self.vol_window = previous_vol_window

        if not sleeve_frames:
            raise ValueError(
                f"No valid per-asset test frames for split {split_id}. "
                f"test_window={pd.Timestamp(start)} -> {pd.Timestamp(end)}; "
                f"selected_by_symbol={selected_by_symbol}; diagnostics={diagnostics}"
            )

        return combine_sleeve_frames(
            sleeve_frames=sleeve_frames,
            init_amount=self.init_amount,
            fees=self.fees,
            summary_meta=self._summary_meta(
                start=pd.Timestamp(start),
                end=pd.Timestamp(end),
                ma="per_asset",
                vol_window="per_asset",
            )
            | {"split_id": split_id, "selection_mode": "per_asset"},
        )

    def run_walkforward(
        self,
        *,
        bias,
        fees,
        target_vol,
        init_amount,
        ma_values,
        vol_window_values,
        train_days,
        test_days,
        step_days,
        selection_mode="cross_asset",
    ):
        self.bias = bias
        self.fees = fees
        self.target_vol = target_vol
        self.init_amount = init_amount
        self.selection_mode = selection_mode

        if selection_mode not in {"cross_asset", "per_asset"}:
            raise ValueError("selection_mode must be 'cross_asset' or 'per_asset'.")

        ma_values = [int(value) for value in ma_values]
        vol_window_values = [int(value) for value in vol_window_values]
        if not ma_values or not vol_window_values:
            raise ValueError("ma_values and vol_window_values must be non-empty.")

        if not self.raw_data:
            self.fetch_data()

        self.data = pd.DataFrame()
        self.summary = pd.DataFrame()
        self.buy_and_hold_data = pd.DataFrame()
        self.buy_and_hold_summary = {}
        self.walkforward_selected_by_symbol = {}
        split_records = []
        grid_records = []
        test_frames = []
        ready_days_by_symbol = self._grid_ready_days_by_symbol(ma_values, vol_window_values)
        first_ready_day = min(ready_days_by_symbol.values())
        splits = self._make_momentum_walkforward_splits(
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
            start_day=first_ready_day,
        )

        for split in splits:
            training_close_source = self._close_source_ready_for_grid(
                ready_days_by_symbol,
                split["train_start"],
            )
            if not training_close_source:
                continue

            if selection_mode == "cross_asset":
                candidates = []
                for MA, vol_window in product(ma_values, vol_window_values):
                    try:
                        _, candidate_summary = self._evaluate_walkforward_candidate(
                            MA=MA,
                            vol_window=vol_window,
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
                            "selection_mode": "cross_asset",
                            "selection_window_start": split["train_start"],
                            "selection_window_end": split["train_end"],
                            "candidate_symbol": "portfolio",
                            "MA": int(MA),
                            "vol_window": int(vol_window),
                        }
                    )
                    grid_records.append(candidate_row)
                    candidates.append((self._score_candidate_summary(candidate_summary), MA, vol_window, candidate_summary))

                if not candidates:
                    raise ValueError(f"No valid MA / vol_window candidate for split {split['split_id']}.")

                _, selected_ma, selected_vol_window, selected_summary = min(candidates, key=lambda item: item[0])
                try:
                    selected_data, _ = self._evaluate_walkforward_candidate(
                        MA=selected_ma,
                        vol_window=selected_vol_window,
                        start=split["test_start"],
                        end=split["test_end"],
                        split_id=split["split_id"],
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
                                "selection_mode": "cross_asset",
                                "selection_window_start": split["train_start"],
                                "selection_window_end": split["train_end"],
                                "test_window_start": split["test_start"],
                                "test_window_end": split["test_end"],
                                "selected_ma": int(selected_ma),
                                "selected_vol_window": int(selected_vol_window),
                                "selected_by_symbol": "",
                                "status": "live_selection_only",
                            }
                        )
                        continue
                    raise ValueError(
                        f"Selected cross-asset parameters failed on test split {split['split_id']}. "
                        f"test_window={pd.Timestamp(split['test_start'])} -> {pd.Timestamp(split['test_end'])}; "
                        f"selected_ma={selected_ma}; selected_vol_window={selected_vol_window}; "
                        f"native_test_counts={native_test_counts}; error={exc}"
                    ) from exc
                selected_data = selected_data.copy()
                if isinstance(selected_data.columns, pd.MultiIndex):
                    selected_data[("portfolio", "split_id")] = split["split_id"]
                    selected_data[("portfolio", "selected_ma")] = int(selected_ma)
                    selected_data[("portfolio", "selected_vol_window")] = int(selected_vol_window)
                    selected_data[("portfolio", "selection_mode")] = "cross_asset"
                else:
                    selected_data["split_id"] = split["split_id"]
                    selected_data["selected_ma"] = int(selected_ma)
                    selected_data["selected_vol_window"] = int(selected_vol_window)
                    selected_data["selection_mode"] = "cross_asset"
                test_frames.append(selected_data)

                selected_row = selected_summary.iloc[0].to_dict()
                selected_row.update(
                    {
                        "split_id": split["split_id"],
                        "selection_mode": "cross_asset",
                        "selection_window_start": split["train_start"],
                        "selection_window_end": split["train_end"],
                        "test_window_start": split["test_start"],
                        "test_window_end": split["test_end"],
                        "selected_ma": int(selected_ma),
                        "selected_vol_window": int(selected_vol_window),
                        "selected_by_symbol": "",
                        "status": "backtested",
                    }
                )
                split_records.append(selected_row)
                continue

            selected_by_symbol = {}
            selected_summaries = {}
            for ticker_name, close_series in training_close_source.items():
                asset_candidates = []
                for MA, vol_window in product(ma_values, vol_window_values):
                    try:
                        _, candidate_summary = self._evaluate_walkforward_candidate(
                            MA=MA,
                            vol_window=vol_window,
                            start=split["train_start"],
                            end=split["train_end"],
                            split_id=split["split_id"],
                            close_source={ticker_name: close_series},
                        )
                    except ValueError:
                        continue

                    candidate_row = candidate_summary.iloc[0].to_dict()
                    candidate_row.update(
                        {
                            "split_id": split["split_id"],
                            "selection_mode": "per_asset",
                            "selection_window_start": split["train_start"],
                            "selection_window_end": split["train_end"],
                            "candidate_symbol": ticker_name,
                            "MA": int(MA),
                            "vol_window": int(vol_window),
                        }
                    )
                    grid_records.append(candidate_row)
                    asset_candidates.append(
                        (self._score_candidate_summary(candidate_summary), MA, vol_window, candidate_summary)
                    )

                if not asset_candidates:
                    continue

                _, selected_ma, selected_vol_window, selected_summary = min(
                    asset_candidates,
                    key=lambda item: item[0],
                )
                selected_by_symbol[ticker_name] = {
                    "MA": int(selected_ma),
                    "vol_window": int(selected_vol_window),
                }
                selected_summaries[ticker_name] = selected_summary

            if not selected_by_symbol:
                raise ValueError(f"No valid per-asset MA / vol_window candidate for split {split['split_id']}.")

            selected_by_symbol_text = ";".join(
                f"{symbol}:MA={params['MA']},vol={params['vol_window']}"
                for symbol, params in selected_by_symbol.items()
            )
            native_test_counts = self._native_test_row_counts(
                selected_by_symbol,
                start=split["test_start"],
                end=split["test_end"],
            )
            if native_test_counts and max(native_test_counts.values()) < 2:
                self.walkforward_selected_by_symbol[split["split_id"]] = selected_by_symbol
                split_records.append(
                    {
                        "split_id": split["split_id"],
                        "selection_mode": "per_asset",
                        "selection_window_start": split["train_start"],
                        "selection_window_end": split["train_end"],
                        "test_window_start": split["test_start"],
                        "test_window_end": split["test_end"],
                        "selected_ma": "per_asset",
                        "selected_vol_window": "per_asset",
                        "selected_by_symbol": selected_by_symbol_text,
                        "status": "live_selection_only",
                    }
                )
                continue

            selected_data, selected_summary = self._evaluate_walkforward_per_asset_test(
                selected_by_symbol,
                start=split["test_start"],
                end=split["test_end"],
                split_id=split["split_id"],
            )
            selected_data = selected_data.copy()
            selected_data[("portfolio", "split_id")] = split["split_id"]
            selected_data[("portfolio", "selected_ma")] = "per_asset"
            selected_data[("portfolio", "selected_vol_window")] = "per_asset"
            selected_data[("portfolio", "selection_mode")] = "per_asset"
            test_frames.append(selected_data)

            selected_row = selected_summary.iloc[0].to_dict()
            selected_row.update(
                {
                    "split_id": split["split_id"],
                    "selection_mode": "per_asset",
                    "selection_window_start": split["train_start"],
                    "selection_window_end": split["train_end"],
                    "test_window_start": split["test_start"],
                    "test_window_end": split["test_end"],
                    "selected_ma": "per_asset",
                    "selected_vol_window": "per_asset",
                    "selected_by_symbol": selected_by_symbol_text,
                    "status": "backtested",
                }
            )
            split_records.append(selected_row)
            self.walkforward_selected_by_symbol[split["split_id"]] = selected_by_symbol

        if not test_frames:
            raise ValueError("No walk-forward test frames were generated.")

        self.data = pd.concat(test_frames, axis=0).sort_index()
        self.data = self.data.loc[~self.data.index.duplicated(keep="first")].copy()
        self.walkforward_schedule = pd.DataFrame(split_records)
        self.walkforward_grid_results = pd.DataFrame(grid_records)

        last_selection = self.walkforward_schedule.iloc[-1]
        if selection_mode == "cross_asset":
            self.ma = int(last_selection["selected_ma"])
            self.vol_window = int(last_selection["selected_vol_window"])
        else:
            self.ma = None
            self.vol_window = None

        summary_meta = self._summary_meta(
            start=self.data.index[0],
            end=self.data.index[-1],
            ma="walkforward",
            vol_window="walkforward",
        ) | {
            "walkforward_train_days": int(train_days),
            "walkforward_test_days": int(test_days),
            "walkforward_step_days": int(step_days),
        }
        self.data, self.summary = self._rebuild_walkforward_summary(self.data, summary_meta)
        _, baseline_summary = self._update_buy_and_hold_baseline_from_current_data()
        attach_buy_and_hold_metrics(self.summary, baseline_summary)
        self.summary["selected_ma"] = last_selection["selected_ma"]
        self.summary["selected_vol_window"] = last_selection["selected_vol_window"]
        self.summary["selection_mode"] = selection_mode
        return self.data

    # Build the Monte Carlo input payload from the currently fetched close data.
    def _monte_carlo_close_input(self):
        close_input = self._close_map()
        if len(close_input) == 1:
            return close_input[next(iter(close_input))]
        return close_input

    # Compute and store the historical buy-and-hold baseline on the same close inputs used by the strategy.
    def _update_buy_and_hold_baseline(self):
        evaluation_index = self.data.index if not self.data.empty else None

        baseline_data, baseline_summary = calculate_buy_and_hold_baseline(
            close_source=self._monte_carlo_close_input(),
            init_amount=self.init_amount,
            target_vol=self.target_vol,
            vol_window=self.vol_window,
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

    def _current_data_close_source(self):
        if self.data.empty:
            raise ValueError("Run run_walkforward(...) before building the B&H baseline.")

        if isinstance(self.data.columns, pd.MultiIndex):
            close_source = {}
            for symbol in self.data.columns.get_level_values(0).unique():
                if symbol == "portfolio":
                    continue
                if (symbol, "close") in self.data.columns:
                    close = self.data[(symbol, "close")].dropna()
                    if not close.empty:
                        close_source[symbol] = close
            if close_source:
                return close_source
            raise ValueError("No close columns found in current strategy data.")

        if "close" not in self.data.columns:
            raise ValueError("No close column found in current strategy data.")
        return self.data["close"].dropna()

    def _update_buy_and_hold_baseline_from_current_data(self):
        evaluation_index = pd.DatetimeIndex(self.data.index)
        baseline_data, baseline_summary = calculate_buy_and_hold_baseline(
            close_source=self._current_data_close_source(),
            init_amount=self.init_amount,
            target_vol=self.target_vol,
            vol_window=self.vol_window,
            fees=self.fees,
            evaluation_index=evaluation_index,
            summary_meta={
                "ticker": self.ticker_label,
                "start": evaluation_index[0],
                "end": evaluation_index[-1],
                "benchmark": "equal_weight_rebalanced",
                "tf": self.tf,
                "fees": self.fees,
                "hour": self.hour,
                "hour_timezone": self.hour_timezone,
            },
        )
        self.buy_and_hold_data = baseline_data.reindex(evaluation_index)
        self.buy_and_hold_summary = baseline_summary
        return self.buy_and_hold_data, baseline_summary

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
        self.monte_carlo_summary["B&H_sharpe_ratio_annualized"] = baseline_summary["sharpe_ratio_annualized"]
        self.monte_carlo_summary["selected_block_length"] = best_length
        self.monte_carlo_summary["search_n_paths"] = search_n_paths

        return best_length

    # Momentum is walk-forward only; keep run(...) as a direct alias for notebook ergonomics.
    def run(self, **kwargs):
        return self.run_walkforward(**kwargs)

    def latest_positions(self):
        """Return the latest momentum positions available from the current raw close data."""
        require_attributes(
            self,
            ["bias", "fees", "target_vol", "init_amount"],
            "Missing strategy parameters. Run run_walkforward(...) first: ",
        )

        if not self.raw_data:
            self.fetch_data()

        close_map = self._close_map()
        frames = {}
        valid_indexes = []
        latest_per_asset_selection = {}
        if self.selection_mode == "per_asset" and self.walkforward_selected_by_symbol:
            latest_split_id = max(self.walkforward_selected_by_symbol)
            latest_per_asset_selection = self.walkforward_selected_by_symbol[latest_split_id]

        previous_ma = self.ma
        previous_vol_window = self.vol_window
        for symbol, close_series in close_map.items():
            if self.selection_mode == "per_asset":
                if symbol not in latest_per_asset_selection:
                    continue
                self.ma = int(latest_per_asset_selection[symbol]["MA"])
                self.vol_window = int(latest_per_asset_selection[symbol]["vol_window"])
            else:
                require_attributes(
                    self,
                    ["ma", "vol_window"],
                    "Missing strategy parameters. Run run_walkforward(...) first: ",
                )

            try:
                frame = self._build_single_ticker_frame(close_series)
                frame = frame.loc[frame["position"].notna()].copy()
                if frame.empty:
                    continue
                frames[symbol] = frame
                valid_indexes.append(frame.index)
            finally:
                self.ma = previous_ma
                self.vol_window = previous_vol_window

        if not frames:
            raise ValueError("No latest momentum positions are available.")

        common_index = valid_indexes[0]
        for index in valid_indexes[1:]:
            common_index = common_index.intersection(index)
        if common_index.empty:
            raise ValueError("No common latest momentum timestamp is available.")

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
                    "signal": frame.loc[timestamp, "signal"],
                    "recent_vol": frame.loc[timestamp, "recent_vol"],
                    "selected_ma": (
                        latest_per_asset_selection[symbol]["MA"]
                        if self.selection_mode == "per_asset"
                        else self.ma
                    ),
                    "selected_vol_window": (
                        latest_per_asset_selection[symbol]["vol_window"]
                        if self.selection_mode == "per_asset"
                        else self.vol_window
                    ),
                    "signal_side": "long" if raw_position > 0 else "short" if raw_position < 0 else "flat",
                    "raw_position": raw_position,
                    "portfolio_weighted_position": raw_position * portfolio_weight,
                }
            )

        return pd.DataFrame(rows).sort_values("SYMBOL").reset_index(drop=True)

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
        attach_buy_and_hold_metrics(self.monte_carlo_summary, baseline_summary)
        return self.monte_carlo_summary

    # Plot the wealth curve of the real backtest using total portfolio wealth when several sleeves are used.
    def plot_wealth(self):
        if self.data.empty:
            raise ValueError("Run run_walkforward(...) before plotting wealth.")

        wealth = portfolio_column(self.data, "wealth")
        self._update_buy_and_hold_baseline_from_current_data()
        benchmark_wealth = portfolio_column(self.buy_and_hold_data, "wealth")
        benchmark_wealth = benchmark_wealth.reindex(wealth.index).dropna()
        if benchmark_wealth.empty:
            raise ValueError("B&H benchmark could not be aligned to the current strategy index.")
        return plot_wealth(
            wealth,
            title=f"{self.ticker_label} Momentum Strategy Wealth",
            log_scale=True,
            benchmark_wealth=benchmark_wealth,
            benchmark_label="B&H",
        )

    # Plot how the walk-forward parameter choices evolve through time.
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

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        if "selection_mode" in schedule.columns and schedule["selection_mode"].eq("per_asset").any():
            rows = []
            for _, row in schedule.iterrows():
                split_id = row["split_id"]
                selections = self.walkforward_selected_by_symbol.get(split_id, {})
                for symbol, params in selections.items():
                    rows.append(
                        {
                            "test_window_start": row["test_window_start"],
                            "symbol": symbol,
                            "selected_ma": params["MA"],
                            "selected_vol_window": params["vol_window"],
                        }
                    )
            if not rows:
                raise ValueError("Per-asset selected parameters are not available for plotting.")
            parameter_frame = pd.DataFrame(rows)
            for symbol, symbol_frame in parameter_frame.groupby("symbol", sort=False):
                axes[0].step(
                    symbol_frame["test_window_start"],
                    symbol_frame["selected_ma"],
                    where="post",
                    linewidth=2,
                    label=symbol,
                )
                axes[0].scatter(symbol_frame["test_window_start"], symbol_frame["selected_ma"], s=28)
                axes[1].step(
                    symbol_frame["test_window_start"],
                    symbol_frame["selected_vol_window"],
                    where="post",
                    linewidth=2,
                    label=symbol,
                )
                axes[1].scatter(symbol_frame["test_window_start"], symbol_frame["selected_vol_window"], s=28)
            axes[0].legend()
            axes[1].legend()
        else:
            axes[0].step(
                schedule["test_window_start"],
                schedule["selected_ma"],
                where="post",
                linewidth=2,
            )
            axes[0].scatter(schedule["test_window_start"], schedule["selected_ma"], s=28)
            axes[1].step(
                schedule["test_window_start"],
                schedule["selected_vol_window"],
                where="post",
                linewidth=2,
            )
            axes[1].scatter(schedule["test_window_start"], schedule["selected_vol_window"], s=28)
        axes[0].set_ylabel("Selected MA")
        axes[0].set_title("Momentum Walk-Forward Selected Parameters")
        axes[1].set_ylabel("Selected Vol Window")
        axes[1].set_xlabel("Test Window Start")

        fig.autofmt_xdate()
        fig.tight_layout()
        return fig, axes

    # Plot the spread of Monte Carlo wealth paths together with the mean path and 95% envelope.
    def plot_monte_carlo(self):
        if self.monte_carlo_wealth.empty:
            self.run_monte_carlo()

        benchmark_wealth = portfolio_column(self.buy_and_hold_data, "wealth")
        return plot_monte_carlo_wealth(
            self.monte_carlo_wealth,
            title=f"{self.ticker_label} Momentum Strategy Monte Carlo Wealth",
            log_scale=True,
            benchmark_wealth=benchmark_wealth,
            benchmark_label="B&H",
        )
