from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from utils import (
    calculate_buy_and_hold_baseline,
    estimate_periods_per_year,
    plot_wealth,
    summarize_returns,
)


@dataclass
class StrategyFileData:
    name: str
    path: Path
    frame: pd.DataFrame
    closes: pd.DataFrame
    positions: pd.DataFrame
    active: pd.Series
    timeframe: pd.Timedelta | None


def _detect_separator(path):
    with Path(path).open() as file:
        first_line = file.readline()
    return ";" if first_line.startswith(";") else ","


def _infer_timeframe(index):
    index = pd.DatetimeIndex(index).sort_values()
    if len(index) < 2:
        return None
    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        return None
    return deltas.median()


def _format_timedelta(value):
    if value is None:
        return "unknown"
    return str(pd.Timedelta(value))


class AssemblingStrategy:
    """Combine existing strategy output files into one fee-aware portfolio."""

    def __init__(
        self,
        strategy_paths=None,
        *,
        weights=None,
        combination_method="weighted_average",
        fees=None,
        init_amount=1000,
        benchmark_target_vol=None,
        benchmark_vol_window=None,
        volatility_target=None,
        volatility_window=None,
        rolling_sharpe_window=None,
        rolling_sharpe_weight_grid=None,
        summary_path="local_outputs/xgb/current/summary.csv",
    ):
        self.strategy_paths = strategy_paths or {
            "momentum": "momentum_result.csv",
            "xgb": "local_outputs/xgb/current/strategy_data.csv",
        }
        self.weights = weights or {name: 1.0 / len(self.strategy_paths) for name in self.strategy_paths}
        self.combination_method = combination_method
        self.summary_path = Path(summary_path) if summary_path is not None else None
        self.fees = fees
        self.init_amount = init_amount
        self.benchmark_target_vol = benchmark_target_vol
        self.benchmark_vol_window = benchmark_vol_window
        self.volatility_target = volatility_target
        self.volatility_window = volatility_window
        self.rolling_sharpe_window = rolling_sharpe_window
        self.rolling_sharpe_weight_grid = rolling_sharpe_weight_grid

        self.inputs = {}
        self.data = pd.DataFrame()
        self.summary = pd.DataFrame()
        self.variant_data = {}
        self.variant_summaries = pd.DataFrame()
        self.variant_wealth = pd.DataFrame()
        self.variant_positions = {}
        self.variant_weights = {}
        self.buy_and_hold_data = pd.DataFrame()
        self.buy_and_hold_summary = pd.DataFrame()
        self.component_summaries = pd.DataFrame()
        self.component_wealth = pd.DataFrame()
        self.comparison = pd.DataFrame()
        self.common_index = pd.DatetimeIndex([])
        self.active_index = pd.DatetimeIndex([])
        self.symbols = []
        self.closes = pd.DataFrame()
        self.component_positions = {}
        self.component_returns = pd.DataFrame()
        self.combined_positions = pd.DataFrame()

    def _load_strategy_file(self, name, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{name} strategy file not found: {path}")

        separator = _detect_separator(path)
        frame = pd.read_csv(path, sep=separator, header=[0, 1], index_col=0)
        frame.index = pd.to_datetime(frame.index, utc=True)
        frame = frame.sort_index()

        if not isinstance(frame.columns, pd.MultiIndex):
            raise ValueError(f"{name} file must use two CSV header rows.")

        level_0 = frame.columns.get_level_values(0)
        symbols = [symbol for symbol in level_0.unique() if symbol != "portfolio"]
        close_columns = {}
        position_columns = {}

        for symbol in symbols:
            if (symbol, "close") not in frame.columns:
                warnings.warn(f"{name}: missing close column for {symbol}; symbol will be skipped.")
                continue
            if (symbol, "weighted_position") in frame.columns:
                position_col = (symbol, "weighted_position")
            elif (symbol, "position") in frame.columns:
                position_col = (symbol, "position")
            else:
                warnings.warn(f"{name}: missing position column for {symbol}; symbol will be skipped.")
                continue

            close_columns[symbol] = pd.to_numeric(frame[(symbol, "close")], errors="coerce")
            position_columns[symbol] = pd.to_numeric(frame[position_col], errors="coerce")

        if not close_columns or not position_columns:
            raise ValueError(f"{name}: no usable symbol close/position columns were found.")

        closes = pd.DataFrame(close_columns, index=frame.index)
        raw_positions = pd.DataFrame(position_columns, index=frame.index)
        active = raw_positions.notna().all(axis=1) & closes.notna().all(axis=1)
        positions = raw_positions.fillna(0.0)

        return StrategyFileData(
            name=name,
            path=path,
            frame=frame,
            closes=closes,
            positions=positions,
            active=active,
            timeframe=_infer_timeframe(frame.index),
        )

    def _load_inputs(self):
        self.inputs = {
            name: self._load_strategy_file(name, path)
            for name, path in self.strategy_paths.items()
        }
        if len(self.inputs) < 2:
            raise ValueError("At least two strategy files are required.")
        return self.inputs

    def _infer_defaults_from_summary(self):
        if self.summary_path is None or not self.summary_path.exists():
            return

        try:
            summary = pd.read_csv(self.summary_path)
        except pd.errors.EmptyDataError:
            return
        if summary.empty:
            return

        row = summary.iloc[0]
        if self.fees is None and "fees" in row:
            self.fees = float(row["fees"])
        if self.benchmark_target_vol is None and "target_vol" in row:
            self.benchmark_target_vol = float(row["target_vol"])
        if self.benchmark_vol_window is None and "vol_lookback" in row:
            self.benchmark_vol_window = int(row["vol_lookback"])

    def _validate_and_align(self):
        names = list(self.inputs)
        first = self.inputs[names[0]]

        for other_name in names[1:]:
            other = self.inputs[other_name]
            if first.timeframe != other.timeframe:
                warnings.warn(
                    f"Strategy timeframe mismatch: {first.name} has {_format_timedelta(first.timeframe)} "
                    f"and {other.name} has {_format_timedelta(other.timeframe)}."
                )

            if not first.frame.index.equals(other.frame.index):
                missing_from_other = first.frame.index.difference(other.frame.index)
                missing_from_first = other.frame.index.difference(first.frame.index)
                warnings.warn(
                    f"Strategy index mismatch between {first.name} and {other.name}. "
                    f"Using the common active timestamps only. "
                    f"{len(missing_from_other)} timestamp(s) exist only in {first.name}; "
                    f"{len(missing_from_first)} timestamp(s) exist only in {other.name}."
                )

            first_symbols = set(first.positions.columns)
            other_symbols = set(other.positions.columns)
            if first_symbols != other_symbols:
                warnings.warn(
                    f"Strategy symbol mismatch between {first.name} and {other.name}. "
                    "Using common symbols only."
                )

        symbol_sets = [set(input_data.positions.columns) for input_data in self.inputs.values()]
        self.symbols = sorted(set.intersection(*symbol_sets))
        if not self.symbols:
            raise ValueError("No common symbols were found across strategy files.")

        index_sets = [input_data.frame.index for input_data in self.inputs.values()]
        common_index = index_sets[0]
        for index in index_sets[1:]:
            common_index = common_index.intersection(index)
        common_index = common_index.sort_values()
        if common_index.empty:
            raise ValueError("No common timestamps were found across strategy files.")

        active_mask = pd.Series(True, index=common_index)
        for input_data in self.inputs.values():
            active_mask &= input_data.active.reindex(common_index).fillna(False)

        self.common_index = common_index
        self.active_index = common_index[active_mask.to_numpy()]
        if self.active_index.empty:
            raise ValueError("No common timestamps are active in every strategy file.")

        reference = self.inputs[names[0]].closes.reindex(self.active_index)[self.symbols]
        for other_name in names[1:]:
            other_close = self.inputs[other_name].closes.reindex(self.active_index)[self.symbols]
            diff = ((other_close - reference).abs() / reference.replace(0.0, np.nan).abs()).max().max()
            if pd.notna(diff) and diff > 1e-8:
                warnings.warn(
                    f"Close prices differ between {names[0]} and {other_name}; "
                    f"maximum relative difference on common active rows is {diff:.3g}."
                )

        return self.active_index

    def _normalised_weights(self):
        weights = pd.Series(self.weights, dtype=float)
        missing = set(self.inputs) - set(weights.index)
        if missing:
            raise ValueError(f"Missing ensemble weights for: {sorted(missing)}")
        weights = weights.reindex(self.inputs.keys())
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Ensemble weights must sum to a positive number.")
        return weights / total

    def _combination_method_name(self):
        if callable(self.combination_method):
            return getattr(self.combination_method, "__name__", "custom")
        return str(self.combination_method)

    def _strategy_label(self):
        weights = self._normalised_weights()
        equal_weight = len(weights) > 0 and np.allclose(
            weights.to_numpy(),
            np.full(len(weights), 1.0 / len(weights)),
        )
        method = self._combination_method_name()
        if method in {"weighted_average", "equal_weight"} and equal_weight:
            if len(weights) == 2:
                percentage = int(round(100 / len(weights)))
                return f"Assembling {percentage}/{percentage}"
            return "Assembling equal-weight"
        return f"Assembling {method}"

    def _combine_positions(self, component_positions, weights):
        method = self.combination_method
        if callable(method):
            combined = method(component_positions=component_positions, weights=weights)
            combined = pd.DataFrame(combined).reindex(self.active_index)[self.symbols]
            return combined.astype(float).fillna(0.0)

        method = str(method)
        if method == "equal_weight":
            weights = pd.Series(
                1.0 / len(component_positions),
                index=list(component_positions),
                dtype=float,
            )
        elif method != "weighted_average":
            raise ValueError(
                "Unsupported combination_method. Use 'weighted_average', 'equal_weight', "
                "or pass a callable."
            )

        combined = pd.DataFrame(0.0, index=self.active_index, columns=self.symbols)
        for name, positions in component_positions.items():
            combined = combined.add(positions * weights.loc[name], fill_value=0.0)
        return combined.fillna(0.0)

    def _build_summary_meta(self, label=None, weighting=None):
        weighting = weighting if weighting is not None else self._normalised_weights()
        return {
            "strategy": label or "assembling",
            "combination_method": self._combination_method_name(),
            "components": ",".join(self.inputs.keys()),
            "ticker": ",".join(self.symbols),
            "start": self.active_index[0],
            "end": self.active_index[-1],
            "fees": self.fees,
            "weighting": ",".join(
                f"{name}:{weight:.4f}" for name, weight in weighting.items()
            ),
            "common_rows": len(self.common_index),
            "active_rows": len(self.active_index),
        }

    def _build_comparison(self):
        if self.summary.empty or self.buy_and_hold_summary.empty:
            self.comparison = pd.DataFrame()
            return self.comparison

        rows = []
        if not self.component_summaries.empty:
            rows.extend(self.component_summaries.to_dict(orient="records"))

        for label, summary in list(self.variant_summaries.groupby("strategy", sort=False)) + [
            ("B&H", self.buy_and_hold_summary),
        ]:
            row = summary.iloc[0]
            rows.append(
                {
                    "strategy": label,
                    "yearly_factor": row.get("yearly_factor"),
                    "max_drawdown": row.get("max_drawdown"),
                    "sharpe_ratio_annualized": row.get("sharpe_ratio_annualized"),
                    "winrate": row.get("winrate"),
                    "total_fees": row.get("total_fees"),
                    "turnover": row.get("turnover"),
                }
            )
        self.comparison = pd.DataFrame(rows)
        return self.comparison

    def _variant_label_50_50(self):
        return self._strategy_label()

    def _variant_label_volatility(self):
        return "Assembling volatility target"

    def _variant_label_rolling_sharpe(self):
        return "Assembling rolling Sharpe"

    def _component_portfolio_series(self, input_data, column):
        if ("portfolio", column) not in input_data.frame.columns:
            return pd.Series(index=self.active_index, dtype=float)
        return pd.to_numeric(input_data.frame[("portfolio", column)], errors="coerce").reindex(self.active_index)

    def _build_component_returns(self):
        columns = {}
        for name, input_data in self.inputs.items():
            columns[name] = self._component_portfolio_series(input_data, "net_strategy_return")
        self.component_returns = pd.DataFrame(columns, index=self.active_index).astype(float)
        return self.component_returns

    def _portfolio_returns_from_positions(self, positions, asset_returns):
        previous_positions = positions.shift(1).fillna(0.0)
        gross_by_symbol = previous_positions * asset_returns
        turnover_by_symbol = (positions - previous_positions).abs()
        fee_by_symbol = turnover_by_symbol * float(self.fees)
        portfolio_returns = gross_by_symbol.sum(axis=1)
        portfolio_fees = fee_by_symbol.sum(axis=1)
        return previous_positions, gross_by_symbol, turnover_by_symbol, fee_by_symbol, portfolio_returns, portfolio_fees

    def _evaluate_positions(self, label, positions, asset_returns, weighting=None):
        positions = positions.reindex(self.active_index)[self.symbols].astype(float)
        (
            previous_positions,
            gross_by_symbol,
            turnover_by_symbol,
            fee_by_symbol,
            portfolio_returns,
            portfolio_fees,
        ) = self._portfolio_returns_from_positions(positions, asset_returns)

        summary_data, summary = summarize_returns(
            init_amount=self.init_amount,
            strategy_returns=portfolio_returns - portfolio_fees,
            fee_cost=portfolio_fees,
            summary_meta=self._build_summary_meta(label=label, weighting=weighting),
            active_mask=previous_positions.abs().sum(axis=1).gt(0),
        )

        frames = {}
        for symbol in self.symbols:
            symbol_frame = pd.DataFrame(index=self.active_index)
            symbol_frame["close"] = self.closes[symbol]
            symbol_frame["return"] = asset_returns[symbol]
            for name, component_position in self.component_positions.items():
                symbol_frame[f"{name}_position"] = component_position[symbol]
            symbol_frame["position"] = positions[symbol]
            symbol_frame["position_prev"] = previous_positions[symbol]
            symbol_frame["gross_strategy_return"] = gross_by_symbol[symbol]
            symbol_frame["turnover"] = turnover_by_symbol[symbol]
            symbol_frame["fee_cost"] = fee_by_symbol[symbol]
            frames[symbol] = symbol_frame

        portfolio_frame = pd.DataFrame(index=self.active_index)
        portfolio_frame["net_strategy_return"] = summary_data["net_strategy_return"]
        portfolio_frame["net_log_return"] = summary_data["net_log_return"]
        portfolio_frame["turnover"] = turnover_by_symbol.sum(axis=1)
        portfolio_frame["fee_cost"] = summary_data["fee_cost"]
        portfolio_frame["cum_fees"] = summary_data["cum_fees"]
        portfolio_frame["wealth"] = summary_data["wealth"]
        portfolio_frame["running_peak"] = summary_data["running_peak"]
        portfolio_frame["drawdown%"] = summary_data["drawdown%"]
        frames["portfolio"] = portfolio_frame

        data = pd.concat(frames, axis=1)
        summary["turnover"] = portfolio_frame["turnover"].mean()
        return data, summary

    def _component_names_for_pair_methods(self):
        names = list(self.inputs)
        if len(names) != 2:
            raise ValueError("Volatility and rolling-Sharpe ensemble methods currently require exactly two strategies.")
        momentum_names = [name for name in names if "momentum" in name.lower()]
        if momentum_names:
            first = momentum_names[0]
            second = next(name for name in names if name != first)
            return first, second
        return names[0], names[1]

    def _build_volatility_positions(self, asset_returns):
        if self.volatility_window is None:
            return None, None

        window = int(self.volatility_window)
        if window < 2:
            raise ValueError("volatility_window must be at least 2.")

        periods_per_year = estimate_periods_per_year(self.active_index)
        if not np.isfinite(periods_per_year):
            raise ValueError("Not enough timestamps to annualize volatility.")

        component_vol = (
            self.component_returns.rolling(window=window, min_periods=window).std().shift(1)
            * np.sqrt(periods_per_year)
        )
        inv_vol = 1.0 / component_vol.replace(0.0, np.nan)
        weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)

        raw_positions = pd.DataFrame(0.0, index=self.active_index, columns=self.symbols)
        for name, positions in self.component_positions.items():
            raw_positions = raw_positions.add(positions.mul(weights[name], axis=0), fill_value=0.0)
        raw_positions = raw_positions.where(weights.notna().all(axis=1), np.nan)

        if self.volatility_target is not None:
            raw_returns = self._portfolio_returns_from_positions(raw_positions.fillna(0.0), asset_returns)[4]
            raw_vol = raw_returns.rolling(window=window, min_periods=window).std().shift(1) * np.sqrt(periods_per_year)
            scaler = float(self.volatility_target) / raw_vol.replace(0.0, np.nan)
            positions = raw_positions.mul(scaler, axis=0)
        else:
            positions = raw_positions

        return positions, weights

    def _build_rolling_sharpe_positions(self):
        if self.rolling_sharpe_window is None:
            return None, None

        window = int(self.rolling_sharpe_window)
        if window < 2:
            raise ValueError("rolling_sharpe_window must be at least 2.")

        grid = self.rolling_sharpe_weight_grid
        if grid is None:
            grid = np.linspace(0.0, 1.0, 11)
        grid = np.array(list(grid), dtype=float)
        if grid.size == 0 or np.any((grid < 0.0) | (grid > 1.0)):
            raise ValueError("rolling_sharpe_weight_grid must contain weights between 0 and 1.")

        momentum_name, other_name = self._component_names_for_pair_methods()
        selected = pd.Series(np.nan, index=self.active_index, dtype=float)
        returns = self.component_returns[[momentum_name, other_name]]

        for i in range(window, len(self.active_index)):
            history = returns.iloc[i - window : i]
            best_weight = np.nan
            best_sharpe = -np.inf
            for weight in grid:
                candidate = weight * history[momentum_name] + (1.0 - weight) * history[other_name]
                std = candidate.std()
                if not np.isfinite(std) or std <= 0:
                    sharpe = -np.inf
                else:
                    sharpe = candidate.mean() / std
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_weight = weight
            selected.iloc[i] = best_weight

        positions = (
            self.component_positions[momentum_name].mul(selected, axis=0)
            + self.component_positions[other_name].mul(1.0 - selected, axis=0)
        )
        weights = pd.DataFrame(
            {
                momentum_name: selected,
                other_name: 1.0 - selected,
            },
            index=self.active_index,
        )
        return positions, weights

    def _build_component_results(self):
        summary_rows = []
        wealth_columns = {}

        for name, input_data in self.inputs.items():
            returns = self._component_portfolio_series(input_data, "net_strategy_return").fillna(0.0)
            fees = self._component_portfolio_series(input_data, "fee_cost").fillna(0.0)
            if returns.empty:
                continue

            if ("portfolio", "active_assets") in input_data.frame.columns:
                active_mask = (
                    pd.to_numeric(input_data.frame[("portfolio", "active_assets")], errors="coerce")
                    .reindex(self.active_index)
                    .fillna(0)
                    .gt(0)
                )
            else:
                active_mask = returns.ne(0.0)

            component_data, component_summary = summarize_returns(
                init_amount=self.init_amount,
                strategy_returns=returns,
                fee_cost=fees,
                summary_meta={
                    "strategy": name,
                    "start": self.active_index[0],
                    "end": self.active_index[-1],
                },
                active_mask=active_mask,
            )
            turnover = self._component_portfolio_series(input_data, "turnover").fillna(0.0)
            component_summary["turnover"] = turnover.mean()
            row = component_summary.iloc[0].to_dict()
            summary_rows.append(
                {
                    "strategy": name,
                    "yearly_factor": row.get("yearly_factor"),
                    "max_drawdown": row.get("max_drawdown"),
                    "sharpe_ratio_annualized": row.get("sharpe_ratio_annualized"),
                    "winrate": row.get("winrate"),
                    "total_fees": row.get("total_fees"),
                    "turnover": row.get("turnover"),
                }
            )
            wealth_columns[name] = component_data["wealth"]

        self.component_summaries = pd.DataFrame(summary_rows)
        self.component_wealth = pd.DataFrame(wealth_columns, index=self.active_index)
        return self.component_summaries

    def run(self):
        self._load_inputs()
        self._infer_defaults_from_summary()
        if self.fees is None:
            self.fees = 0.0005
        if self.benchmark_target_vol is None:
            self.benchmark_target_vol = 1.0
        if self.benchmark_vol_window is None:
            self.benchmark_vol_window = 21

        self._validate_and_align()
        weights = self._normalised_weights()

        reference_name = next(iter(self.inputs))
        self.closes = self.inputs[reference_name].closes.reindex(self.active_index)[self.symbols]
        self.component_positions = {}
        for name, input_data in self.inputs.items():
            positions = input_data.positions.reindex(self.active_index)[self.symbols]
            self.component_positions[name] = positions

        self._build_component_returns()

        base_positions = self._combine_positions(self.component_positions, weights)
        self.variant_positions = {self._variant_label_50_50(): base_positions}
        self.variant_weights = {
            self._variant_label_50_50(): pd.DataFrame(
                {name: weights.loc[name] for name in self.inputs},
                index=self.active_index,
            )
        }

        asset_returns = self.closes.pct_change().fillna(0.0)
        volatility_positions, volatility_weights = self._build_volatility_positions(asset_returns)
        if volatility_positions is not None:
            self.variant_positions[self._variant_label_volatility()] = volatility_positions
            self.variant_weights[self._variant_label_volatility()] = volatility_weights

        rolling_positions, rolling_weights = self._build_rolling_sharpe_positions()
        if rolling_positions is not None:
            self.variant_positions[self._variant_label_rolling_sharpe()] = rolling_positions
            self.variant_weights[self._variant_label_rolling_sharpe()] = rolling_weights

        assigned_mask = pd.Series(True, index=self.active_index)
        for positions in self.variant_positions.values():
            assigned_mask &= positions[self.symbols].notna().all(axis=1)
        if not assigned_mask.any():
            raise ValueError("No common timestamps remain after ensemble method warmup.")
        if not assigned_mask.all():
            self.active_index = self.active_index[assigned_mask.to_numpy()]
            self.closes = self.closes.reindex(self.active_index)
            self.component_returns = self.component_returns.reindex(self.active_index)
            self.component_positions = {
                name: positions.reindex(self.active_index)
                for name, positions in self.component_positions.items()
            }
            self.variant_positions = {
                name: positions.reindex(self.active_index)
                for name, positions in self.variant_positions.items()
            }
            self.variant_weights = {
                name: method_weights.reindex(self.active_index)
                for name, method_weights in self.variant_weights.items()
            }
            asset_returns = self.closes.pct_change().fillna(0.0)

        self.variant_data = {}
        summary_rows = []
        for label, positions in self.variant_positions.items():
            average_weights = self.variant_weights[label].mean(numeric_only=True).dropna()
            data, summary = self._evaluate_positions(
                label,
                positions.fillna(0.0),
                asset_returns,
                weighting=average_weights,
            )
            self.variant_data[label] = data
            summary_rows.append(summary.iloc[0].to_dict())

        self.variant_summaries = pd.DataFrame(summary_rows)
        self.variant_wealth = pd.DataFrame(
            {
                label: data["portfolio"]["wealth"]
                for label, data in self.variant_data.items()
            },
            index=self.active_index,
        )
        self.data = self.variant_data[self._variant_label_50_50()]
        self.summary = self.variant_summaries.loc[
            self.variant_summaries["strategy"].eq(self._variant_label_50_50())
        ].reset_index(drop=True)
        self.combined_positions = self.variant_positions[self._variant_label_50_50()]
        self._build_component_results()

        close_source = {symbol: self.closes[symbol] for symbol in self.symbols}
        self.buy_and_hold_data, self.buy_and_hold_summary = calculate_buy_and_hold_baseline(
            close_source=close_source,
            init_amount=self.init_amount,
            target_vol=self.benchmark_target_vol,
            vol_window=self.benchmark_vol_window,
            fees=self.fees,
            summary_meta={
                "strategy": "buy_and_hold",
                "ticker": ",".join(self.symbols),
                "start": self.active_index[0],
                "end": self.active_index[-1],
                "benchmark": "equal_weight_rebalanced",
                "fees": self.fees,
            },
        )
        self.summary["B&H_yearly_factor"] = self.buy_and_hold_summary["yearly_factor"]
        self.summary["B&H_max_drawdown"] = self.buy_and_hold_summary["max_drawdown"]
        self.summary["B&H_sharpe_ratio_annualized"] = self.buy_and_hold_summary["sharpe_ratio_annualized"]
        self._build_comparison()
        return self.data

    def summary_table(self):
        if self.comparison.empty:
            self.run()
        return self.comparison.copy()

    def latest_positions(self):
        if self.data.empty:
            self.run()

        timestamp = self.active_index[-1]
        rows = []
        for symbol in self.symbols:
            row = {
                "available_at": timestamp,
                "applies_to": "next_bar",
                "SYMBOL": symbol,
                "close": self.closes.loc[timestamp, symbol],
            }
            for name, positions in self.component_positions.items():
                row[f"{name}_position"] = positions.loc[timestamp, symbol]
            for label, positions in self.variant_positions.items():
                column = label.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
                position = positions.loc[timestamp, symbol]
                row[f"{column}_position"] = position
                row[f"{column}_signal_side"] = (
                    "long" if position > 0 else "short" if position < 0 else "flat"
                )
            rows.append(row)

        return pd.DataFrame(rows).sort_values("SYMBOL").reset_index(drop=True)

    def plot_wealth(self, include_components=True):
        if self.data.empty:
            self.run()

        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(figsize=(12, 6))

        if include_components and not self.component_wealth.empty:
            for name in self.component_wealth.columns:
                ax.plot(
                    self.component_wealth.index,
                    self.component_wealth[name],
                    label=name,
                    linewidth=1.8,
                    alpha=0.65,
                )

        colors = ["black", "tab:blue", "tab:green", "tab:red", "tab:purple"]
        for color, (label, wealth) in zip(colors, self.variant_wealth.items()):
            ax.plot(
                wealth.index,
                wealth,
                label=label,
                linewidth=2.2 if label == self._variant_label_50_50() else 2.0,
                alpha=0.8,
                color=color,
            )

        benchmark_wealth = (
            self.buy_and_hold_data["portfolio"]["wealth"]
            if isinstance(self.buy_and_hold_data.columns, pd.MultiIndex)
            else self.buy_and_hold_data.get("wealth")
        )
        if benchmark_wealth is not None:
            ax.plot(
                benchmark_wealth.index,
                benchmark_wealth,
                label="B&H",
                linewidth=2.0,
                color="darkorange",
                alpha=0.75,
            )

        ax.set_yscale("log")
        ax.set_title("Equity curves comparison")
        ax.set_xlabel("Date")
        ax.set_ylabel("Wealth")
        ax.legend()
        plt.tight_layout()
        plt.show()
        return fig, ax

    def plot_assembled_wealth(self):
        if self.data.empty:
            self.run()
        benchmark_wealth = (
            self.buy_and_hold_data["portfolio"]["wealth"]
            if isinstance(self.buy_and_hold_data.columns, pd.MultiIndex)
            else self.buy_and_hold_data.get("wealth")
        )
        return plot_wealth(
            self.data["portfolio"]["wealth"],
            title="Equity curves comparison",
            log_scale=True,
            benchmark_wealth=benchmark_wealth,
            benchmark_label="B&H",
        )


EnsembleStrategy = AssemblingStrategy
