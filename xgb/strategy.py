from copy import deepcopy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (
    apply_xgb_cross_sectional_positions,
    build_search_grid,
    build_xgb_sleeve_frame,
    calculate_buy_and_hold_baseline,
    calculate_performance,
    combine_sleeve_frames,
    estimate_asset_annualized_volatility,
    estimate_periods_per_year,
    extract_close_map,
    fit_feature_pca,
    fit_feature_scaler,
    fit_xgb_model,
    make_xgb_model_frame,
    make_walkforward_splits,
    normalize_symbol_input,
    plot_wealth,
    prediction_metrics,
    select_split_frame,
    to_xy,
    transform_with_pca,
    transform_with_scaler,
    tune_xgb_model,
)

from .data import (
    build_or_load_feature_panel,
    data_signature,
    feature_signature,
    fetch_or_load_raw_data,
    load_frame,
    output_dir_for,
    safe_run_name,
    save_frame,
)
from .diagnostics import FEATURE_GROUPS, build_diagnostics


class XGBStrategy:
    """Compact three-stage XGBoost strategy with local cache reuse."""

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
        output_root="local_outputs/xgb",
        run_name=None,
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
        self.run_name = run_name or safe_run_name(
            self.symbols, start, end, tf, hour=hour, hour_timezone=hour_timezone
        )
        self.output_dir = output_dir_for(self.run_name, root=output_root)

        self.raw_data = {}
        self.feature_panel = pd.DataFrame()
        self.model_frame = pd.DataFrame()
        self.feature_columns = []
        self.predictions = pd.DataFrame()
        self.walkforward_metrics = pd.DataFrame()
        self.best_params_by_split = {}
        self.overall_oos_metrics = {}
        self.parameter_schedule = pd.DataFrame()
        self.parameter_sweep = pd.DataFrame()
        self.split_performance = pd.DataFrame()
        self.data = pd.DataFrame()
        self.summary = pd.DataFrame()
        self.buy_and_hold_data = pd.DataFrame()
        self.buy_and_hold_summary = {}
        self.feature_ablation = pd.DataFrame()
        self.diagnostics_report = {}
        self.diagnostics_path = None
        self._oos_close_source = {}
        self._oos_evaluation_index = pd.DatetimeIndex([])
        self._suppress_schedule_prints = False

        self._set_walkforward_params()
        self._set_backtest_params()

    def _set_walkforward_params(
        self,
        *,
        train_size=None,
        tune_size=None,
        test_size=None,
        step_size=None,
        mode=None,
        refit_on_train_plus_tune=None,
        model_params=None,
        tuning_grid=None,
        max_trials=None,
        random_state=None,
        pca_enabled=None,
        pca_n_components=None,
        tuning_min_side_balance=None,
        metric_filter_quantile=None,
        tuning_score_tolerance=None,
        tuning_min_prediction_unique=None,
        tuning_min_prediction_std=None,
        split_end=None,
    ):
        values = {
            "train_size": train_size,
            "tune_size": tune_size,
            "test_size": test_size,
            "step_size": step_size,
            "mode": mode,
            "refit_on_train_plus_tune": refit_on_train_plus_tune,
            "model_params": deepcopy(model_params) if model_params is not None else None,
            "tuning_grid": deepcopy(tuning_grid) if tuning_grid is not None else None,
            "max_trials": max_trials,
            "random_state": random_state,
            "pca_enabled": pca_enabled,
            "pca_n_components": pca_n_components,
            "tuning_min_side_balance": tuning_min_side_balance,
            "metric_filter_quantile": metric_filter_quantile,
            "tuning_score_tolerance": tuning_score_tolerance,
            "tuning_min_prediction_unique": tuning_min_prediction_unique,
            "tuning_min_prediction_std": tuning_min_prediction_std,
            "split_end": split_end,
        }
        for name, value in values.items():
            if value is not None or not hasattr(self, name):
                setattr(self, name, value)

    def _set_backtest_params(
        self,
        *,
        fees=None,
        target_vol=None,
        vol_lookback=None,
        vol_lookback_min=None,
        vol_lookback_max=None,
        vol_lookback_step=None,
        leverage_cap=None,
        threshold_quantile=None,
        threshold_quantile_min=None,
        threshold_quantile_max=None,
        threshold_quantile_step=None,
        evaluation_start_split=None,
        init_amount=None,
    ):
        if threshold_quantile is not None and not 0.0 <= float(threshold_quantile) <= 1.0:
            raise ValueError("threshold_quantile must be between 0 and 1")
        if threshold_quantile_min is not None and not 0.0 <= float(threshold_quantile_min) <= 1.0:
            raise ValueError("threshold_quantile_min must be between 0 and 1")
        if threshold_quantile_max is not None and not 0.0 <= float(threshold_quantile_max) <= 1.0:
            raise ValueError("threshold_quantile_max must be between 0 and 1")
        values = {
            "fees": fees,
            "target_vol": target_vol,
            "vol_lookback": vol_lookback,
            "vol_lookback_min": vol_lookback_min,
            "vol_lookback_max": vol_lookback_max,
            "vol_lookback_step": vol_lookback_step,
            "leverage_cap": leverage_cap,
            "threshold_quantile": threshold_quantile,
            "threshold_quantile_min": threshold_quantile_min,
            "threshold_quantile_max": threshold_quantile_max,
            "threshold_quantile_step": threshold_quantile_step,
            "evaluation_start_split": evaluation_start_split,
            "init_amount": init_amount,
        }
        for name, value in values.items():
            if value is not None or not hasattr(self, name):
                setattr(self, name, value)

    def fetch_data(self, force=False, save=True):
        self.raw_data = fetch_or_load_raw_data(self, force=force, save=save)
        build_or_load_feature_panel(self, force=force, save=save)
        print(f"XGB cache directory: {self.output_dir}")
        return self.raw_data

    def _ensure_feature_panel(self):
        if self.feature_panel.empty or self.model_frame.empty or not self.feature_columns:
            build_or_load_feature_panel(self, force=False, save=True)
        return self.feature_panel

    def use_feature_groups(self, include=None, exclude=None):
        """Select feature groups for the next walk-forward run without rebuilding raw features."""
        self._ensure_feature_panel()
        include = set(include or [])
        exclude = set(exclude or [])
        groups = {
            name: [column for column in columns if column in self.feature_panel.columns]
            for name, columns in FEATURE_GROUPS.items()
        }
        unknown = (include | exclude) - set(groups)
        if unknown:
            raise ValueError(f"unknown feature group(s): {sorted(unknown)}")

        if include:
            selected = []
            for name in include:
                selected.extend(groups[name])
        else:
            selected = list(self.feature_columns)

        excluded_columns = set()
        for name in exclude:
            excluded_columns.update(groups[name])

        self.feature_columns = [column for column in selected if column not in excluded_columns]
        self.model_frame = make_xgb_model_frame(
            self.feature_panel,
            self.feature_columns,
            target_column="target_next_log_return",
        )
        self.predictions = pd.DataFrame()
        self.walkforward_metrics = pd.DataFrame()
        self.best_params_by_split = {}
        self.overall_oos_metrics = {}
        print(f"Using {len(self.feature_columns)} XGB feature(s). Excluded groups: {sorted(exclude) or 'none'}.")
        return self.feature_columns

    def _summary_meta(self):
        return {
            "ticker": self.ticker_label,
            "start": pd.to_datetime(self.start),
            "end": pd.to_datetime(self.end),
            "tf": self.tf,
            "fees": self.fees,
            "target_vol": self.target_vol,
            "vol_lookback": self.vol_lookback,
            "leverage_cap": self.leverage_cap,
            "threshold_quantile": self.threshold_quantile,
            "evaluation_start_split": self.evaluation_start_split,
            "train_size": self.train_size,
            "tune_size": self.tune_size,
            "test_size": self.test_size,
            "step_size": self.step_size,
            "mode": self.mode,
            "hour": self.hour,
            "hour_timezone": self.hour_timezone,
            "run_name": self.run_name,
        }

    def _format_seconds(self, seconds):
        minutes, sec = divmod(max(int(seconds), 0), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:d}h {minutes:02d}m {sec:02d}s" if hours else f"{minutes:02d}m {sec:02d}s"

    def _format_eta(self, start_time, completed, total):
        if completed <= 0 or total <= completed:
            return "ETA 00m 00s"
        elapsed = time.perf_counter() - start_time
        remaining = (elapsed / completed) * (total - completed)
        return f"ETA {self._format_seconds(remaining)}"

    def _walkforward_config(self):
        return {
            "symbols": self.symbols,
            "start": str(self.start),
            "end": str(self.end),
            "tf": self.tf,
            "hour": self.hour,
            "hour_timezone": self.hour_timezone,
            "feature_columns": list(self.feature_columns),
            "train_size": self.train_size,
            "tune_size": self.tune_size,
            "test_size": self.test_size,
            "step_size": self.step_size,
            "mode": self.mode,
            "refit_on_train_plus_tune": self.refit_on_train_plus_tune,
            "model_params": self.model_params,
            "tuning_grid": self.tuning_grid,
            "max_trials": self.max_trials,
            "random_state": self.random_state,
            "pca_enabled": self.pca_enabled,
            "pca_n_components": self.pca_n_components,
            "tuning_min_side_balance": self.tuning_min_side_balance,
            "metric_filter_quantile": self.metric_filter_quantile,
            "tuning_score_tolerance": self.tuning_score_tolerance,
            "tuning_min_prediction_unique": self.tuning_min_prediction_unique,
            "tuning_min_prediction_std": self.tuning_min_prediction_std,
            "split_end": self.split_end,
            "threshold_calibration": "previous_split_absolute_value_v1",
        }

    def _walkforward_model_frame(self):
        if self.split_end is None:
            return self.model_frame
        cutoff_date = pd.Timestamp(self.split_end).date()
        trade_dates = pd.to_datetime(self.model_frame["trade_date"]).dt.date
        return self.model_frame.loc[trade_dates <= cutoff_date].copy()

    def _save_walkforward_outputs(self):
        save_frame(self.predictions, self.output_dir / "predictions.parquet")
        self.predictions.to_csv(self.output_dir / "predictions.csv", index=False)
        self.walkforward_metrics.to_csv(self.output_dir / "walkforward_metrics.csv", index=False)
        for name in [
            "parameter_schedule.csv",
            "parameter_sweep.csv",
            "split_performance.csv",
            "strategy_data.csv",
            "buy_and_hold_data.csv",
            "summary.csv",
        ]:
            path = self.output_dir / name
            if path.exists():
                path.unlink()
        self.parameter_schedule = pd.DataFrame()
        self.parameter_sweep = pd.DataFrame()
        self.split_performance = pd.DataFrame()
        self.data = pd.DataFrame()
        self.summary = pd.DataFrame()
        self.buy_and_hold_data = pd.DataFrame()
        self.buy_and_hold_summary = {}
        self._save_run_diagnostics()

    def _load_predictions(self, require_config_match=False):
        diagnostics_path = self.output_dir / "xgb_run_diagnostics.json"
        if require_config_match:
            if diagnostics_path.exists():
                diagnostics = json.loads(diagnostics_path.read_text())
                cached_config = diagnostics.get("walkforward_config") or diagnostics.get("walkforward", {})
            else:
                config_path = self.output_dir / "walkforward_config.json"
                if not config_path.exists():
                    return False
                cached_config = json.loads(config_path.read_text())
            current_config = self._json_safe(self._walkforward_config())
            if cached_config != current_config:
                return False

        path = self.output_dir / "predictions.parquet"
        if not path.exists() and not path.with_suffix(".pkl").exists():
            return False
        self.predictions = load_frame(path)
        metrics_path = self.output_dir / "walkforward_metrics.csv"
        params_path = self.output_dir / "best_params_by_split.json"
        overall_path = self.output_dir / "overall_oos_metrics.json"
        if metrics_path.exists():
            try:
                self.walkforward_metrics = pd.read_csv(metrics_path)
            except pd.errors.EmptyDataError:
                self.walkforward_metrics = pd.DataFrame()
        if params_path.exists():
            self.best_params_by_split = json.loads(params_path.read_text())
        if overall_path.exists():
            self.overall_oos_metrics = json.loads(overall_path.read_text())
        if diagnostics_path.exists():
            diagnostics = json.loads(diagnostics_path.read_text())
            self.best_params_by_split = diagnostics.get("best_params_by_split", self.best_params_by_split)
            self.overall_oos_metrics = diagnostics.get("overall_oos_metrics", self.overall_oos_metrics)
        return True

    def _predictions_cover_splits(self, splits):
        if self.predictions.empty or "trade_date" not in self.predictions.columns:
            return False
        expected_days = pd.Index(day for split in splits for day in split.test_days).drop_duplicates()
        predicted_days = pd.Index(pd.to_datetime(self.predictions["trade_date"]).dropna().unique())
        return expected_days.difference(predicted_days).empty

    def run_walkforward(
        self,
        *,
        train_size=None,
        tune_size=None,
        test_size=None,
        step_size=None,
        mode=None,
        refit_on_train_plus_tune=None,
        model_params=None,
        tuning_grid=None,
        max_trials=None,
        random_state=None,
        pca_enabled=None,
        pca_n_components=None,
        tuning_min_side_balance=None,
        metric_filter_quantile=None,
        tuning_score_tolerance=None,
        tuning_min_prediction_unique=None,
        tuning_min_prediction_std=None,
        split_end=None,
        force=False,
    ):
        self._set_walkforward_params(
            train_size=train_size,
            tune_size=tune_size,
            test_size=test_size,
            step_size=step_size,
            mode=mode,
            refit_on_train_plus_tune=refit_on_train_plus_tune,
            model_params=model_params,
            tuning_grid=tuning_grid,
            max_trials=max_trials,
            random_state=random_state,
            pca_enabled=pca_enabled,
            pca_n_components=pca_n_components,
            tuning_min_side_balance=tuning_min_side_balance,
            metric_filter_quantile=metric_filter_quantile,
            tuning_score_tolerance=tuning_score_tolerance,
            tuning_min_prediction_unique=tuning_min_prediction_unique,
            tuning_min_prediction_std=tuning_min_prediction_std,
            split_end=split_end,
        )
        required = [
            "train_size",
            "tune_size",
            "test_size",
            "step_size",
            "mode",
            "refit_on_train_plus_tune",
            "model_params",
            "tuning_grid",
            "max_trials",
            "random_state",
            "pca_enabled",
            "pca_n_components",
            "tuning_min_side_balance",
            "metric_filter_quantile",
            "tuning_score_tolerance",
            "tuning_min_prediction_unique",
            "tuning_min_prediction_std",
        ]
        missing = [name for name in required if getattr(self, name) is None]
        if missing:
            raise ValueError(
                "Pass private walk-forward parameters explicitly from the ignored notebook: "
                + ", ".join(missing)
            )
        self._ensure_feature_panel()
        split_model_frame = self._walkforward_model_frame()
        splits = make_walkforward_splits(
            split_model_frame,
            train_size=self.train_size,
            tune_size=self.tune_size,
            test_size=self.test_size,
            step_size=self.step_size,
            mode=self.mode,
        )
        if not force and self.predictions.empty and self._load_predictions(require_config_match=True):
            if self._predictions_cover_splits(splits):
                print(f"Loaded cached XGB predictions from {self.output_dir}.")
                return self.predictions
            print("Cached XGB predictions do not cover the current walk-forward split schedule; recomputing.")
            self.predictions = pd.DataFrame()

        prediction_frames = []
        metric_rows = []
        best_params_by_split = {}
        latest_model = latest_scaler = latest_pca = None
        latest_modeling_columns = self.feature_columns
        run_start = time.perf_counter()
        print(f"Starting XGBoost walk-forward with {len(splits)} split(s).")

        for split_idx, split in enumerate(splits, start=1):
            split_start = time.perf_counter()
            print(
                f"Running split {split_idx}/{len(splits)} "
                f"(train {split.train_days[0]} -> {split.train_days[-1]}, "
                f"tune {split.tune_days[0]} -> {split.tune_days[-1]}, "
                f"test {split.test_days[0]} -> {split.test_days[-1]})."
            )
            train_df = select_split_frame(split_model_frame, split.train_days)
            tune_df = select_split_frame(split_model_frame, split.tune_days)
            test_df = select_split_frame(split_model_frame, split.test_days)

            scaler = fit_feature_scaler(train_df, self.feature_columns)
            train_scaled = transform_with_scaler(train_df, scaler, self.feature_columns)
            tune_scaled = transform_with_scaler(tune_df, scaler, self.feature_columns)
            test_scaled = transform_with_scaler(test_df, scaler, self.feature_columns)

            modeling_columns = self.feature_columns
            pca = None
            if self.pca_enabled:
                pca, modeling_columns = fit_feature_pca(
                    train_scaled,
                    n_components=self.pca_n_components,
                    random_state=self.random_state,
                )
                train_scaled = transform_with_pca(train_scaled, pca, modeling_columns)
                tune_scaled = transform_with_pca(tune_scaled, pca, modeling_columns)
                test_scaled = transform_with_pca(test_scaled, pca, modeling_columns)

            X_train, y_train = to_xy(
                pd.concat([train_scaled, train_df[["target_next_log_return"]]], axis=1),
                modeling_columns,
                "target_next_log_return",
            )
            X_tune, y_tune = to_xy(
                pd.concat([tune_scaled, tune_df[["target_next_log_return"]]], axis=1),
                modeling_columns,
                "target_next_log_return",
            )
            search_result = tune_xgb_model(
                X_train,
                y_train,
                X_tune,
                y_tune,
                base_params=self.model_params,
                tuning_grid=self.tuning_grid,
                max_trials=self.max_trials,
                random_state=self.random_state,
                score_tolerance=self.tuning_score_tolerance,
                score_quantile=self.metric_filter_quantile,
                min_side_balance=self.tuning_min_side_balance,
                min_prediction_unique=self.tuning_min_prediction_unique,
                min_prediction_std=self.tuning_min_prediction_std,
            )
            best_params_by_split[str(split.split_id)] = deepcopy(search_result["best_params"])

            refit_df = (
                pd.concat([train_df, tune_df], ignore_index=True)
                if self.refit_on_train_plus_tune
                else train_df.copy()
            )
            refit_scaled = transform_with_scaler(refit_df, scaler, self.feature_columns)
            if self.pca_enabled:
                refit_scaled = transform_with_pca(refit_scaled, pca, modeling_columns)
            X_refit, y_refit = to_xy(
                pd.concat([refit_scaled, refit_df[["target_next_log_return"]]], axis=1),
                modeling_columns,
                "target_next_log_return",
            )

            model = fit_xgb_model(X_refit, y_refit, search_result["best_params"])
            predictions = model.predict(test_scaled[modeling_columns])
            latest_model, latest_scaler, latest_pca = model, scaler, pca
            latest_modeling_columns = modeling_columns

            test_output = test_df[
                [
                    "timestamp",
                    "trade_date",
                    "SYMBOL",
                    "close",
                    "simple_return",
                    "log_return",
                    "target_next_simple_return",
                    "target_next_log_return",
                ]
            ].copy()
            test_output["prediction"] = predictions
            test_output["split_id"] = split.split_id
            prediction_frames.append(test_output)

            metric_prefix = f"filtered_q{int(round(float(self.metric_filter_quantile) * 100))}"
            metrics = prediction_metrics(
                test_df["target_next_log_return"],
                predictions,
                metric_filter_quantile=self.metric_filter_quantile,
            )
            metrics["split_id"] = split.split_id
            metrics["tuning_metric"] = search_result.get("best_score_name")
            metrics[f"best_tune_{metric_prefix}_sign_accuracy"] = search_result["best_score"]
            metrics[f"best_train_{metric_prefix}_sign_accuracy"] = search_result.get("best_train_score")
            metrics[f"best_train_tune_{metric_prefix}_gap"] = search_result.get("best_overfit_gap")
            metrics[f"best_tune_{metric_prefix}_side_balance"] = search_result.get(
                "best_tune_filtered_side_balance"
            )
            metrics[f"best_train_{metric_prefix}_side_balance"] = search_result.get(
                "best_train_filtered_side_balance"
            )
            metrics["best_tune_oos_r2"] = search_result.get("best_tune_oos_r2")
            metrics["best_train_oos_r2"] = search_result.get("best_train_oos_r2")
            metrics["best_tune_prediction_unique"] = search_result.get("best_tune_prediction_unique")
            metrics["best_tune_prediction_std"] = search_result.get("best_tune_prediction_std")
            metric_rows.append(metrics)
            elapsed = time.perf_counter() - split_start
            print(
                f"Completed split {split_idx}/{len(splits)} in {self._format_seconds(elapsed)} "
                f"({self._format_eta(run_start, split_idx, len(splits))})."
            )

        live_frame = self.feature_panel.loc[self.feature_panel["target_next_log_return"].isna()].copy()
        if latest_model is not None and not live_frame.empty:
            live_frame = live_frame.dropna(subset=self.feature_columns).copy()
            if not live_frame.empty:
                live_scaled = transform_with_scaler(live_frame, latest_scaler, self.feature_columns)
                if self.pca_enabled and latest_pca is not None:
                    live_scaled = transform_with_pca(live_scaled, latest_pca, latest_modeling_columns)
                live_output = live_frame[
                    ["timestamp", "trade_date", "SYMBOL", "close", "simple_return", "log_return"]
                ].copy()
                live_output["target_next_simple_return"] = pd.NA
                live_output["target_next_log_return"] = pd.NA
                live_output["prediction"] = latest_model.predict(live_scaled[latest_modeling_columns])
                live_output["split_id"] = "live"
                prediction_frames.append(live_output)

        self.predictions = (
            pd.concat(prediction_frames, ignore_index=True)
            .sort_values(["timestamp", "SYMBOL"])
            .reset_index(drop=True)
        )
        self.walkforward_metrics = pd.DataFrame(metric_rows)
        self.best_params_by_split = best_params_by_split
        completed = self.predictions.loc[self.predictions["target_next_log_return"].notna()]
        self.overall_oos_metrics = (
            prediction_metrics(
                completed["target_next_log_return"],
                completed["prediction"],
                metric_filter_quantile=self.metric_filter_quantile,
            )
            if not completed.empty
            else {}
        )
        self._save_walkforward_outputs()
        print(f"Finished XGBoost walk-forward in {self._format_seconds(time.perf_counter() - run_start)}.")
        return self.predictions

    run1 = run_walkforward

    def _candidate_rank(self, summary_row):
        turnover = float(summary_row.get("turnover", 0.0) or 0.0)
        active_rows = float(summary_row.get("active_rows", 0.0) or 0.0)

        sharpe = float(summary_row.get("sharpe_ratio_annualized", float("nan")))
        yearly_factor = float(summary_row.get("yearly_factor", float("nan")))
        fees = float(summary_row.get("total_fees", float("inf")))

        if not np.isfinite(sharpe):
            sharpe = float("-inf")
        if not np.isfinite(yearly_factor):
            yearly_factor = float("-inf")
        if not np.isfinite(fees):
            fees = float("inf")

        if not np.isfinite(turnover) or turnover <= 0.0 or active_rows <= 0:
            return (3, 0.0, 0.0, 0.0, 0.0, float("inf"), float("inf"))

        return (
            0,
            -sharpe,
            -yearly_factor,
            fees,
            turnover,
        )

    def _threshold_value_for_quantile(self, prediction_frame, threshold_quantile):
        values = pd.to_numeric(prediction_frame["prediction"].abs(), errors="coerce").dropna()
        if values.empty:
            return float("inf")
        quantile = float(np.clip(threshold_quantile, 0.0, 1.0))
        return float(values.quantile(quantile))

    def _active_signal_mask(self, prediction_frame, threshold_quantile):
        predictions = pd.to_numeric(prediction_frame["prediction"], errors="coerce")
        valid = predictions.notna() & predictions.ne(0.0)
        mask = pd.Series(False, index=prediction_frame.index)
        if not valid.any():
            return mask
        keep_fraction = 1.0 - float(np.clip(threshold_quantile, 0.0, 1.0))
        keep_n = max(1, int(np.ceil(valid.sum() * keep_fraction - 1e-12)))
        ranked_index = predictions.loc[valid].abs().sort_values(ascending=False, kind="mergesort").index
        mask.loc[ranked_index[:keep_n]] = True
        return mask

    def _deduplicate_prediction_bars(self, prediction_frame):
        frame = prediction_frame.copy()
        if not frame.duplicated(["timestamp", "SYMBOL"]).any():
            return frame
        frame["_split_sort_key"] = pd.to_numeric(frame["split_id"], errors="coerce").fillna(np.inf)
        frame = (
            frame.sort_values(["timestamp", "SYMBOL", "_split_sort_key"])
            .drop_duplicates(["timestamp", "SYMBOL"], keep="last")
            .drop(columns="_split_sort_key")
            .reset_index(drop=True)
        )
        return frame

    def _backtest_prediction_frame(self, prediction_frame, threshold_quantile, vol_lookback):
        vol_col = f"asset_vol_annualized_{int(vol_lookback)}"
        prediction_frame = prediction_frame.copy()
        prediction_frame["is_active_signal"] = self._active_signal_mask(prediction_frame, threshold_quantile)
        positioned = apply_xgb_cross_sectional_positions(
            prediction_frame,
            target_vol=self.target_vol,
            leverage_cap=self.leverage_cap,
            vol_col=vol_col,
            prediction_col="prediction",
            active_col="is_active_signal",
        )
        positioned["asset_vol_annualized"] = positioned[vol_col]
        return self._evaluate_positioned_frame(positioned)

    def _score_candidate_fast(self, prediction_frame, threshold_quantile, vol_lookback, threshold_value=None):
        vol_col = f"asset_vol_annualized_{int(vol_lookback)}"
        frame = self._deduplicate_prediction_bars(
            prediction_frame.sort_values(["timestamp", "SYMBOL"]).copy()
        )
        if threshold_value is None:
            signal_mask = self._active_signal_mask(frame, threshold_quantile)
        else:
            predictions = pd.to_numeric(frame["prediction"], errors="coerce")
            signal_mask = predictions.abs().ge(float(threshold_value)) & predictions.ne(0.0)
        active = (
            signal_mask
            & frame["prediction"].ne(0.0)
            & frame[vol_col].notna()
            & frame[vol_col].gt(0.0)
        )

        scale = (self.target_vol / frame[vol_col]).clip(upper=self.leverage_cap)
        frame["position"] = 0.0
        frame.loc[active, "position"] = np.sign(frame.loc[active, "prediction"]) * scale.loc[active]

        close = frame.pivot(index="timestamp", columns="SYMBOL", values="close").sort_index()
        positions = frame.pivot(index="timestamp", columns="SYMBOL", values="position").reindex(close.index).fillna(0.0)
        asset_returns = close.pct_change().fillna(0.0)
        active_assets = positions.ne(0.0).sum(axis=1)
        weights = positions.ne(0.0).div(active_assets.replace(0, np.nan), axis=0).fillna(0.0)
        weighted_positions = positions * weights

        gross_returns = (weighted_positions.shift(1).fillna(0.0) * asset_returns).sum(axis=1)
        turnover = positions.diff().abs()
        if not turnover.empty:
            turnover.iloc[0] = positions.iloc[0].abs()
        weighted_turnover = (turnover.fillna(0.0) * weights).sum(axis=1)
        fee_cost = weighted_turnover * self.fees
        net_returns = gross_returns - fee_cost

        periods_per_year = estimate_periods_per_year(net_returns.index)
        net_returns = net_returns.fillna(0.0).clip(lower=-0.999999)
        net_log_returns = np.log1p(net_returns)
        wealth = np.exp(net_log_returns.cumsum()) * self.init_amount
        running_peak = wealth.cummax()
        drawdown = wealth / running_peak - 1.0
        active_previous = weighted_positions.shift(1).fillna(0.0).abs().sum(axis=1) > 0
        active_returns = net_returns.loc[active_previous]
        active_logs = net_log_returns.loc[active_previous]

        ret_std = net_returns.std(ddof=1)
        sharpe = (
            (net_returns.mean() / ret_std) * np.sqrt(periods_per_year)
            if pd.notna(ret_std) and ret_std > 0 and np.isfinite(periods_per_year)
            else np.nan
        )
        elapsed_years = (
            (net_returns.index[-1] - net_returns.index[0]).total_seconds() / (365.25 * 24 * 60 * 60)
            if len(net_returns.index) >= 2
            else np.nan
        )
        yearly_factor = (
            (wealth.iloc[-1] / self.init_amount) ** (1 / elapsed_years)
            if pd.notna(elapsed_years) and elapsed_years > 0 and not wealth.empty
            else np.nan
        )
        return {
            "yearly_factor": yearly_factor,
            "total_fees": float(fee_cost.sum()),
            "turnover": float(weighted_turnover.mean()),
            "active_rows": int(active.sum()),
            "active_share": float(active.mean()),
            "max_drawdown": float(drawdown.min()) if not drawdown.empty else np.nan,
            "winrate": float((active_returns > 0).mean()) if not active_returns.empty else np.nan,
            "average_return_factor": float(np.exp(active_logs.mean())) if not active_logs.empty else np.nan,
            "sharpe_ratio_annualized": sharpe,
        }

    def _build_parameter_schedule(self, prediction_frame):
        threshold_quantiles = build_search_grid(
            self.threshold_quantile_min,
            self.threshold_quantile_max,
            self.threshold_quantile_step,
            cast=float,
        )
        lookbacks = build_search_grid(
            self.vol_lookback_min,
            self.vol_lookback_max,
            self.vol_lookback_step,
            cast=lambda value: int(round(value)),
        )
        candidates = [
            {"threshold_quantile": float(threshold_quantile), "vol_lookback": int(lookback)}
            for threshold_quantile in threshold_quantiles
            for lookback in lookbacks
        ]
        split_ids = sorted(
            prediction_frame.loc[prediction_frame["split_id"].astype(str) != "live", "split_id"]
            .dropna()
            .unique()
            .tolist()
        )
        schedule_rows = []
        sweep_rows = []
        for split_position, split_id in enumerate(split_ids):
            current_predictions = prediction_frame.loc[prediction_frame["split_id"] == split_id].copy()
            if split_position == 0:
                threshold_value = self._threshold_value_for_quantile(
                    current_predictions,
                    float(self.threshold_quantile),
                )
                schedule_rows.append(
                    {
                        "split_id": split_id,
                        "selection_source": "default",
                        "validation_split_count": 0,
                        "applied_threshold_quantile": float(self.threshold_quantile),
                        "applied_threshold_value": threshold_value,
                        "threshold_reference": "current_split_bootstrap",
                        "applied_vol_lookback": int(self.vol_lookback),
                    }
                )
                if not self._suppress_schedule_prints:
                    print(
                        f"Split {split_id}: default threshold quantile={self.threshold_quantile:.3f}, "
                        f"lookback={self.vol_lookback}."
                    )
                continue

            validation_ids = split_ids[:split_position]
            validation_predictions = prediction_frame.loc[prediction_frame["split_id"].isin(validation_ids)].copy()
            threshold_reference_id = validation_ids[-1]
            threshold_reference_predictions = prediction_frame.loc[
                prediction_frame["split_id"] == threshold_reference_id
            ].copy()
            rows = []
            for candidate in candidates:
                row = self._score_candidate_fast(
                    validation_predictions,
                    threshold_quantile=candidate["threshold_quantile"],
                    vol_lookback=candidate["vol_lookback"],
                )
                rows.append(
                    {
                        "selection_split_id": split_id,
                        "validation_split_count": len(validation_ids),
                        **candidate,
                        "yearly_factor": row.get("yearly_factor"),
                        "max_drawdown": row.get("max_drawdown"),
                        "winrate": row.get("winrate"),
                        "sharpe_ratio_annualized": row.get("sharpe_ratio_annualized"),
                        "total_fees": row.get("total_fees"),
                        "turnover": row.get("turnover"),
                        "active_rows": row.get("active_rows"),
                        "active_share": row.get("active_share"),
                    }
                )
            candidate_df = pd.DataFrame(rows)
            if (candidate_df["active_rows"].fillna(0) <= 0).all():
                default_mask = (
                    np.isclose(candidate_df["threshold_quantile"], float(self.threshold_quantile))
                    & (candidate_df["vol_lookback"].astype(int) == int(self.vol_lookback))
                )
                best_index = candidate_df.loc[default_mask].index[0] if default_mask.any() else candidate_df.index[0]
            else:
                best_index = min(candidate_df.index, key=lambda index: self._candidate_rank(candidate_df.loc[index]))
            candidate_df["selected_for_next_split"] = candidate_df.index == best_index
            sweep_rows.extend(candidate_df.to_dict(orient="records"))
            best = candidate_df.loc[best_index]
            threshold_value = self._threshold_value_for_quantile(
                threshold_reference_predictions,
                float(best["threshold_quantile"]),
            )
            schedule_rows.append(
                {
                    "split_id": split_id,
                    "selection_source": "expanding_validation",
                    "validation_split_count": len(validation_ids),
                    "applied_threshold_quantile": float(best["threshold_quantile"]),
                    "applied_threshold_value": threshold_value,
                    "threshold_reference": f"previous_split:{threshold_reference_id}",
                    "applied_vol_lookback": int(best["vol_lookback"]),
                    "selected_validation_sharpe_ratio_annualized": best.get("sharpe_ratio_annualized"),
                    "selected_validation_yearly_factor": best.get("yearly_factor"),
                    "selected_validation_total_fees": best.get("total_fees"),
                    "selected_validation_turnover": best.get("turnover"),
                }
            )
            if not self._suppress_schedule_prints:
                print(
                    f"Split {split_id}: selected threshold quantile={float(best['threshold_quantile']):.3f}, "
                    f"lookback={int(best['vol_lookback'])} from prior splits."
                )
        self.parameter_schedule = pd.DataFrame(schedule_rows).sort_values("split_id").reset_index(drop=True)
        self.parameter_sweep = pd.DataFrame(sweep_rows)
        return self.parameter_schedule

    def _evaluate_positioned_frame(self, positioned):
        sleeve_frames = {}
        single_summary = None
        for symbol, symbol_frame in positioned.groupby("SYMBOL", sort=False):
            sleeve_frame = build_xgb_sleeve_frame(symbol_frame)
            performance_data, summary = calculate_performance(
                init_amount=self.init_amount,
                returns=sleeve_frame["log_return"],
                positions=sleeve_frame["position"],
                fees=self.fees,
                log_return=True,
                summary_meta=self._summary_meta() | {"ticker": symbol},
            )
            sleeve_frames[symbol] = sleeve_frame.join(
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
            single_summary = summary
        if len(sleeve_frames) == 1:
            return sleeve_frames[next(iter(sleeve_frames))], single_summary
        return combine_sleeve_frames(
            sleeve_frames=sleeve_frames,
            init_amount=self.init_amount,
            fees=self.fees,
            summary_meta=self._summary_meta() | {"ticker": self.ticker_label},
        )

    def _build_split_performance(self, prediction_frame):
        rows = []
        for _, schedule_row in self.parameter_schedule.iterrows():
            split_id = schedule_row["split_id"]
            split_frame = prediction_frame.loc[prediction_frame["split_id"] == split_id].copy()
            if split_frame.empty:
                continue
            threshold_value = float(schedule_row["applied_threshold_value"])
            row = self._score_candidate_fast(
                split_frame,
                threshold_quantile=float(schedule_row["applied_threshold_quantile"]),
                vol_lookback=int(schedule_row["applied_vol_lookback"]),
                threshold_value=threshold_value,
            )
            rows.append(
                {
                    "split_id": split_id,
                    "applied_threshold_quantile": float(schedule_row["applied_threshold_quantile"]),
                    "applied_threshold_value": threshold_value,
                    "threshold_reference": schedule_row.get("threshold_reference"),
                    "applied_vol_lookback": int(schedule_row["applied_vol_lookback"]),
                    "included_in_summary": int(split_id) >= int(self.evaluation_start_split),
                    **row,
                }
            )
        self.split_performance = pd.DataFrame(rows)
        return self.split_performance

    def _evaluate_predictions(self):
        self._ensure_feature_panel()
        lookbacks = build_search_grid(
            self.vol_lookback_min,
            self.vol_lookback_max,
            self.vol_lookback_step,
            cast=lambda value: int(round(value)),
        )
        volatility_panel = self.feature_panel.loc[:, ["timestamp", "SYMBOL", "simple_return"]].copy()
        for lookback in lookbacks:
            volatility_panel = estimate_asset_annualized_volatility(
                volatility_panel,
                lookback_bars=lookback,
                return_col="simple_return",
                output_col=f"asset_vol_annualized_{lookback}",
            )

        prediction_frame = self.predictions.merge(
            volatility_panel.drop(columns=["simple_return"]),
            on=["timestamp", "SYMBOL"],
            how="left",
        )
        self._build_parameter_schedule(prediction_frame)
        self._build_split_performance(
            prediction_frame.loc[prediction_frame["split_id"].astype(str) != "live"].copy()
        )
        prediction_frame = prediction_frame.merge(
            self.parameter_schedule[
                [
                    "split_id",
                    "applied_threshold_quantile",
                    "applied_threshold_value",
                    "threshold_reference",
                    "applied_vol_lookback",
                ]
            ],
            on="split_id",
            how="left",
        )
        if "live" in prediction_frame["split_id"].astype(str).values and not self.parameter_schedule.empty:
            last = self.parameter_schedule.iloc[-1]
            live = prediction_frame["split_id"].astype(str) == "live"
            prediction_frame.loc[live, "applied_threshold_quantile"] = float(last["applied_threshold_quantile"])
            prediction_frame.loc[live, "applied_threshold_value"] = float(last["applied_threshold_value"])
            prediction_frame.loc[live, "threshold_reference"] = str(last["threshold_reference"])
            prediction_frame.loc[live, "applied_vol_lookback"] = int(last["applied_vol_lookback"])

        prediction_frame["is_active_signal"] = False
        for split_id, split_frame in prediction_frame.groupby("split_id", sort=False):
            if split_frame.empty or pd.isna(split_frame["applied_threshold_value"].iloc[0]):
                continue
            threshold_value = float(split_frame["applied_threshold_value"].iloc[0])
            predictions = pd.to_numeric(split_frame["prediction"], errors="coerce")
            active = predictions.abs().ge(threshold_value) & predictions.ne(0.0)
            prediction_frame.loc[split_frame.index, "is_active_signal"] = active.to_numpy()

        prediction_frame["asset_vol_annualized"] = prediction_frame.apply(
            lambda row: (
                row[f"asset_vol_annualized_{int(row['applied_vol_lookback'])}"]
                if pd.notna(row["applied_vol_lookback"])
                else pd.NA
            ),
            axis=1,
        )
        backtest_predictions = prediction_frame.loc[
            prediction_frame["split_id"].astype(str) != "live"
        ].copy()
        backtest_predictions = self._deduplicate_prediction_bars(backtest_predictions)
        evaluation_predictions = backtest_predictions.loc[
            pd.to_numeric(backtest_predictions["split_id"], errors="coerce").ge(
                int(self.evaluation_start_split)
            )
        ].copy()

        positioned = apply_xgb_cross_sectional_positions(
            evaluation_predictions,
            target_vol=self.target_vol,
            leverage_cap=self.leverage_cap,
            vol_col="asset_vol_annualized",
            prediction_col="prediction",
            threshold_value_col="applied_threshold_value",
            active_col="is_active_signal",
        )
        data, summary = self._evaluate_positioned_frame(positioned)

        close_map = {}
        for symbol, symbol_frame in self.feature_panel.groupby("SYMBOL", sort=False):
            close_map[symbol] = (
                symbol_frame.sort_values("timestamp")
                .set_index("timestamp")["close"]
                .astype(float)
            )
        self._oos_close_source = close_map
        self._oos_evaluation_index = pd.DatetimeIndex(
            pd.to_datetime(evaluation_predictions["timestamp"]).drop_duplicates().sort_values()
        )
        return data, summary

    def _save_threshold_outputs(self):
        self.parameter_schedule.to_csv(self.output_dir / "parameter_schedule.csv", index=False)
        self.parameter_sweep.to_csv(self.output_dir / "parameter_sweep.csv", index=False)
        self.split_performance.to_csv(self.output_dir / "split_performance.csv", index=False)
        self.summary.to_csv(self.output_dir / "summary.csv", index=False)
        if not self.data.empty:
            self.data.to_csv(self.output_dir / "strategy_data.csv")
        if not self.buy_and_hold_data.empty:
            self.buy_and_hold_data.to_csv(self.output_dir / "buy_and_hold_data.csv")
        self._save_run_diagnostics()

    def _json_safe(self, value):
        if isinstance(value, pd.DataFrame):
            records = value.astype(object).where(pd.notna(value), None).to_dict(orient="records")
            return self._json_safe(records)
        if isinstance(value, pd.Series):
            return self._json_safe(value.to_dict())
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        if value is pd.NA or (isinstance(value, float) and np.isnan(value)):
            return None
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return value

    def _save_run_diagnostics(self):
        payload = {
            "strategy": "xgb",
            "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "run_name": self.run_name,
            "output_dir": str(self.output_dir),
            "symbols": self.symbols,
            "start": str(self.start),
            "end": str(self.end),
            "tf": self.tf,
            "hour": self.hour,
            "hour_timezone": self.hour_timezone,
            "cache": {
                "data": data_signature(self),
                "features": feature_signature(self),
            },
            "walkforward_config": self._json_safe(self._walkforward_config()),
            "walkforward": {
                "train_size": self.train_size,
                "tune_size": self.tune_size,
                "test_size": self.test_size,
                "step_size": self.step_size,
                "mode": self.mode,
                "refit_on_train_plus_tune": self.refit_on_train_plus_tune,
                "pca_enabled": self.pca_enabled,
                "pca_n_components": self.pca_n_components,
                "max_trials": self.max_trials,
                "model_params": self.model_params,
                "tuning_grid": self.tuning_grid,
            },
            "threshold": {
                "fees": self.fees,
                "target_vol": self.target_vol,
                "leverage_cap": self.leverage_cap,
                "default_threshold_quantile": self.threshold_quantile,
                "threshold_quantile_min": self.threshold_quantile_min,
                "threshold_quantile_max": self.threshold_quantile_max,
                "threshold_quantile_step": self.threshold_quantile_step,
                "evaluation_start_split": self.evaluation_start_split,
                "default_vol_lookback": self.vol_lookback,
                "vol_lookback_min": self.vol_lookback_min,
                "vol_lookback_max": self.vol_lookback_max,
                "vol_lookback_step": self.vol_lookback_step,
            },
            "overall_oos_metrics": self.overall_oos_metrics,
            "walkforward_metrics": self.walkforward_metrics,
            "best_params_by_split": self.best_params_by_split,
            "parameter_schedule": self.parameter_schedule,
            "parameter_sweep": self.parameter_sweep,
            "split_performance": self.split_performance,
            "feature_ablation": self.feature_ablation,
            "diagnostics_report": self.diagnostics_report,
            "summary": self.summary,
        }
        path = self.output_dir / "xgb_run_diagnostics.json"
        path.write_text(json.dumps(self._json_safe(payload), ensure_ascii=False, indent=2))
        self.diagnostics_path = path
        print(f"Saved XGB diagnostics to {path}.")
        return path

    def run_threshold(
        self,
        *,
        fees=None,
        target_vol=None,
        vol_lookback=None,
        vol_lookback_min=None,
        vol_lookback_max=None,
        vol_lookback_step=None,
        leverage_cap=None,
        threshold_quantile=None,
        threshold_quantile_min=None,
        threshold_quantile_max=None,
        threshold_quantile_step=None,
        evaluation_start_split=None,
        init_amount=None,
    ):
        self._set_backtest_params(
            fees=fees,
            target_vol=target_vol,
            vol_lookback=vol_lookback,
            vol_lookback_min=vol_lookback_min,
            vol_lookback_max=vol_lookback_max,
            vol_lookback_step=vol_lookback_step,
            leverage_cap=leverage_cap,
            threshold_quantile=threshold_quantile,
            threshold_quantile_min=threshold_quantile_min,
            threshold_quantile_max=threshold_quantile_max,
            threshold_quantile_step=threshold_quantile_step,
            evaluation_start_split=evaluation_start_split,
            init_amount=init_amount,
        )
        required = [
            "fees",
            "target_vol",
            "vol_lookback",
            "vol_lookback_min",
            "vol_lookback_max",
            "vol_lookback_step",
            "leverage_cap",
            "threshold_quantile",
            "threshold_quantile_min",
            "threshold_quantile_max",
            "threshold_quantile_step",
            "evaluation_start_split",
            "init_amount",
        ]
        missing = [name for name in required if getattr(self, name) is None]
        if missing:
            raise ValueError(
                "Pass private threshold/backtest parameters explicitly from the ignored notebook: "
                + ", ".join(missing)
            )
        if self.predictions.empty and not self._load_predictions():
            print("No cached walk-forward predictions found, running walk-forward first.")
            self.run_walkforward()
        else:
            print("Reusing cached walk-forward predictions for threshold/backtest.")

        self.data, self.summary = self._evaluate_predictions()
        close_source = (
            next(iter(self._oos_close_source.values()))
            if len(self._oos_close_source) == 1
            else self._oos_close_source
        )
        self.buy_and_hold_data, self.buy_and_hold_summary = calculate_buy_and_hold_baseline(
            close_source=close_source,
            init_amount=self.init_amount,
            target_vol=self.target_vol,
            vol_window=self.vol_lookback,
            fees=self.fees,
            evaluation_index=self._oos_evaluation_index,
            summary_meta={
                "ticker": self.ticker_label,
                "start": self._oos_evaluation_index[0],
                "end": self._oos_evaluation_index[-1],
                "benchmark": "equal_weight_rebalanced",
                "tf": self.tf,
                "fees": self.fees,
                "hour": self.hour,
                "hour_timezone": self.hour_timezone,
                "run_name": self.run_name,
            },
        )
        self.summary["B&H_yearly_factor"] = self.buy_and_hold_summary["yearly_factor"]
        self.summary["B&H_max_drawdown"] = self.buy_and_hold_summary["max_drawdown"]
        self.summary["B&H_sharpe_ratio_annualized"] = self.buy_and_hold_summary["sharpe_ratio_annualized"]
        if not self.parameter_schedule.empty:
            self.summary["selected_threshold_quantile"] = float(
                self.parameter_schedule["applied_threshold_quantile"].iloc[-1]
            )
            self.summary["selected_vol_lookback"] = int(
                self.parameter_schedule["applied_vol_lookback"].iloc[-1]
            )
        self._save_threshold_outputs()
        return self.data

    run2 = run_threshold

    def make_diagnostics(self, save=True, show=False):
        """Build compact ML-style tables and figures from the current run."""
        if self.data.empty:
            self.run_threshold()
        report = build_diagnostics(self, save=save, show=show)
        self._save_run_diagnostics()
        return report

    def latest_positions(self):
        """Return the latest paper-trading positions available from cached/live predictions."""
        if self.predictions.empty and not self._load_predictions():
            raise ValueError("No predictions available. Run run_walkforward() first.")
        self._ensure_feature_panel()
        if self.parameter_schedule.empty:
            if self.data.empty:
                self.run_threshold()
            elif self.parameter_schedule.empty:
                raise ValueError("No parameter schedule available. Run run_threshold() first.")

        prediction_frame = self.predictions.copy()
        latest_timestamp = prediction_frame["timestamp"].max()
        latest = prediction_frame.loc[prediction_frame["timestamp"] == latest_timestamp].copy()
        latest = latest.sort_values("SYMBOL").reset_index(drop=True)
        if latest.empty:
            return latest

        last_schedule = self.parameter_schedule.iloc[-1]
        threshold_quantile = float(last_schedule["applied_threshold_quantile"])
        threshold_value = float(last_schedule["applied_threshold_value"])
        threshold_source = str(last_schedule.get("threshold_reference", "parameter_schedule"))
        vol_lookback = int(last_schedule["applied_vol_lookback"])

        volatility_panel = estimate_asset_annualized_volatility(
            self.feature_panel.loc[:, ["timestamp", "SYMBOL", "simple_return"]].copy(),
            lookback_bars=vol_lookback,
            return_col="simple_return",
            output_col="asset_vol_annualized",
        )
        latest = latest.merge(
            volatility_panel[["timestamp", "SYMBOL", "asset_vol_annualized"]],
            on=["timestamp", "SYMBOL"],
            how="left",
        )
        predictions = pd.to_numeric(latest["prediction"], errors="coerce")
        latest["abs_prediction"] = predictions.abs()
        latest["is_active_signal"] = predictions.abs().ge(threshold_value) & predictions.ne(0.0)
        latest["raw_position"] = 0.0
        valid = latest["is_active_signal"] & latest["asset_vol_annualized"].notna() & latest["asset_vol_annualized"].gt(0)
        scale = (self.target_vol / latest.loc[valid, "asset_vol_annualized"]).clip(upper=self.leverage_cap)
        latest.loc[valid, "raw_position"] = np.sign(latest.loc[valid, "prediction"]) * scale
        active_count = max(int(latest["raw_position"].ne(0).sum()), 1)
        latest["portfolio_weighted_position"] = latest["raw_position"] / active_count
        latest["signal_side"] = np.where(latest["raw_position"] > 0, "long", np.where(latest["raw_position"] < 0, "short", "flat"))
        latest["available_at"] = latest["timestamp"]
        latest["applies_to"] = "next_bar"
        latest["threshold_quantile"] = threshold_quantile
        latest["threshold_value"] = threshold_value
        latest["threshold_source"] = threshold_source
        latest["vol_lookback"] = vol_lookback
        return latest[
            [
                "available_at",
                "applies_to",
                "SYMBOL",
                "prediction",
                "abs_prediction",
                "asset_vol_annualized",
                "threshold_quantile",
                "threshold_value",
                "threshold_source",
                "vol_lookback",
                "is_active_signal",
                "signal_side",
                "raw_position",
                "portfolio_weighted_position",
            ]
        ]

    def feature_groups(self):
        return {
            name: [column for column in columns if column in self.feature_columns]
            for name, columns in FEATURE_GROUPS.items()
            if any(column in self.feature_columns for column in columns)
        }

    def run_feature_ablation(self, groups=None, max_trials=None, random_state=None):
        """Run leave-one-feature-family-out walk-forward diagnostics without changing the main predictions."""
        self._ensure_feature_panel()
        base_columns = list(self.feature_columns)
        groups = groups or self.feature_groups()
        max_trials = self.max_trials if max_trials is None else max_trials
        random_state = self.random_state if random_state is None else random_state

        rows = []
        variants = {"full": base_columns}
        for group_name, group_columns in groups.items():
            excluded = set(group_columns)
            variants[f"minus_{group_name}"] = [column for column in base_columns if column not in excluded]

        split_model_frame = self._walkforward_model_frame()
        splits = make_walkforward_splits(
            split_model_frame,
            train_size=self.train_size,
            tune_size=self.tune_size,
            test_size=self.test_size,
            step_size=self.step_size,
            mode=self.mode,
        )

        ablation_start = time.perf_counter()
        variant_items = list(variants.items())

        saved_state = {
            "predictions": self.predictions.copy(),
            "parameter_schedule": self.parameter_schedule.copy(),
            "parameter_sweep": self.parameter_sweep.copy(),
            "split_performance": self.split_performance.copy(),
            "data": self.data.copy(),
            "summary": self.summary.copy(),
            "oos_close_source": dict(self._oos_close_source),
            "oos_evaluation_index": self._oos_evaluation_index.copy(),
        }

        for variant_idx, (variant_name, variant_columns) in enumerate(variant_items, start=1):
            if not variant_columns:
                continue
            variant_frame = make_xgb_model_frame(
                self.feature_panel,
                variant_columns,
                target_column="target_next_log_return",
            )
            prediction_parts = []
            tune_scores = []
            train_scores = []
            tune_r2_scores = []
            train_r2_scores = []
            tune_balance_scores = []
            train_balance_scores = []
            print(
                f"Ablation {variant_idx}/{len(variant_items)} {variant_name}: "
                f"{len(variant_columns)} features, {len(splits)} splits..."
            )
            variant_start = time.perf_counter()
            for split_idx, split in enumerate(splits, start=1):
                split_start = time.perf_counter()
                print(f"  split {split_idx}/{len(splits)}", end="\r")
                train_df = select_split_frame(variant_frame, split.train_days)
                tune_df = select_split_frame(variant_frame, split.tune_days)
                test_df = select_split_frame(variant_frame, split.test_days)
                if train_df.empty or tune_df.empty or test_df.empty:
                    continue

                scaler = fit_feature_scaler(train_df, variant_columns)
                train_scaled = transform_with_scaler(train_df, scaler, variant_columns)
                tune_scaled = transform_with_scaler(tune_df, scaler, variant_columns)
                test_scaled = transform_with_scaler(test_df, scaler, variant_columns)
                modeling_columns = variant_columns
                pca = None
                if self.pca_enabled:
                    pca, modeling_columns = fit_feature_pca(
                        train_scaled,
                        n_components=self.pca_n_components,
                        random_state=random_state,
                    )
                    train_scaled = transform_with_pca(train_scaled, pca, modeling_columns)
                    tune_scaled = transform_with_pca(tune_scaled, pca, modeling_columns)
                    test_scaled = transform_with_pca(test_scaled, pca, modeling_columns)

                X_train, y_train = to_xy(
                    pd.concat([train_scaled, train_df[["target_next_log_return"]]], axis=1),
                    modeling_columns,
                    "target_next_log_return",
                )
                X_tune, y_tune = to_xy(
                    pd.concat([tune_scaled, tune_df[["target_next_log_return"]]], axis=1),
                    modeling_columns,
                    "target_next_log_return",
                )
                search_result = tune_xgb_model(
                    X_train,
                    y_train,
                    X_tune,
                    y_tune,
                    base_params=self.model_params,
                    tuning_grid=self.tuning_grid,
                    max_trials=max_trials,
                    random_state=random_state,
                    score_tolerance=self.tuning_score_tolerance,
                    score_quantile=self.metric_filter_quantile,
                    min_side_balance=self.tuning_min_side_balance,
                    min_prediction_unique=self.tuning_min_prediction_unique,
                    min_prediction_std=self.tuning_min_prediction_std,
                )
                tune_scores.append(search_result.get("best_score"))
                train_scores.append(search_result.get("best_train_score"))
                tune_r2_scores.append(search_result.get("best_tune_oos_r2"))
                train_r2_scores.append(search_result.get("best_train_oos_r2"))
                tune_balance_scores.append(search_result.get("best_tune_filtered_side_balance"))
                train_balance_scores.append(search_result.get("best_train_filtered_side_balance"))

                refit_df = (
                    pd.concat([train_df, tune_df], ignore_index=True)
                    if self.refit_on_train_plus_tune
                    else train_df.copy()
                )
                refit_scaled = transform_with_scaler(refit_df, scaler, variant_columns)
                if self.pca_enabled:
                    refit_scaled = transform_with_pca(refit_scaled, pca, modeling_columns)
                X_refit, y_refit = to_xy(
                    pd.concat([refit_scaled, refit_df[["target_next_log_return"]]], axis=1),
                    modeling_columns,
                    "target_next_log_return",
                )
                model = fit_xgb_model(X_refit, y_refit, search_result["best_params"])
                test_output = test_df[
                    [
                        "timestamp",
                        "trade_date",
                        "SYMBOL",
                        "close",
                        "simple_return",
                        "log_return",
                        "target_next_simple_return",
                        "target_next_log_return",
                    ]
                ].copy()
                test_output["prediction"] = model.predict(test_scaled[modeling_columns])
                test_output["split_id"] = split.split_id
                prediction_parts.append(
                    test_output
                )
            print(" " * 40, end="\r")

            if not prediction_parts:
                continue
            predictions = (
                pd.concat(prediction_parts, ignore_index=True)
                .sort_values(["timestamp", "SYMBOL"])
                .reset_index(drop=True)
            )
            metric_prefix = f"filtered_q{int(round(float(self.metric_filter_quantile) * 100))}"
            metrics = prediction_metrics(
                predictions["target_next_log_return"],
                predictions["prediction"],
                metric_filter_quantile=self.metric_filter_quantile,
            )

            self.predictions = predictions
            self._suppress_schedule_prints = True
            try:
                variant_data, variant_summary = self._evaluate_predictions()
            finally:
                self._suppress_schedule_prints = False
            summary_row = variant_summary.iloc[0].to_dict() if not variant_summary.empty else {}
            selected_threshold = (
                float(self.parameter_schedule["applied_threshold_quantile"].iloc[-1])
                if not self.parameter_schedule.empty
                else np.nan
            )
            selected_lookback = (
                int(self.parameter_schedule["applied_vol_lookback"].iloc[-1])
                if not self.parameter_schedule.empty
                else np.nan
            )

            metrics.update(
                {
                    "variant": variant_name,
                    "removed_group": None if variant_name == "full" else variant_name.replace("minus_", "", 1),
                    "feature_count": len(variant_columns),
                    f"mean_best_tune_{metric_prefix}_sign_accuracy": float(pd.Series(tune_scores).mean()),
                    f"mean_best_train_{metric_prefix}_sign_accuracy": float(pd.Series(train_scores).mean()),
                    f"mean_best_tune_{metric_prefix}_side_balance": float(pd.Series(tune_balance_scores).mean()),
                    f"mean_best_train_{metric_prefix}_side_balance": float(pd.Series(train_balance_scores).mean()),
                    "mean_best_tune_oos_r2": float(pd.Series(tune_r2_scores).mean()),
                    "mean_best_train_oos_r2": float(pd.Series(train_r2_scores).mean()),
                    "yearly_factor": summary_row.get("yearly_factor"),
                    "net_sharpe": summary_row.get("sharpe_ratio_annualized"),
                    "max_drawdown": summary_row.get("max_drawdown"),
                    "turnover": summary_row.get("turnover"),
                    "total_fees": summary_row.get("total_fees"),
                    "winrate": summary_row.get("winrate"),
                    "selected_threshold_quantile": selected_threshold,
                    "selected_vol_lookback": selected_lookback,
                    "bars": len(variant_data),
                }
            )
            rows.append(metrics)
            print(
                f"Ablation {variant_idx}/{len(variant_items)} {variant_name}: "
                f"Sharpe={metrics.get('net_sharpe', np.nan):.3f}, "
                f"yearly={metrics.get('yearly_factor', np.nan):.3f}, "
                f"drawdown={metrics.get('max_drawdown', np.nan):.2%}, "
                f"time={self._format_seconds(time.perf_counter() - variant_start)} "
                f"({self._format_eta(ablation_start, variant_idx, len(variant_items))})."
            )

        self.predictions = saved_state["predictions"]
        self.parameter_schedule = saved_state["parameter_schedule"]
        self.parameter_sweep = saved_state["parameter_sweep"]
        self.split_performance = saved_state["split_performance"]
        self.data = saved_state["data"]
        self.summary = saved_state["summary"]
        self._oos_close_source = saved_state["oos_close_source"]
        self._oos_evaluation_index = saved_state["oos_evaluation_index"]

        self.feature_ablation = pd.DataFrame(rows)
        if not self.feature_ablation.empty:
            self.feature_ablation.to_csv(self.output_dir / "feature_ablation.csv", index=False)
        self._save_run_diagnostics()
        return self.feature_ablation

    def run(self, **kwargs):
        walkforward_keys = {
            "train_size",
            "tune_size",
            "test_size",
            "step_size",
            "mode",
            "refit_on_train_plus_tune",
            "model_params",
            "tuning_grid",
            "max_trials",
            "random_state",
            "pca_enabled",
            "pca_n_components",
            "tuning_min_side_balance",
            "metric_filter_quantile",
            "tuning_score_tolerance",
            "tuning_min_prediction_unique",
            "tuning_min_prediction_std",
            "split_end",
            "force",
        }
        walkforward_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in walkforward_keys}
        if not self.raw_data and self.feature_panel.empty:
            self.fetch_data(force=False, save=True)
        self.run_walkforward(**walkforward_kwargs)
        return self.run_threshold(**kwargs)

    def plot_wealth(self):
        if self.data.empty:
            self.run_threshold()
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
            title=f"{self.ticker_label} XGB Strategy Wealth",
            log_scale=True,
            benchmark_wealth=benchmark_wealth,
            benchmark_label="B&H",
        )
