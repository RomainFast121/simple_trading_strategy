from dataclasses import dataclass
from decimal import Decimal
from datetime import timedelta, timezone
from itertools import product
import json
import random
from statistics import NormalDist
import re

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import ccxt
except ImportError:  # pragma: no cover - handled at runtime when crypto data is requested
    ccxt = None


BINANCE_TIMEFRAMES = {
    "1s",
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1M",
}


TIMEFRAME_ALIASES = {
    "1min": "1m",
    "3min": "3m",
    "5min": "5m",
    "15min": "15m",
    "30min": "30m",
    "60m": "1h",
    "60min": "1h",
    "1hr": "1h",
    "2hr": "2h",
    "4hr": "4h",
    "6hr": "6h",
    "8hr": "8h",
    "12hr": "12h",
    "1day": "1d",
    "1wk": "1w",
    "1mo": "1M",
}


# Parse a timezone label such as UTC, UTC+1, UTC-05:00, or an IANA zone name.
def parse_timezone(timezone_label):
    if timezone_label is None:
        return timezone.utc

    if not isinstance(timezone_label, str):
        raise ValueError("timezone_label must be a string such as 'UTC' or 'UTC+1'.")

    label = timezone_label.strip()
    if label.upper() == "UTC":
        return timezone.utc

    match = re.fullmatch(r"UTC([+-])(\d{1,2})(?::?(\d{2}))?", label.upper())
    if match:
        sign = 1 if match.group(1) == "+" else -1
        hours = int(match.group(2))
        minutes = int(match.group(3) or 0)
        offset = timedelta(hours=hours, minutes=minutes) * sign
        return timezone(offset)

    return label


# Normalize a string/list input into a clean list of symbols.
def normalize_symbol_input(symbols):
    if symbols is None:
        return []

    if isinstance(symbols, str):
        symbols = [symbols]

    normalized = []
    for symbol in symbols:
        if symbol is None:
            continue
        symbol = str(symbol).strip()
        if symbol:
            normalized.append(symbol)
    return normalized


# Select one hourly bar per local day, using the bar that starts at the chosen hour.
def build_daily_snapshot_from_hourly(data, hour=0, hour_timezone="UTC"):
    if data.empty:
        raise ValueError("No hourly data available to build daily snapshots.")

    parsed_timezone = parse_timezone(hour_timezone)
    intraday = data.copy()
    intraday.index = pd.to_datetime(intraday.index)

    if intraday.index.tz is None:
        intraday.index = intraday.index.tz_localize("UTC")

    intraday.index = intraday.index.tz_convert(parsed_timezone)
    selected = intraday[intraday.index.hour == hour].copy()

    if selected.empty:
        raise ValueError(
            f"No hourly bars found at hour {hour} in timezone {hour_timezone}."
        )

    selected["local_date"] = selected.index.normalize()
    selected = selected.groupby("local_date", sort=True, group_keys=False).tail(1)
    selected = selected.drop(columns="local_date")
    selected.index.name = None
    return selected.sort_index()


# Return the exchange timeframe string used for the requested logical interval.
def resolve_fetch_interval(interval, hour):
    resolved = "1h" if interval == "1d" and hour is not None else str(interval).strip()
    return TIMEFRAME_ALIASES.get(resolved, resolved)


def validate_binance_timeframe(timeframe):
    if timeframe not in BINANCE_TIMEFRAMES:
        supported = ", ".join(sorted(BINANCE_TIMEFRAMES))
        raise ValueError(
            f"Binance does not support timeframe '{timeframe}'. "
            f"Use one of: {supported}. For 15-minute bars, set tf='15m'."
        )


# Treat date-like end inputs as inclusive so requesting today's date also includes today's bars.
def resolve_fetch_end(end):
    timestamp = pd.Timestamp(end)

    is_date_only_string = isinstance(end, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", end.strip()) is not None
    is_midnight_timestamp = (
        timestamp.hour == 0
        and timestamp.minute == 0
        and timestamp.second == 0
        and timestamp.microsecond == 0
        and timestamp.nanosecond == 0
    )

    if is_date_only_string or is_midnight_timestamp:
        timestamp = timestamp + pd.Timedelta(days=1)

    return timestamp


# Convert a raw OHLCV dataframe to the project-normalized column format.
def normalize_ohlcv_frame(data):
    frame = data.copy()
    frame.index = pd.to_datetime(frame.index)

    if isinstance(frame.columns, pd.MultiIndex):
        if frame.columns.nlevels == 2 and len(frame.columns.get_level_values(1).unique()) == 1:
            frame.columns = frame.columns.get_level_values(0)
            frame = frame.rename(columns=str.lower)
            return frame.sort_index()

        frame.columns = pd.MultiIndex.from_tuples(
            [(str(level_0).lower(), str(level_1)) for level_0, level_1 in frame.columns]
        )
    else:
        frame = frame.rename(columns=str.lower)

    return frame.sort_index()


# Convert a fetched dataframe to a native daily frame or an hourly-snapshot daily frame.
def finalize_market_frame(data, interval, hour, hour_timezone):
    frame = normalize_ohlcv_frame(data)
    if interval == "1d" and hour is not None:
        return build_daily_snapshot_from_hourly(frame, hour=hour, hour_timezone=hour_timezone)

    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_convert("UTC").tz_localize(None)

    return frame


# Fetch one Yahoo Finance symbol.
def fetch_yahoo_symbol(
    symbol,
    start,
    end,
    interval="1d",
    auto_adjust=True,
    progress=False,
    hour=None,
    hour_timezone="UTC",
):
    yahoo_interval = resolve_fetch_interval(interval, hour)
    resolved_end = resolve_fetch_end(end)
    data = yf.download(
        symbol,
        start=start,
        end=resolved_end,
        interval=yahoo_interval,
        auto_adjust=auto_adjust,
        progress=progress,
    )

    if data.empty:
        raise ValueError(f"No Yahoo Finance data returned for {symbol}.")

    return finalize_market_frame(data, interval=interval, hour=hour, hour_timezone=hour_timezone)


# Fetch one Binance symbol using chunked OHLCV calls.
def fetch_binance_symbol(
    symbol,
    start,
    end,
    interval="1d",
    hour=None,
    hour_timezone="UTC",
):
    if ccxt is None:
        raise ImportError("ccxt is required to fetch crypto data from Binance.")

    exchange = ccxt.binance()
    fetch_interval = resolve_fetch_interval(interval, hour)
    validate_binance_timeframe(fetch_interval)
    timeframe_ms = exchange.parse_timeframe(fetch_interval) * 1000

    start_ts = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    resolved_end = resolve_fetch_end(end)
    end_ts = int(pd.Timestamp(resolved_end, tz="UTC").timestamp() * 1000)

    rows = []
    since = start_ts
    chunk = 1000

    while since < end_ts:
        batch = exchange.fetch_ohlcv(symbol, timeframe=fetch_interval, since=since, limit=chunk)
        if not batch:
            break

        rows.extend(batch)
        next_since = batch[-1][0] + timeframe_ms
        if next_since <= since:
            break
        since = next_since

    if not rows:
        raise ValueError(f"No Binance data returned for {symbol}.")

    frame = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
    frame = frame[
        (frame.index >= pd.Timestamp(start, tz="UTC"))
        & (frame.index < pd.Timestamp(resolved_end, tz="UTC"))
    ]

    if frame.empty:
        raise ValueError(f"No Binance data returned for {symbol} in the requested period.")

    return finalize_market_frame(frame, interval=interval, hour=hour, hour_timezone=hour_timezone)


# Fetch Yahoo and Binance symbols and return one native dataframe per sleeve.
def fetch_data(
    ticker=None,
    crypto=None,
    start=None,
    end=None,
    interval="1d",
    auto_adjust=True,
    progress=False,
    hour=None,
    hour_timezone="UTC",
):
    yahoo_symbols = normalize_symbol_input(ticker)
    crypto_symbols = normalize_symbol_input(crypto)

    if not yahoo_symbols and not crypto_symbols:
        raise ValueError("select at least one ticker")

    data_map = {}
    for symbol in yahoo_symbols:
        data_map[symbol] = fetch_yahoo_symbol(
            symbol=symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            progress=progress,
            hour=hour,
            hour_timezone=hour_timezone,
        )

    for symbol in crypto_symbols:
        data_map[symbol] = fetch_binance_symbol(
            symbol=symbol,
            start=start,
            end=end,
            interval=interval,
            hour=hour,
            hour_timezone=hour_timezone,
        )

    return data_map


# Convert a close price series into log returns.
def log_return(close):
    close = pd.Series(close, copy=False).astype(float)
    return np.log(close / close.shift(1))


# Backward-compatible alias with a more explicit name.
def calculate_log_return(close):
    return log_return(close)


# Estimate how many observations there are in one year based on timestamp spacing.
def estimate_periods_per_year(index):
    if len(index) < 2:
        return np.nan

    deltas = index.to_series().diff().dropna().dt.total_seconds()
    if deltas.empty:
        return np.nan

    median_seconds = deltas.median()
    if median_seconds <= 0:
        return np.nan

    return (365.25 * 24 * 60 * 60) / median_seconds


# Compute annualized rolling volatility on a time-based window.
def rolling_annualized_vol(log_returns, window, min_periods):
    periods_per_year = estimate_periods_per_year(log_returns.index)
    if not np.isfinite(periods_per_year):
        raise ValueError("Not enough data points to estimate annualization.")

    rolling_std = log_returns.rolling(window=window, min_periods=min_periods).std()
    return rolling_std * np.sqrt(periods_per_year)


# Extract one close series per sleeve from a Series, DataFrame, or mapping source.
def extract_close_map(source, symbols=None):
    fallback_symbols = normalize_symbol_input(symbols)

    if isinstance(source, pd.Series):
        symbol = fallback_symbols[0] if fallback_symbols else "asset"
        return {symbol: source.astype(float)}

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
            symbol = fallback_symbols[0] if fallback_symbols else "asset"
            return {symbol: source["close"].astype(float)}
        return {symbol: source[symbol].astype(float) for symbol in source.columns}

    raise ValueError("Close source must be a Series, DataFrame, or mapping.")


# Build a stacked close panel with one row per timestamp and symbol from fetched close series.
def build_close_panel(close_source, symbols=None):
    close_map = extract_close_map(close_source, symbols=symbols)
    panel_frames = []

    for symbol, close_series in close_map.items():
        series = pd.Series(close_series, copy=False).astype(float).dropna()
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(series.index),
                "SYMBOL": str(symbol),
                "close": series.to_numpy(),
            }
        )
        panel_frames.append(frame)

    if not panel_frames:
        raise ValueError("No close data available to build the panel.")

    panel = pd.concat(panel_frames, ignore_index=True)
    panel = panel.sort_values(["SYMBOL", "timestamp"]).reset_index(drop=True)
    panel["trade_date"] = panel["timestamp"].dt.normalize()

    bars_per_day = panel.groupby("trade_date")["timestamp"].nunique()
    if not bars_per_day.empty and bars_per_day.max() > 1:
        bar_lookup = (
            panel[["trade_date", "timestamp"]]
            .drop_duplicates()
            .sort_values(["trade_date", "timestamp"])
            .reset_index(drop=True)
        )
        bar_lookup["bar_number"] = bar_lookup.groupby("trade_date").cumcount() + 1
        panel = panel.merge(bar_lookup, on=["trade_date", "timestamp"], how="left", validate="many_to_one")

    return panel.sort_values(["timestamp", "SYMBOL"]).reset_index(drop=True)


# Add same-symbol close-to-close returns and next-bar targets from the fetched close data.
def add_close_return_targets(panel, drop_incomplete=True):
    frame = panel.sort_values(["SYMBOL", "timestamp"]).copy()
    grouped = frame.groupby("SYMBOL", sort=False)

    previous_close = grouped["close"].shift(1)
    next_close = grouped["close"].shift(-1)

    frame["simple_return"] = frame["close"] / previous_close - 1.0
    frame["log_return"] = np.log(frame["close"] / previous_close)
    frame["target_next_simple_return"] = next_close / frame["close"] - 1.0
    frame["target_next_log_return"] = np.log(next_close / frame["close"])

    invalid_rows = frame[["simple_return", "log_return"]].isna().any(axis=1)
    if drop_incomplete:
        invalid_rows = invalid_rows | frame[
            ["target_next_simple_return", "target_next_log_return"]
        ].isna().any(axis=1)
    return frame.loc[~invalid_rows].reset_index(drop=True)


def _lag_by_symbol(panel, column, lag):
    return panel.groupby("SYMBOL", sort=False)[column].shift(lag)


def _rolling_by_symbol(panel, column, window, operation):
    grouped = panel.groupby("SYMBOL", sort=False)[column]
    shifted = grouped.transform(lambda series: series.shift(1))

    if operation == "sum":
        return shifted.groupby(panel["SYMBOL"]).transform(
            lambda series: series.rolling(window, min_periods=window).sum()
        )
    if operation == "std":
        return shifted.groupby(panel["SYMBOL"]).transform(
            lambda series: series.rolling(window, min_periods=window).std()
        )
    if operation == "mean":
        return shifted.groupby(panel["SYMBOL"]).transform(
            lambda series: series.rolling(window, min_periods=window).mean()
        )
    if operation == "mean_abs":
        return shifted.abs().groupby(panel["SYMBOL"]).transform(
            lambda series: series.rolling(window, min_periods=window).mean()
        )

    raise ValueError(f"Unsupported rolling operation: {operation}")


# Build close-based XGB features using only past information.
def build_xgb_features(
    panel,
    lag_windows=(1, 2, 3, 6, 12, 24),
    roll_windows=(3, 6, 12),
    vol_windows=(6, 12, 24),
):
    frame = panel.sort_values(["SYMBOL", "timestamp"]).copy()

    for lag in lag_windows:
        frame[f"lag_log_return_{lag}"] = _lag_by_symbol(frame, "log_return", lag)

    for window in roll_windows:
        frame[f"cumret_{window}"] = _rolling_by_symbol(frame, "log_return", window, "sum")

    for window in vol_windows:
        frame[f"rv_{window}"] = _rolling_by_symbol(frame, "log_return", window, "std")
        frame[f"mean_abs_return_{window}"] = _rolling_by_symbol(frame, "log_return", window, "mean_abs")

    rolling_mean_12 = _rolling_by_symbol(frame, "log_return", 12, "mean")
    rolling_std_12 = _rolling_by_symbol(frame, "log_return", 12, "std")
    rolling_mean_24 = _rolling_by_symbol(frame, "log_return", 24, "mean")
    rolling_std_24 = _rolling_by_symbol(frame, "log_return", 24, "std")

    frame["zscore_return_12"] = (
        frame["lag_log_return_1"] - rolling_mean_12
    ) / rolling_std_12.replace(0.0, np.nan)
    frame["zscore_return_24"] = (
        frame["lag_log_return_1"] - rolling_mean_24
    ) / rolling_std_24.replace(0.0, np.nan)

    if "bar_number" in frame.columns and frame["bar_number"].nunique() > 1:
        max_bar = frame.groupby("trade_date")["bar_number"].transform("max").replace(0, np.nan)
        angle = 2.0 * np.pi * (frame["bar_number"] - 1) / max_bar
        frame["bar_of_day_sin"] = np.sin(angle)
        frame["bar_of_day_cos"] = np.cos(angle)

    market = (
        frame.groupby("timestamp", as_index=False)
        .agg(
            eq_market_return=("log_return", "mean"),
            cross_sectional_dispersion=("log_return", "std"),
        )
        .sort_values("timestamp")
    )
    market["lag_eq_market_return"] = market["eq_market_return"].shift(1)
    market["lag_cross_sectional_dispersion"] = market["cross_sectional_dispersion"].shift(1)

    frame = frame.merge(
        market[["timestamp", "lag_eq_market_return", "lag_cross_sectional_dispersion"]],
        on="timestamp",
        how="left",
        validate="many_to_one",
    )
    frame["stock_minus_market_lag"] = frame["lag_log_return_1"] - frame["lag_eq_market_return"]
    frame["lag_rank_pct"] = frame.groupby("timestamp")["lag_log_return_1"].rank(pct=True)

    return frame.sort_values(["timestamp", "SYMBOL"]).reset_index(drop=True)


# Infer the model feature columns from the close-based XGB feature panel.
def infer_xgb_feature_columns(panel):
    excluded = {
        "timestamp",
        "trade_date",
        "SYMBOL",
        "close",
        "simple_return",
        "log_return",
        "target_next_simple_return",
        "target_next_log_return",
        "bar_number",
    }
    return [column for column in panel.columns if column not in excluded]


# Keep only rows with complete features and target for model fitting.
def make_xgb_model_frame(panel, feature_columns, target_column="target_next_log_return"):
    required = feature_columns + [
        target_column,
        "timestamp",
        "trade_date",
        "SYMBOL",
        "close",
        "simple_return",
        "log_return",
        "target_next_simple_return",
    ]
    available = [column for column in required if column in panel.columns]
    frame = panel[available].copy()
    frame = frame.dropna(subset=[target_column] + feature_columns)
    return frame.reset_index(drop=True)


# Split a modeling panel into feature matrix and target vector.
def to_xy(frame, feature_columns, target_column):
    return frame[feature_columns].copy(), frame[target_column].copy()


@dataclass
class WalkForwardSplit:
    split_id: int
    train_days: list[pd.Timestamp]
    tune_days: list[pd.Timestamp]
    test_days: list[pd.Timestamp]


def _resolve_window_size(value, total):
    if isinstance(value, float) and 0 < value < 1:
        return max(1, int(round(total * value)))
    return max(1, int(round(value)))


# Build expanding or rolling walk-forward splits on unique trade dates.
def make_walkforward_splits(
    panel,
    train_size,
    tune_size,
    test_size,
    step_size,
    mode,
):
    unique_days = pd.Index(sorted(pd.to_datetime(panel["trade_date"]).drop_duplicates()))
    total_days = len(unique_days)

    if total_days < 3:
        raise ValueError("Not enough trade dates to create walk-forward splits.")

    train_count = _resolve_window_size(train_size, total_days)
    tune_count = _resolve_window_size(tune_size, total_days)
    test_count = _resolve_window_size(test_size, total_days)
    step_count = _resolve_window_size(step_size, total_days)

    if train_count + tune_count + test_count > total_days:
        raise ValueError("Walk-forward windows are larger than the available sample.")

    mode = str(mode).lower()
    if mode not in {"expanding", "rolling"}:
        raise ValueError("mode must be 'expanding' or 'rolling'.")

    splits = []
    split_id = 1
    train_start = 0
    train_end = train_count

    while True:
        tune_start = train_end
        tune_end = tune_start + tune_count
        test_start = tune_end
        test_end = test_start + test_count

        if test_end > total_days:
            break

        splits.append(
            WalkForwardSplit(
                split_id=split_id,
                train_days=unique_days[train_start:train_end].tolist(),
                tune_days=unique_days[tune_start:tune_end].tolist(),
                test_days=unique_days[test_start:test_end].tolist(),
            )
        )
        split_id += 1

        if mode == "expanding":
            train_end += step_count
        else:
            train_start += step_count
            train_end = train_start + train_count

        if train_end + tune_count + test_count > total_days:
            break

    if not splits:
        raise ValueError("No walk-forward splits could be created with the chosen parameters.")

    # Keep every valid modeled day in the out-of-sample evaluation. Fixed-size
    # windows can otherwise strand a short tail just before the live no-target row.
    last_test_day = splits[-1].test_days[-1]
    last_test_position = unique_days.get_loc(last_test_day)
    if last_test_position + 1 < total_days:
        tail_days = unique_days[last_test_position + 1 :].tolist()
        last_split = splits[-1]
        splits[-1] = WalkForwardSplit(
            split_id=last_split.split_id,
            train_days=last_split.train_days,
            tune_days=last_split.tune_days,
            test_days=last_split.test_days + tail_days,
        )

    return splits


# Select rows belonging to a set of trade dates.
def select_split_frame(panel, split_days):
    split_days = pd.to_datetime(pd.Index(split_days))
    return panel.loc[pd.to_datetime(panel["trade_date"]).isin(split_days)].copy()


# Fit a standard scaler on the train features only.
def fit_feature_scaler(train_frame, feature_columns):
    try:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(train_frame[feature_columns])
        return scaler
    except ModuleNotFoundError:
        class SimpleStandardScaler:
            def fit(self, frame):
                self.center_ = frame.mean()
                self.scale_ = frame.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
                return self

            def transform(self, frame):
                transformed = (frame - self.center_) / self.scale_
                return transformed.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()

        return SimpleStandardScaler().fit(train_frame[feature_columns])


# Apply a fitted scaler and return a DataFrame with the same index.
def transform_with_scaler(frame, scaler, feature_columns):
    transformed = scaler.transform(frame[feature_columns])
    return pd.DataFrame(transformed, index=frame.index, columns=feature_columns)


# Fit PCA on train features only and expose stable component column names.
def fit_feature_pca(train_features, n_components=0.95, random_state=42):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(train_features)
    output_columns = [f"pc_{column_number + 1}" for column_number in range(pca.n_components_)]
    return pca, output_columns


# Apply a fitted PCA transform and return a DataFrame with stable component names.
def transform_with_pca(frame, pca, output_columns):
    transformed = pca.transform(frame)
    return pd.DataFrame(transformed, index=frame.index, columns=output_columns)


# Out-of-sample R² kept as a diagnostic for XGB predictions.
def oos_r2(y_true, y_pred):
    observed = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    denominator = np.square(observed).sum()
    if denominator <= 0:
        return np.nan
    return 1.0 - (np.square(observed - predicted).sum() / denominator)


# Lightweight prediction diagnostics for walk-forward reporting.
def prediction_metrics(y_true, y_pred, metric_filter_quantile):
    observed = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    errors = observed - predicted
    rmse = float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else np.nan
    mae = float(np.mean(np.abs(errors))) if len(errors) else np.nan
    nonzero_observed = observed != 0.0
    sign_accuracy = (
        float(np.mean(np.sign(observed[nonzero_observed]) == np.sign(predicted[nonzero_observed])))
        if nonzero_observed.any()
        else np.nan
    )
    finite = np.isfinite(observed) & np.isfinite(predicted)
    rank_ic = np.nan
    if finite.sum() >= 2:
        observed_rank_input = observed[finite]
        predicted_rank_input = predicted[finite]
        if np.unique(observed_rank_input).size > 1 and np.unique(predicted_rank_input).size > 1:
            rank_ic = float(
                pd.Series(observed_rank_input).corr(
                    pd.Series(predicted_rank_input),
                    method="spearman",
                )
            )
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "oos_r2": oos_r2(observed, predicted),
        "sign_accuracy": sign_accuracy,
        "rank_ic": rank_ic,
    }
    prefix = f"filtered_q{int(round(float(metric_filter_quantile) * 100))}"
    metrics.update(filtered_signal_metrics(observed, predicted, quantile=metric_filter_quantile, prefix=prefix))
    return metrics


def filtered_signal_metrics(y_true, y_pred, quantile, prefix=None):
    observed = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[finite]
    predicted = predicted[finite]
    label = prefix or f"filtered_q{int(round(float(quantile) * 100))}"
    output = {
        f"{label}_sign_accuracy": np.nan,
        f"{label}_active_share": np.nan,
        f"{label}_count": 0,
        f"{label}_side_balance": np.nan,
        f"{label}_long_mean_return": np.nan,
        f"{label}_short_mean_return": np.nan,
        f"{label}_hit_rate_edge": np.nan,
    }
    if len(observed) == 0:
        return output

    keep_fraction = 1.0 - float(np.clip(quantile, 0.0, 1.0))
    keep_n = max(1, int(np.ceil(len(predicted) * keep_fraction - 1e-12)))
    order = np.argsort(-np.abs(predicted), kind="mergesort")
    active = np.zeros(len(predicted), dtype=bool)
    active[order[:keep_n]] = True
    nonzero = observed != 0.0
    scored = active & nonzero
    output[f"{label}_active_share"] = float(active.mean())
    output[f"{label}_count"] = int(scored.sum())
    if scored.any():
        sign_accuracy = float(np.mean(np.sign(observed[scored]) == np.sign(predicted[scored])))
        output[f"{label}_sign_accuracy"] = sign_accuracy
        output[f"{label}_hit_rate_edge"] = sign_accuracy - 0.5

    long_active = active & (predicted > 0.0)
    short_active = active & (predicted < 0.0)
    if active.any():
        long_share = float(long_active.sum() / active.sum())
        output[f"{label}_side_balance"] = min(long_share, 1.0 - long_share) / 0.5
    if long_active.any():
        output[f"{label}_long_mean_return"] = float(observed[long_active].mean())
    if short_active.any():
        output[f"{label}_short_mean_return"] = float(observed[short_active].mean())
    return output


def filtered_sign_accuracy_score(y_true, y_pred, quantile):
    return filtered_signal_metrics(y_true, y_pred, quantile=quantile)[
        f"filtered_q{int(round(float(quantile) * 100))}_sign_accuracy"
    ]


# Fit one XGB regressor from the provided parameters.
def fit_xgb_model(X_train, y_train, params):
    try:
        from xgboost import XGBRegressor
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("xgboost is required. Install requirements.txt first.") from exc

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model


def _xgb_complexity_rank(params):
    """Lower is simpler; used only to break near-ties on the tune split."""
    params = params or {}
    required = [
        "max_depth",
        "n_estimators",
        "learning_rate",
        "min_child_weight",
        "reg_lambda",
        "reg_alpha",
        "gamma",
        "subsample",
        "colsample_bytree",
    ]
    if not all(key in params for key in required):
        return (json.dumps(params, sort_keys=True, default=str),)
    return (
        float(params["max_depth"]),
        float(params["n_estimators"]),
        float(params["learning_rate"]),
        -float(params["min_child_weight"]),
        -float(params["reg_lambda"]),
        -float(params["reg_alpha"]),
        -float(params["gamma"]),
        float(params["subsample"]),
        float(params["colsample_bytree"]),
    )


# Tune XGB with a sampled grid search on the walk-forward tune split.
def tune_xgb_model(
    X_train,
    y_train,
    X_tune,
    y_tune,
    base_params,
    tuning_grid=None,
    max_trials=None,
    random_state=None,
    score_tolerance=None,
    score_quantile=None,
    min_side_balance=None,
    min_prediction_unique=None,
    min_prediction_std=None,
):
    required = {
        "max_trials": max_trials,
        "random_state": random_state,
        "score_tolerance": score_tolerance,
        "score_quantile": score_quantile,
        "min_side_balance": min_side_balance,
        "min_prediction_unique": min_prediction_unique,
        "min_prediction_std": min_prediction_std,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError("Pass private XGB tuning parameters explicitly: " + ", ".join(missing))

    base_params = dict(base_params or {})
    tuning_grid = tuning_grid or {}

    if not tuning_grid:
        model = fit_xgb_model(X_train, y_train, base_params)
        tune_pred = model.predict(X_tune)
        train_pred = model.predict(X_train)
        score = filtered_sign_accuracy_score(y_tune, tune_pred, quantile=score_quantile)
        train_score = filtered_sign_accuracy_score(y_train, train_pred, quantile=score_quantile)
        tune_metrics = filtered_signal_metrics(y_tune, tune_pred, quantile=score_quantile)
        train_metrics = filtered_signal_metrics(y_train, train_pred, quantile=score_quantile)
        score_name = f"filtered_q{int(round(float(score_quantile) * 100))}_sign_accuracy"
        return {
            "best_params": base_params,
            "best_score": score,
            "best_score_name": score_name,
            "best_train_score": train_score,
            "best_overfit_gap": train_score - score,
            "best_tune_oos_r2": oos_r2(y_tune, tune_pred),
            "best_train_oos_r2": oos_r2(y_train, train_pred),
            "best_tune_filtered_side_balance": tune_metrics.get(
                f"filtered_q{int(round(float(score_quantile) * 100))}_side_balance"
            ),
            "best_train_filtered_side_balance": train_metrics.get(
                f"filtered_q{int(round(float(score_quantile) * 100))}_side_balance"
            ),
            "min_side_balance": float(np.clip(min_side_balance, 0.0, 1.0)),
            "best_tune_prediction_unique": int(pd.Series(tune_pred).nunique()),
            "best_tune_prediction_std": float(pd.Series(tune_pred).std(ddof=0)),
        }

    keys = list(tuning_grid.keys())
    all_combinations = list(product(*(tuning_grid[key] for key in keys)))
    if max_trials < len(all_combinations):
        rng = random.Random(int(random_state))
        sampled_indices = sorted(rng.sample(range(len(all_combinations)), int(max_trials)))
        combinations = [all_combinations[index] for index in sampled_indices]
    else:
        combinations = all_combinations

    min_side_balance = float(np.clip(min_side_balance, 0.0, 1.0))
    best_params = base_params.copy()
    best_score = -np.inf
    best_rank = (2, float("inf"), float("inf"), float("inf"))
    best_train_score = np.nan
    best_tune_oos_r2 = np.nan
    best_train_oos_r2 = np.nan
    best_tune_side_balance = np.nan
    best_train_side_balance = np.nan
    best_tune_prediction_unique = 0
    best_tune_prediction_std = np.nan
    best_complexity = _xgb_complexity_rank(best_params)

    for values in combinations:
        params = base_params.copy()
        params.update(dict(zip(keys, values)))
        model = fit_xgb_model(X_train, y_train, params)
        tune_pred = model.predict(X_tune)
        tune_pred_series = pd.Series(tune_pred)
        prediction_unique = int(tune_pred_series.nunique())
        prediction_std = float(tune_pred_series.std(ddof=0))
        degenerate_prediction = (
            prediction_unique < int(min_prediction_unique)
            or not np.isfinite(prediction_std)
            or prediction_std < float(min_prediction_std)
        )
        score = filtered_sign_accuracy_score(y_tune, tune_pred, quantile=score_quantile)
        if not np.isfinite(score):
            score = -np.inf
        tune_metrics = filtered_signal_metrics(y_tune, tune_pred, quantile=score_quantile)
        metric_prefix = f"filtered_q{int(round(float(score_quantile) * 100))}"
        side_balance = tune_metrics.get(f"{metric_prefix}_side_balance")
        side_balance = float(side_balance) if np.isfinite(side_balance) else -np.inf
        side_deficit = max(0.0, min_side_balance - side_balance)
        candidate_rank = (
            2 if degenerate_prediction else (0 if side_deficit <= score_tolerance else 1),
            side_deficit,
            -score,
            _xgb_complexity_rank(params),
        )
        complexity = _xgb_complexity_rank(params)
        if candidate_rank < best_rank:
            best_rank = candidate_rank
            best_score = score
            train_pred = model.predict(X_train)
            train_metrics = filtered_signal_metrics(y_train, train_pred, quantile=score_quantile)
            best_train_score = filtered_sign_accuracy_score(y_train, train_pred, quantile=score_quantile)
            best_tune_oos_r2 = oos_r2(y_tune, tune_pred)
            best_train_oos_r2 = oos_r2(y_train, train_pred)
            best_tune_side_balance = side_balance
            best_train_side_balance = train_metrics.get(f"{metric_prefix}_side_balance")
            best_tune_prediction_unique = prediction_unique
            best_tune_prediction_std = prediction_std
            best_complexity = complexity
            best_params = params

    score_name = f"filtered_q{int(round(float(score_quantile) * 100))}_sign_accuracy"
    return {
        "best_params": best_params,
        "best_score": best_score,
        "best_score_name": score_name,
        "best_train_score": best_train_score,
        "best_overfit_gap": best_train_score - best_score,
        "best_tune_oos_r2": best_tune_oos_r2,
        "best_train_oos_r2": best_train_oos_r2,
        "best_tune_filtered_side_balance": best_tune_side_balance,
        "best_train_filtered_side_balance": best_train_side_balance,
        "min_side_balance": min_side_balance,
        "best_tune_prediction_unique": best_tune_prediction_unique,
        "best_tune_prediction_std": best_tune_prediction_std,
        "min_prediction_unique": int(min_prediction_unique),
        "min_prediction_std": float(min_prediction_std),
    }


# Estimate per-asset annualized volatility from rolling close-to-close simple returns.
def estimate_asset_annualized_volatility(
    panel,
    lookback_bars,
    return_col="simple_return",
    output_col="asset_vol_annualized",
):
    if lookback_bars <= 0:
        raise ValueError("lookback_bars must be positive")
    if "timestamp" not in panel.columns or "SYMBOL" not in panel.columns:
        raise ValueError("panel must contain timestamp and SYMBOL columns")
    if return_col not in panel.columns:
        raise ValueError(f"panel must contain {return_col}")

    frame = panel.sort_values(["SYMBOL", "timestamp"]).copy()

    def _per_symbol_volatility(series):
        valid = pd.Series(series, copy=False).astype(float)
        indexed = pd.Series(valid.to_numpy(), index=frame.loc[valid.index, "timestamp"])
        annualized = rolling_annualized_vol(indexed, window=lookback_bars, min_periods=lookback_bars)
        annualized.index = valid.index
        return annualized

    frame[output_col] = frame.groupby("SYMBOL", sort=False)[return_col].transform(_per_symbol_volatility)
    return frame


# Build a stable inclusive search grid from min / max / step values.
def build_search_grid(min_value, max_value, step, cast=float):
    if step <= 0:
        raise ValueError("step must be positive")
    if max_value < min_value:
        raise ValueError("max_value must be greater than or equal to min_value")

    current = Decimal(str(min_value))
    maximum = Decimal(str(max_value))
    increment = Decimal(str(step))
    values = []

    while current <= maximum + Decimal("1e-12"):
        values.append(cast(float(current)))
        current += increment

    return values


# Build cross-sectional positions from XGB predictions and rolling asset volatility estimates.
def apply_xgb_cross_sectional_positions(
    prediction_frame,
    target_vol,
    leverage_cap,
    vol_col="asset_vol_annualized",
    vol_col_per_row=None,
    prediction_col="prediction",
    threshold_value=0.0,
    threshold_value_col=None,
    active_col=None,
):
    if leverage_cap <= 0.0:
        raise ValueError("leverage_cap must be positive")
    if target_vol <= 0.0:
        raise ValueError("target_vol must be positive")
    if vol_col_per_row is None and vol_col not in prediction_frame.columns:
        raise ValueError(f"prediction_frame must contain {vol_col}")

    panel = prediction_frame.sort_values(["timestamp", "SYMBOL"]).copy()
    panel["position"] = 0.0

    positioned_groups = []
    for _, group in panel.groupby("timestamp", sort=False):
        local = group.copy()
        resolved_threshold = float(threshold_value)
        if threshold_value_col is not None and threshold_value_col in local.columns and not local.empty:
            resolved_threshold = float(local[threshold_value_col].iloc[0])
        if active_col is not None and active_col in local.columns:
            local = local.loc[local[active_col].astype(bool)].copy()
        elif resolved_threshold > 0.0:
            local = local.loc[local[prediction_col].abs() >= resolved_threshold].copy()

        local = local.loc[local[prediction_col] != 0.0].copy()
        resolved_vol_col = vol_col
        if vol_col_per_row is not None and vol_col_per_row in local.columns and not local.empty:
            resolved_vol_col = str(local[vol_col_per_row].iloc[0])
        local = local.loc[local[resolved_vol_col].notna() & (local[resolved_vol_col] > 0.0)].copy()
        if not local.empty:
            local["scale"] = (target_vol / local[resolved_vol_col]).clip(upper=leverage_cap)

        group["position"] = 0.0
        if not local.empty:
            signed_scale = np.where(local[prediction_col] > 0.0, 1.0, -1.0) * local["scale"].to_numpy()
            group.loc[local.index, "position"] = signed_scale
        positioned_groups.append(group)

    return pd.concat(positioned_groups, axis=0).sort_values(["timestamp", "SYMBOL"]).reset_index(drop=True)


# Build one sleeve frame with close returns, predictions, volatility, and target position.
def build_xgb_sleeve_frame(symbol_frame):
    frame = symbol_frame.sort_values("timestamp").copy()
    frame = frame.set_index("timestamp")
    output = pd.DataFrame(index=frame.index)
    output["close"] = frame["close"].astype(float)
    output["return"] = output["close"].pct_change()
    output["log_return"] = log_return(output["close"])
    output["prediction"] = frame["prediction"].astype(float)
    output["asset_vol_annualized"] = frame["asset_vol_annualized"].astype(float)
    output["position"] = frame["position"].astype(float)
    return output


# Turn a net strategy return stream into wealth, drawdown, and summary metrics.
def summarize_returns(init_amount, strategy_returns, fee_cost=None, summary_meta=None, active_mask=None):
    strategy_returns = pd.Series(strategy_returns, copy=False).astype(float)
    strategy_returns.index = pd.to_datetime(strategy_returns.index)

    periods_per_year = estimate_periods_per_year(strategy_returns.index)
    if not np.isfinite(periods_per_year):
        raise ValueError("Not enough data points to estimate annualization.")

    summary_data = pd.DataFrame(index=strategy_returns.index)
    summary_data["net_strategy_return"] = strategy_returns.fillna(0.0).clip(lower=-0.999999)
    summary_data["net_log_return"] = np.log1p(summary_data["net_strategy_return"])

    if fee_cost is None:
        summary_data["fee_cost"] = 0.0
    else:
        fee_series = pd.Series(fee_cost, copy=False).reindex(summary_data.index).fillna(0.0)
        summary_data["fee_cost"] = fee_series.astype(float)

    summary_data["wealth"] = np.exp(summary_data["net_log_return"].cumsum()) * init_amount
    summary_data["cum_fees"] = summary_data["fee_cost"].cumsum() * init_amount
    summary_data["running_peak"] = summary_data["wealth"].cummax()
    summary_data["drawdown%"] = (summary_data["wealth"] / summary_data["running_peak"]) - 1

    if active_mask is None:
        active_mask = summary_data["net_strategy_return"] != 0
    else:
        active_mask = pd.Series(active_mask, copy=False).reindex(summary_data.index).fillna(False)

    active_returns = summary_data.loc[active_mask, "net_strategy_return"].dropna()
    active_log_returns = summary_data.loc[active_mask, "net_log_return"].dropna()
    all_returns = summary_data["net_strategy_return"]

    win_rate = (active_returns > 0).mean() if not active_returns.empty else np.nan
    average_return_factor = (
        np.exp(active_log_returns.mean()) if not active_log_returns.empty else np.nan
    )

    annualized_return_mean = all_returns.mean() * periods_per_year
    annualized_return_std = all_returns.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = (
        annualized_return_mean / annualized_return_std
        if pd.notna(annualized_return_std) and annualized_return_std > 0
        else np.nan
    )

    elapsed_years = (
        (strategy_returns.index[-1] - strategy_returns.index[0]).total_seconds()
        / (365.25 * 24 * 60 * 60)
        if len(strategy_returns.index) >= 2
        else np.nan
    )
    final_wealth = summary_data["wealth"].iloc[-1] if not summary_data.empty else np.nan
    yearly_factor = (
        (final_wealth / init_amount) ** (1 / elapsed_years)
        if pd.notna(final_wealth) and pd.notna(elapsed_years) and elapsed_years > 0
        else np.nan
    )

    summary_values = dict(summary_meta or {})
    summary_values.update(
        {
            "yearly_factor": yearly_factor,
            "total_fees": summary_data["fee_cost"].sum(),
            "max_drawdown": summary_data["drawdown%"].min() if not summary_data.empty else np.nan,
            "winrate": win_rate,
            "average_return_factor": average_return_factor,
            "sharpe_ratio_annualized": sharpe,
        }
    )

    return summary_data, pd.DataFrame([summary_values])


# Compute wealth-based yearly factor and max drawdown metrics from an existing wealth curve.
def summarize_wealth_curve(init_amount, wealth):
    wealth = pd.Series(wealth, copy=False).astype(float).dropna()
    wealth.index = pd.to_datetime(wealth.index)

    if wealth.empty:
        raise ValueError("Wealth series is empty.")

    summary_data = pd.DataFrame(index=wealth.index)
    summary_data["wealth"] = wealth
    summary_data["running_peak"] = summary_data["wealth"].cummax()
    summary_data["drawdown%"] = (
        summary_data["wealth"] / summary_data["running_peak"]
    ) - 1

    elapsed_years = (
        (wealth.index[-1] - wealth.index[0]).total_seconds() / (365.25 * 24 * 60 * 60)
        if len(wealth.index) >= 2
        else np.nan
    )
    final_wealth = summary_data["wealth"].iloc[-1]
    yearly_factor = (
        (final_wealth / init_amount) ** (1 / elapsed_years)
        if pd.notna(elapsed_years) and elapsed_years > 0
        else np.nan
    )

    return summary_data, {
        "yearly_factor": yearly_factor,
        "max_drawdown": summary_data["drawdown%"].min(),
    }


# Compute a simple equal-weight buy-and-hold baseline with per-bar rebalancing fees.
def calculate_buy_and_hold_baseline(
    close_source,
    init_amount,
    target_vol=None,
    vol_window=None,
    fees=None,
    summary_meta=None,
    evaluation_index=None,
):
    if fees is None:
        raise ValueError("Pass fee parameters explicitly from the strategy or ignored notebook.")
    close_map = {
        symbol: pd.Series(series, copy=False).astype(float).dropna()
        for symbol, series in extract_close_map(close_source).items()
    }

    if not close_map:
        raise ValueError("No close data available for buy-and-hold baseline.")

    baseline_meta = dict(summary_meta or {})
    if evaluation_index is not None:
        evaluation_index = pd.DatetimeIndex(pd.to_datetime(evaluation_index)).drop_duplicates().sort_values()
        if evaluation_index.empty:
            raise ValueError("evaluation_index is empty.")

    if evaluation_index is None:
        index = pd.DatetimeIndex(sorted(set().union(*[series.index for series in close_map.values()])))
    else:
        index = evaluation_index

    close_frame = pd.DataFrame(
        {
            symbol: series.reindex(index).ffill()
            for symbol, series in close_map.items()
        },
        index=index,
    ).dropna(how="any")

    if close_frame.empty:
        raise ValueError("No common close data available for buy-and-hold baseline.")

    asset_returns = close_frame.pct_change().fillna(0.0)
    asset_log_returns = np.log(close_frame / close_frame.shift(1))
    n_assets = len(close_frame.columns)
    target_weights = pd.DataFrame(
        1.0 / n_assets,
        index=close_frame.index,
        columns=close_frame.columns,
    )
    previous_weights = target_weights.shift(1).fillna(0.0)
    gross_by_asset = previous_weights * asset_returns
    gross_returns = gross_by_asset.sum(axis=1)

    drifted_weights = pd.DataFrame(0.0, index=close_frame.index, columns=close_frame.columns)
    denominator = (1.0 + gross_returns).replace(0.0, np.nan)
    drifted_weights.iloc[1:] = (
        previous_weights.iloc[1:]
        * (1.0 + asset_returns.iloc[1:])
    ).div(denominator.iloc[1:], axis=0).fillna(0.0)

    turnover_by_asset = (target_weights - drifted_weights).abs()
    if not turnover_by_asset.empty:
        turnover_by_asset.iloc[0] = target_weights.iloc[0].abs()

    fee_by_asset = turnover_by_asset * fees
    portfolio_fees = fee_by_asset.sum(axis=1)
    portfolio_returns = gross_returns - portfolio_fees
    active_previous = previous_weights.abs().sum(axis=1) > 0

    portfolio_data, summary = summarize_returns(
        init_amount=init_amount,
        strategy_returns=portfolio_returns,
        fee_cost=portfolio_fees,
        summary_meta=baseline_meta,
        active_mask=active_previous,
    )
    summary["turnover"] = turnover_by_asset.sum(axis=1).mean()

    if n_assets == 1:
        symbol = close_frame.columns[0]
        baseline_frame = pd.DataFrame(index=close_frame.index)
        baseline_frame["close"] = close_frame[symbol]
        baseline_frame["return"] = asset_returns[symbol]
        baseline_frame["log_return"] = asset_log_returns[symbol]
        baseline_frame["position"] = target_weights[symbol]
        baseline_frame["position_prev"] = previous_weights[symbol]
        baseline_frame["asset_simple_return"] = asset_returns[symbol]
        baseline_frame["asset_log_return"] = asset_log_returns[symbol]
        baseline_frame["gross_strategy_return"] = gross_by_asset[symbol]
        baseline_frame["turnover"] = turnover_by_asset[symbol]
        baseline_frame["fee_cost"] = fee_by_asset[symbol]
        baseline_frame["net_strategy_return"] = portfolio_data["net_strategy_return"]
        baseline_frame["net_log_return"] = portfolio_data["net_log_return"]
        baseline_frame["wealth"] = portfolio_data["wealth"]
        baseline_frame["cum_fees"] = portfolio_data["cum_fees"]
        baseline_frame["running_peak"] = portfolio_data["running_peak"]
        baseline_frame["drawdown%"] = portfolio_data["drawdown%"]
        return baseline_frame, summary

    frames = {}
    for symbol in close_frame.columns:
        frame = pd.DataFrame(index=close_frame.index)
        frame["close"] = close_frame[symbol]
        frame["return"] = asset_returns[symbol]
        frame["log_return"] = asset_log_returns[symbol]
        frame["position"] = target_weights[symbol]
        frame["position_prev"] = previous_weights[symbol]
        frame["asset_simple_return"] = asset_returns[symbol]
        frame["asset_log_return"] = asset_log_returns[symbol]
        frame["gross_strategy_return"] = gross_by_asset[symbol]
        frame["turnover"] = turnover_by_asset[symbol]
        frame["fee_cost"] = fee_by_asset[symbol]
        frame["net_strategy_return"] = gross_by_asset[symbol] - fee_by_asset[symbol]
        frame["net_log_return"] = np.log1p(frame["net_strategy_return"].clip(lower=-0.999999))
        frames[symbol] = frame

    portfolio_frame = pd.DataFrame(index=close_frame.index)
    portfolio_frame["active_assets"] = n_assets
    portfolio_frame["net_strategy_return"] = portfolio_data["net_strategy_return"]
    portfolio_frame["net_log_return"] = portfolio_data["net_log_return"]
    portfolio_frame["turnover"] = turnover_by_asset.sum(axis=1)
    portfolio_frame["fee_cost"] = portfolio_data["fee_cost"]
    portfolio_frame["cum_fees"] = portfolio_data["cum_fees"]
    portfolio_frame["wealth"] = portfolio_data["wealth"]
    portfolio_frame["running_peak"] = portfolio_data["running_peak"]
    portfolio_frame["drawdown%"] = portfolio_data["drawdown%"]
    frames["portfolio"] = portfolio_frame
    return pd.concat(frames, axis=1), summary


# Turn asset returns and positions into a full performance dataframe and summary metrics.
def calculate_performance(init_amount, returns, positions, fees=0.0005, log_return=False, summary_meta=None):
    returns = pd.Series(returns, copy=False)
    positions = pd.Series(positions, copy=False)

    data = pd.concat([returns.rename("input_return"), positions.rename("position")], axis=1)
    data.index = pd.to_datetime(data.index)

    if log_return:
        data["asset_log_return"] = data["input_return"].astype(float)
        data["asset_simple_return"] = np.expm1(data["asset_log_return"])
    else:
        data["asset_simple_return"] = data["input_return"].astype(float)
        clipped_simple_return = data["asset_simple_return"].clip(lower=-0.999999)
        data["asset_log_return"] = np.log1p(clipped_simple_return)

    data["position_prev"] = data["position"].shift(1).fillna(0.0)
    data["gross_strategy_return"] = (
        data["position_prev"] * data["asset_simple_return"].fillna(0.0)
    )
    data["turnover"] = (data["position"].fillna(0.0) - data["position_prev"]).abs()
    data["fee_cost"] = data["turnover"] * fees
    data["net_strategy_return"] = data["gross_strategy_return"] - data["fee_cost"]

    summary_data, summary = summarize_returns(
        init_amount=init_amount,
        strategy_returns=data["net_strategy_return"],
        fee_cost=data["fee_cost"],
        summary_meta=summary_meta,
        active_mask=data["position_prev"] != 0,
    )

    for column in ["net_log_return", "wealth", "cum_fees", "running_peak", "drawdown%"]:
        data[column] = summary_data[column]
    summary["turnover"] = data["turnover"].fillna(0.0).mean()
    return data, summary


# Merge fully-constructed sleeve frames onto a shared index and build the portfolio from carried positions.
def combine_sleeve_frames(sleeve_frames, init_amount, fees, summary_meta=None):
    union_index = pd.DatetimeIndex(
        sorted(set().union(*[frame.index for frame in sleeve_frames.values()]))
    )
    merged_frames = {}
    carried_positions = {}

    for symbol, frame in sleeve_frames.items():
        merged = frame.reindex(union_index)
        merged["close"] = merged["close"].ffill()
        merged["position"] = merged["position"].ffill().fillna(0.0)
        merged["native_turnover"] = merged["turnover"].fillna(0.0)
        merged["native_fee_cost"] = merged["fee_cost"].fillna(0.0)
        merged["asset_simple_return"] = merged["close"].pct_change().fillna(0.0)
        merged["active"] = merged["position"] != 0
        merged_frames[symbol] = merged
        carried_positions[symbol] = merged["position"]

    carried_positions = pd.DataFrame(carried_positions, index=union_index)
    active_assets = (carried_positions != 0).sum(axis=1)

    portfolio_return_parts = []
    portfolio_fee_parts = []
    portfolio_turnover_parts = []

    for symbol, merged in merged_frames.items():
        weight = pd.Series(
            np.where(active_assets > 0, 1.0 / active_assets, 0.0),
            index=union_index,
        )
        merged["weight"] = np.where(merged["active"], weight, 0.0)
        merged["weighted_position"] = merged["position"] * merged["weight"]
        merged["weighted_position_prev"] = merged["weighted_position"].shift(1).fillna(0.0)
        merged["gross_strategy_return"] = (
            merged["weighted_position_prev"] * merged["asset_simple_return"]
        )
        merged["turnover"] = merged["native_turnover"] * merged["weight"]
        merged["fee_cost"] = merged["native_fee_cost"] * merged["weight"]
        merged["net_strategy_return"] = merged["gross_strategy_return"] - merged["fee_cost"]
        merged["net_log_return"] = np.log1p(merged["net_strategy_return"].clip(lower=-0.999999))
        merged_frames[symbol] = merged

        portfolio_return_parts.append(merged["net_strategy_return"].rename(symbol))
        portfolio_fee_parts.append(merged["fee_cost"].rename(symbol))
        portfolio_turnover_parts.append(merged["turnover"].rename(symbol))

    portfolio_returns = pd.concat(portfolio_return_parts, axis=1).sum(axis=1)
    portfolio_fees = pd.concat(portfolio_fee_parts, axis=1).sum(axis=1)
    portfolio_turnover = pd.concat(portfolio_turnover_parts, axis=1).sum(axis=1)
    active_previous = (
        pd.concat(
            [frame["weighted_position_prev"].rename(symbol) for symbol, frame in merged_frames.items()],
            axis=1,
        ).abs().sum(axis=1)
        > 0
    )

    portfolio_data, summary = summarize_returns(
        init_amount=init_amount,
        strategy_returns=portfolio_returns,
        fee_cost=portfolio_fees,
        summary_meta=summary_meta,
        active_mask=active_previous,
    )

    portfolio_frame = pd.DataFrame(index=union_index)
    portfolio_frame["active_assets"] = active_assets
    portfolio_frame["net_strategy_return"] = portfolio_data["net_strategy_return"]
    portfolio_frame["net_log_return"] = portfolio_data["net_log_return"]
    portfolio_frame["turnover"] = portfolio_turnover
    portfolio_frame["fee_cost"] = portfolio_data["fee_cost"]
    portfolio_frame["cum_fees"] = portfolio_data["cum_fees"]
    portfolio_frame["wealth"] = portfolio_data["wealth"]
    portfolio_frame["running_peak"] = portfolio_data["running_peak"]
    portfolio_frame["drawdown%"] = portfolio_data["drawdown%"]

    merged_frames["portfolio"] = portfolio_frame
    summary["turnover"] = portfolio_turnover.fillna(0.0).mean()
    return pd.concat(merged_frames, axis=1), summary


# Plot a single wealth curve in a seaborn-style chart, with an optional benchmark overlay.
def _max_drawdown_window(wealth):
    wealth = pd.Series(wealth, copy=False).astype(float).dropna()
    if wealth.empty:
        return None

    running_peak = wealth.cummax()
    drawdown = (wealth / running_peak) - 1
    trough_time = drawdown.idxmin()

    if pd.isna(trough_time) or drawdown.loc[trough_time] >= 0:
        return None

    peak_time = wealth.loc[:trough_time].idxmax()
    peak_wealth = wealth.loc[peak_time]
    trough_wealth = wealth.loc[trough_time]
    return peak_time, trough_time, peak_wealth, trough_wealth


# Plot a single wealth curve in a seaborn-style chart, with an optional benchmark overlay.
def plot_wealth(
    wealth,
    title="Strategy Wealth",
    log_scale=True,
    benchmark_wealth=None,
    benchmark_label="benchmark",
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    wealth = pd.Series(wealth, copy=False)
    benchmark = None if benchmark_wealth is None else pd.Series(benchmark_wealth, copy=False)

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(wealth.index, wealth, label="strategy wealth", linewidth=2.0)

    strategy_drawdown_window = _max_drawdown_window(wealth)
    if strategy_drawdown_window is not None:
        peak_time, trough_time, peak_wealth, trough_wealth = strategy_drawdown_window
        strategy_mask = (wealth.index >= peak_time) & (wealth.index <= trough_time)
        ax.fill_between(
            wealth.index[strategy_mask],
            trough_wealth,
            peak_wealth,
            color="steelblue",
            alpha=0.12,
            label="strategy max drawdown",
        )

    if benchmark is not None and not benchmark.empty:
        ax.plot(
            benchmark.index,
            benchmark,
            label=benchmark_label,
            linewidth=2.0,
            color="darkorange",
        )
        benchmark_drawdown_window = _max_drawdown_window(benchmark)
        if benchmark_drawdown_window is not None:
            peak_time, trough_time, peak_wealth, trough_wealth = benchmark_drawdown_window
            benchmark_mask = (benchmark.index >= peak_time) & (benchmark.index <= trough_time)
            ax.fill_between(
                benchmark.index[benchmark_mask],
                trough_wealth,
                peak_wealth,
                color="darkorange",
                alpha=0.10,
                label=f"{benchmark_label} max drawdown",
            )

    if log_scale:
        ax.set_yscale("log")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Wealth")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig, ax


# Pick a reasonable rolling estimation window when the user does not specify one.
def infer_block_length(n_observations):
    if n_observations <= 1:
        return 1

    return max(5, min(n_observations, int(round(np.sqrt(n_observations)))))


# Build a sensible grid of candidate estimation windows from the sample length.
def build_candidate_windows(n_observations):
    if n_observations <= 5:
        return []
    return [max(5, min(n_observations - 1, infer_block_length(n_observations)))]


# Fill rolling mean and volatility estimates so the simulation can start from the first return.
def prepare_rolling_gbm_parameters(log_returns, window):
    window = max(1, min(int(window), len(log_returns)))
    rolling_mean = log_returns.rolling(window=window, min_periods=window).mean()
    rolling_std = log_returns.rolling(window=window, min_periods=window).std(ddof=1)

    fallback_mean = log_returns.mean()
    fallback_std = log_returns.std(ddof=1)
    if not np.isfinite(fallback_std) or fallback_std <= 0:
        fallback_std = 1e-8

    rolling_mean = rolling_mean.fillna(fallback_mean)
    rolling_std = rolling_std.replace(0.0, np.nan).fillna(fallback_std).clip(lower=1e-8)
    return rolling_mean, rolling_std


# Compute predictive rolling parameters using only prior information for each return.
def prepare_predictive_rolling_gbm_parameters(log_returns, window):
    rolling_mean = log_returns.rolling(window=window, min_periods=window).mean().shift(1)
    rolling_std = log_returns.rolling(window=window, min_periods=window).std(ddof=1).shift(1)

    fallback_mean = log_returns.mean()
    fallback_std = log_returns.std(ddof=1)
    if not np.isfinite(fallback_std) or fallback_std <= 0:
        fallback_std = 1e-8

    rolling_mean = rolling_mean.fillna(fallback_mean)
    rolling_std = rolling_std.replace(0.0, np.nan).fillna(fallback_std).clip(lower=1e-8)
    return rolling_mean, rolling_std


# Convert simulated log returns into synthetic close paths.
def build_simulated_close_paths(start_price, simulated_log_returns, index):
    cumulative_log_returns = np.vstack(
        [np.zeros(simulated_log_returns.shape[1]), np.cumsum(simulated_log_returns, axis=0)]
    )
    simulated_close = start_price * np.exp(cumulative_log_returns)
    columns = [f"path_{path_id:04d}" for path_id in range(1, simulated_log_returns.shape[1] + 1)]
    return pd.DataFrame(simulated_close, index=index, columns=columns)


# Stabilize a covariance matrix before drawing multivariate normal shocks.
def stabilize_covariance_matrix(covariance_matrix):
    covariance_matrix = np.nan_to_num(covariance_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    covariance_matrix = (covariance_matrix + covariance_matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


# Score one window on a single series using one-step-ahead Gaussian predictive likelihood.
def score_single_window(log_returns, window):
    predictive_mean, predictive_std = prepare_predictive_rolling_gbm_parameters(log_returns, window)
    variance = predictive_std.pow(2).clip(lower=1e-12)
    score = -0.5 * (
        np.log(2 * np.pi * variance) + ((log_returns - predictive_mean) ** 2) / variance
    )
    return float(score.replace([np.inf, -np.inf], np.nan).dropna().sum())


# Score one window on a basket using multivariate Gaussian predictive likelihood.
def score_multi_asset_window(log_returns_frame, window):
    predictive_mean = pd.DataFrame(index=log_returns_frame.index, columns=log_returns_frame.columns, dtype=float)
    predictive_std = pd.DataFrame(index=log_returns_frame.index, columns=log_returns_frame.columns, dtype=float)

    for symbol in log_returns_frame.columns:
        symbol_mean, symbol_std = prepare_predictive_rolling_gbm_parameters(log_returns_frame[symbol], window)
        predictive_mean[symbol] = symbol_mean
        predictive_std[symbol] = symbol_std

    correlation_matrix = log_returns_frame.corr().fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(correlation_matrix, 1.0)

    score = 0.0
    for step_number in range(len(log_returns_frame)):
        mean_vector = predictive_mean.iloc[step_number].to_numpy(dtype=float)
        std_vector = predictive_std.iloc[step_number].to_numpy(dtype=float)
        covariance_matrix = np.outer(std_vector, std_vector) * correlation_matrix
        covariance_matrix = stabilize_covariance_matrix(covariance_matrix)
        observed_vector = log_returns_frame.iloc[step_number].to_numpy(dtype=float)
        diff = observed_vector - mean_vector

        sign, logdet = np.linalg.slogdet(covariance_matrix)
        if sign <= 0:
            continue

        inverse_covariance = np.linalg.inv(covariance_matrix)
        dimension = len(observed_vector)
        score += -0.5 * (
            dimension * np.log(2 * np.pi) + logdet + diff @ inverse_covariance @ diff
        )

    return float(score)


# Select the rolling estimation window from the return process itself.
def select_monte_carlo_window(close, candidate_windows=None):
    if isinstance(close, pd.Series):
        series = pd.Series(close, copy=False).astype(float).dropna()
        log_returns = np.log(series / series.shift(1)).dropna()

        if len(log_returns) <= 5:
            return infer_block_length(len(log_returns))

        windows = candidate_windows or build_candidate_windows(len(log_returns))
        scored_windows = [(window, score_single_window(log_returns, window)) for window in windows]
        return max(scored_windows, key=lambda item: item[1])[0]

    if isinstance(close, dict):
        close_map = {
            symbol: pd.Series(series, copy=False).astype(float).dropna()
            for symbol, series in close.items()
        }
        native_indexes = {symbol: series.index for symbol, series in close_map.items()}
        union_index = pd.DatetimeIndex(sorted(set().union(*native_indexes.values())))
        aligned_close = pd.DataFrame(index=union_index)
        for symbol, series in close_map.items():
            aligned_close[symbol] = series.reindex(union_index).ffill()

        log_returns_frame = np.log(aligned_close / aligned_close.shift(1)).fillna(0.0).iloc[1:]
        if len(log_returns_frame) <= 5:
            return infer_block_length(len(log_returns_frame))

        windows = candidate_windows or build_candidate_windows(len(log_returns_frame))
        scored_windows = [(window, score_multi_asset_window(log_returns_frame, window)) for window in windows]
        return max(scored_windows, key=lambda item: item[1])[0]

    frame = pd.DataFrame(close, copy=False).astype(float).dropna(how="all")
    close_map = {symbol: frame[symbol].dropna() for symbol in frame.columns}
    return select_monte_carlo_window(close_map, candidate_windows=candidate_windows)


# Simulate one-sleeve paths from rolling log-return mean and volatility estimates.
def simulate_single_gbm_paths(series, n_paths, rng, block_length):
    series = pd.Series(series, copy=False).astype(float).dropna()
    log_returns = np.log(series / series.shift(1)).dropna()

    if log_returns.empty:
        raise ValueError("Not enough price history to generate Monte Carlo paths.")

    window = select_monte_carlo_window(series) if block_length is None else block_length
    rolling_mean, rolling_std = prepare_rolling_gbm_parameters(log_returns, window)
    simulated_log_returns = rng.normal(
        loc=rolling_mean.to_numpy()[:, None],
        scale=rolling_std.to_numpy()[:, None],
        size=(len(log_returns), n_paths),
    )
    return build_simulated_close_paths(series.iloc[0], simulated_log_returns, series.index)


# Simulate basket paths from rolling mean/volatility estimates plus empirical cross-asset residual shocks.
def simulate_multi_asset_gbm_paths(close_map, n_paths, rng, block_length):
    close_map = {
        symbol: pd.Series(series, copy=False).astype(float).dropna()
        for symbol, series in close_map.items()
    }
    native_indexes = {symbol: series.index for symbol, series in close_map.items()}

    union_index = pd.DatetimeIndex(sorted(set().union(*native_indexes.values())))
    aligned_close = pd.DataFrame(index=union_index)
    for symbol, series in close_map.items():
        aligned_close[symbol] = series.reindex(union_index).ffill()

    aligned_log_returns = np.log(aligned_close / aligned_close.shift(1)).fillna(0.0)
    aligned_log_returns = aligned_log_returns.iloc[1:]

    if aligned_log_returns.empty:
        raise ValueError("Not enough price history to generate Monte Carlo paths.")

    window = select_monte_carlo_window(close_map) if block_length is None else block_length
    rolling_mean = pd.DataFrame(index=aligned_log_returns.index, columns=aligned_log_returns.columns, dtype=float)
    rolling_std = pd.DataFrame(index=aligned_log_returns.index, columns=aligned_log_returns.columns, dtype=float)

    for symbol in aligned_log_returns.columns:
        symbol_mean, symbol_std = prepare_rolling_gbm_parameters(aligned_log_returns[symbol], window)
        rolling_mean[symbol] = symbol_mean
        rolling_std[symbol] = symbol_std

    correlation_matrix = aligned_log_returns.corr().fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(correlation_matrix, 1.0)

    simulated_log_returns = np.zeros((len(aligned_log_returns), len(aligned_log_returns.columns), n_paths))

    for step_number, timestamp in enumerate(aligned_log_returns.index):
        std_vector = rolling_std.iloc[step_number].to_numpy(dtype=float)
        covariance_matrix = np.outer(std_vector, std_vector) * correlation_matrix
        covariance_matrix = stabilize_covariance_matrix(covariance_matrix)
        draws = rng.multivariate_normal(
            mean=rolling_mean.loc[timestamp].to_numpy(dtype=float),
            cov=covariance_matrix,
            size=n_paths,
        )
        simulated_log_returns[step_number] = draws.T

    simulated_paths = {}
    for column_number, symbol in enumerate(aligned_close.columns):
        base_price = close_map[symbol].iloc[0]
        native_index = native_indexes[symbol]
        start_location = union_index.get_loc(native_index[0])
        symbol_log_returns = simulated_log_returns[start_location:, column_number, :]
        symbol_index = union_index[start_location:]
        symbol_close = build_simulated_close_paths(base_price, symbol_log_returns, symbol_index)
        simulated_paths[symbol] = symbol_close.loc[native_index]

    return simulated_paths


# Simulate synthetic close paths from rolling GBM parameters estimated on history.
def generate_monte_carlo_paths(close, n_paths=250, seed=None, block_length=None):
    rng = np.random.default_rng(seed)

    if isinstance(close, pd.Series):
        return simulate_single_gbm_paths(close, n_paths=n_paths, rng=rng, block_length=block_length)

    if isinstance(close, dict):
        return simulate_multi_asset_gbm_paths(close, n_paths=n_paths, rng=rng, block_length=block_length)

    frame = pd.DataFrame(close, copy=False).astype(float).dropna(how="all")
    close_map = {symbol: frame[symbol].dropna() for symbol in frame.columns}
    return simulate_multi_asset_gbm_paths(close_map, n_paths=n_paths, rng=rng, block_length=block_length)


# Aggregate Monte Carlo path metrics and attach a confidence interval for the mean estimate.
def summarize_monte_carlo_results(path_summaries, metric_columns, confidence=0.95, summary_meta=None):
    if path_summaries.empty:
        raise ValueError("Monte Carlo path summaries are empty.")

    z_score = NormalDist().inv_cdf(0.5 + confidence / 2)
    summary_values = dict(summary_meta or {})
    summary_values["monte_carlo_paths"] = len(path_summaries)
    summary_values["confidence_level"] = confidence

    for column in metric_columns:
        values = pd.to_numeric(path_summaries[column], errors="coerce").dropna()
        mean_value = values.mean() if not values.empty else np.nan

        if len(values) > 1:
            std_error = values.std(ddof=1) / np.sqrt(len(values))
            margin = z_score * std_error
            ci_lower = mean_value - margin
            ci_upper = mean_value + margin
        else:
            ci_lower = np.nan
            ci_upper = np.nan

        summary_values[column] = mean_value
        summary_values[f"{column}_ci_lower"] = ci_lower
        summary_values[f"{column}_ci_upper"] = ci_upper

    return pd.DataFrame([summary_values])


# Run a generic Monte Carlo analysis by evaluating each simulated close path with a strategy callback.
def calculate_monte_carlo_performance(
    close,
    evaluator,
    metric_columns,
    n_paths=250,
    seed=None,
    confidence=0.95,
    summary_meta=None,
    block_length=None,
):
    simulated_paths = generate_monte_carlo_paths(
        close,
        n_paths=n_paths,
        seed=seed,
        block_length=block_length,
    )
    path_summaries = []
    wealth_paths = {}

    if isinstance(simulated_paths, dict):
        first_symbol = next(iter(simulated_paths))
        path_names = simulated_paths[first_symbol].columns
        for path_name in path_names:
            path_close = {symbol: paths[path_name] for symbol, paths in simulated_paths.items()}
            path_data, path_summary = evaluator(path_close)
            summary_row = path_summary.iloc[0].to_dict()
            summary_row["path"] = path_name
            path_summaries.append(summary_row)
            wealth_series = (
                path_data["portfolio"]["wealth"]
                if isinstance(path_data.columns, pd.MultiIndex)
                else path_data["wealth"]
            )
            wealth_paths[path_name] = wealth_series
        wealth_index = wealth_paths[path_names[0]].index
    else:
        path_names = simulated_paths.columns
        for path_name in path_names:
            path_close = simulated_paths[path_name]
            path_data, path_summary = evaluator(path_close)
            summary_row = path_summary.iloc[0].to_dict()
            summary_row["path"] = path_name
            path_summaries.append(summary_row)
            wealth_series = (
                path_data["portfolio"]["wealth"]
                if isinstance(path_data.columns, pd.MultiIndex)
                else path_data["wealth"]
            )
            wealth_paths[path_name] = wealth_series
        wealth_index = simulated_paths.index

    path_summaries = pd.DataFrame(path_summaries)
    wealth_paths = pd.DataFrame(wealth_paths, index=wealth_index)
    monte_carlo_summary = summarize_monte_carlo_results(
        path_summaries,
        metric_columns=metric_columns,
        confidence=confidence,
        summary_meta=summary_meta,
    )

    return {
        "paths": simulated_paths,
        "wealth_paths": wealth_paths,
        "path_summaries": path_summaries,
        "summary": monte_carlo_summary,
    }


# Plot the spread of Monte Carlo wealth paths together with the average path, a 95% envelope, and an optional benchmark.
def plot_monte_carlo_wealth(
    wealth_paths,
    title="Monte Carlo Wealth",
    log_scale=True,
    benchmark_wealth=None,
    benchmark_label="benchmark",
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    wealth_paths = pd.DataFrame(wealth_paths, copy=False)
    benchmark = None if benchmark_wealth is None else pd.Series(benchmark_wealth, copy=False)

    if wealth_paths.empty:
        raise ValueError("No Monte Carlo wealth paths available to plot.")

    lower_band = wealth_paths.quantile(0.025, axis=1)
    upper_band = wealth_paths.quantile(0.975, axis=1)
    mean_wealth = wealth_paths.mean(axis=1)

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    for column in wealth_paths.columns:
        ax.plot(wealth_paths.index, wealth_paths[column], color="steelblue", alpha=0.05, linewidth=0.8)

    ax.fill_between(
        wealth_paths.index,
        lower_band,
        upper_band,
        color="skyblue",
        alpha=0.25,
        label="95% path envelope",
    )
    ax.plot(wealth_paths.index, mean_wealth, color="navy", linewidth=2.0, label="mean wealth")

    if benchmark is not None and not benchmark.empty:
        ax.plot(
            benchmark.index,
            benchmark,
            color="darkorange",
            linewidth=2.0,
            linestyle="--",
            label=benchmark_label,
        )

    if log_scale:
        ax.set_yscale("log")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Wealth")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig, ax
