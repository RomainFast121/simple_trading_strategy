from datetime import timedelta, timezone
from statistics import NormalDist
import re

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import ccxt
except ImportError:  # pragma: no cover - handled at runtime when crypto data is requested
    ccxt = None


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
    return "1h" if interval == "1d" and hour is not None else interval


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
    data = yf.download(
        symbol,
        start=start,
        end=end,
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
    timeframe_ms = exchange.parse_timeframe(fetch_interval) * 1000

    start_ts = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ts = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)

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
    frame = frame[(frame.index >= pd.Timestamp(start, tz="UTC")) & (frame.index < pd.Timestamp(end, tz="UTC"))]

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

    ret_std = all_returns.std(ddof=1)
    sharpe = (
        (all_returns.mean() / ret_std) * np.sqrt(periods_per_year)
        if pd.notna(ret_std) and ret_std > 0
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


# Build one buy-and-hold sleeve with constant long exposure and target-vol scaling.
def _build_buy_and_hold_sleeve(close, target_vol, vol_window):
    close = pd.Series(close, copy=False).astype(float).dropna()

    frame = pd.DataFrame(index=close.index)
    frame["close"] = close
    frame["return"] = frame["close"].pct_change()
    frame["log_return"] = np.log(frame["close"] / frame["close"].shift(1))
    frame["signal"] = 1.0
    frame["recent_vol"] = rolling_annualized_vol(
        frame["log_return"],
        window=vol_window,
        min_periods=vol_window,
    )
    frame["position"] = frame["signal"] * (target_vol / frame["recent_vol"])
    frame.loc[~np.isfinite(frame["position"]), "position"] = np.nan
    return frame


# Compute a target-vol buy-and-hold baseline that follows the same sleeve and fee mechanics as the strategy.
def calculate_buy_and_hold_baseline(
    close_source,
    init_amount,
    target_vol,
    vol_window,
    fees=0.0005,
    summary_meta=None,
):
    if isinstance(close_source, pd.Series):
        close_map = {"asset": pd.Series(close_source, copy=False).astype(float).dropna()}
    elif isinstance(close_source, dict):
        close_map = {
            symbol: pd.Series(series, copy=False).astype(float).dropna()
            for symbol, series in close_source.items()
        }
    else:
        frame = pd.DataFrame(close_source, copy=False).astype(float).dropna(how="all")
        close_map = {symbol: frame[symbol].dropna() for symbol in frame.columns}

    if not close_map:
        raise ValueError("No close data available for buy-and-hold baseline.")

    baseline_meta = dict(summary_meta or {})

    if len(close_map) == 1:
        ticker_name = next(iter(close_map))
        sleeve_frame = _build_buy_and_hold_sleeve(
            close_map[ticker_name],
            target_vol=target_vol,
            vol_window=vol_window,
        )
        performance_data, summary = calculate_performance(
            init_amount=init_amount,
            returns=sleeve_frame["log_return"],
            positions=sleeve_frame["position"],
            fees=fees,
            log_return=True,
            summary_meta=baseline_meta,
        )
        baseline_frame = sleeve_frame.join(
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
        return baseline_frame, summary

    sleeve_frames = {}
    for symbol, close_series in close_map.items():
        sleeve_frame = _build_buy_and_hold_sleeve(
            close_series,
            target_vol=target_vol,
            vol_window=vol_window,
        )
        performance_data, _ = calculate_performance(
            init_amount=init_amount,
            returns=sleeve_frame["log_return"],
            positions=sleeve_frame["position"],
            fees=fees,
            log_return=True,
            summary_meta=baseline_meta,
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

    baseline_data, summary = combine_sleeve_frames(
        sleeve_frames=sleeve_frames,
        init_amount=init_amount,
        fees=fees,
        summary_meta=baseline_meta,
    )
    return baseline_data, summary


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

    portfolio_returns = pd.concat(portfolio_return_parts, axis=1).sum(axis=1)
    portfolio_fees = pd.concat(portfolio_fee_parts, axis=1).sum(axis=1)
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
    portfolio_frame["fee_cost"] = portfolio_data["fee_cost"]
    portfolio_frame["cum_fees"] = portfolio_data["cum_fees"]
    portfolio_frame["wealth"] = portfolio_data["wealth"]
    portfolio_frame["running_peak"] = portfolio_data["running_peak"]
    portfolio_frame["drawdown%"] = portfolio_data["drawdown%"]

    merged_frames["portfolio"] = portfolio_frame
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
    base_candidates = [10, 20, 30, 45, 60, 70, 90, 120, 180, 252]
    candidates = sorted({window for window in base_candidates if 5 <= window < n_observations})

    if not candidates and n_observations > 5:
        candidates = [max(5, min(n_observations - 1, infer_block_length(n_observations)))]

    return candidates


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
