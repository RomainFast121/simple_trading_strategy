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


# Plot a single wealth curve in a seaborn-style chart.
def plot_wealth(wealth, title="Strategy Wealth", log_scale=True):
    import matplotlib.pyplot as plt
    import seaborn as sns

    wealth = pd.Series(wealth, copy=False)

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(wealth.index, wealth, label="wealth")

    if log_scale:
        ax.set_yscale("log")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Wealth")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig, ax


# Pick a reasonable block length when the user does not specify one.
def infer_block_length(n_observations):
    if n_observations <= 1:
        return 1

    return max(5, min(n_observations, int(round(np.sqrt(n_observations)))))


# Sample one synthetic path by stitching together contiguous blocks of returns.
def sample_block_bootstrap_returns(return_array, block_length, rng):
    n_observations = len(return_array)
    if n_observations == 0:
        raise ValueError("Not enough return history to generate Monte Carlo paths.")

    block_length = max(1, min(int(block_length), n_observations))
    max_start = n_observations - block_length
    sampled_blocks = []
    collected = 0

    while collected < n_observations:
        start = rng.integers(0, max_start + 1)
        block = return_array[start : start + block_length]
        sampled_blocks.append(block)
        collected += len(block)

    return np.concatenate(sampled_blocks)[:n_observations]


# Convert sampled simple returns back into simulated close paths.
def build_simulated_close_paths(start_price, sampled_returns, index):
    compounded_paths = np.vstack([np.ones(sampled_returns.shape[1]), np.cumprod(1.0 + sampled_returns, axis=0)])
    simulated_close = start_price * compounded_paths
    columns = [f"path_{path_id:04d}" for path_id in range(1, sampled_returns.shape[1] + 1)]
    return pd.DataFrame(simulated_close, index=index, columns=columns)


# Bootstrap historical simple returns with contiguous time blocks so local regimes are preserved better.
def generate_monte_carlo_paths(close, n_paths=250, seed=None, block_length=None):
    rng = np.random.default_rng(seed)

    if isinstance(close, pd.Series):
        series = pd.Series(close, copy=False).astype(float).dropna()
        simple_returns = series.pct_change().dropna()

        if simple_returns.empty:
            raise ValueError("Not enough price history to generate Monte Carlo paths.")

        block_length = infer_block_length(len(simple_returns)) if block_length is None else block_length
        sampled_returns = np.column_stack(
            [
                sample_block_bootstrap_returns(simple_returns.to_numpy(), block_length, rng)
                for _ in range(n_paths)
            ]
        )
        return build_simulated_close_paths(series.iloc[0], sampled_returns, series.index)

    if isinstance(close, dict):
        close_map = {
            symbol: pd.Series(series, copy=False).astype(float).dropna()
            for symbol, series in close.items()
        }
        native_indexes = {symbol: series.index for symbol, series in close_map.items()}

        aligned_close = pd.DataFrame(index=sorted(set().union(*native_indexes.values())))
        for symbol, series in close_map.items():
            aligned_close[symbol] = series.reindex(aligned_close.index).ffill()

        aligned_returns = aligned_close.pct_change().fillna(0.0)

        if len(aligned_returns) <= 1:
            raise ValueError("Not enough price history to generate Monte Carlo paths.")

        sampled_matrix = aligned_returns.iloc[1:].to_numpy()
        block_length = infer_block_length(len(sampled_matrix)) if block_length is None else block_length

        simulated_return_paths = []
        for _ in range(n_paths):
            simulated_return_paths.append(
                sample_block_bootstrap_returns(sampled_matrix, block_length, rng)
            )

        simulated_return_paths = np.stack(simulated_return_paths, axis=2)
        simulated_paths = {}

        for column_number, symbol in enumerate(aligned_close.columns):
            symbol_returns = simulated_return_paths[:, column_number, :]
            base_price = aligned_close[symbol].dropna().iloc[0]
            symbol_close = build_simulated_close_paths(
                base_price,
                symbol_returns,
                aligned_close.index,
            )
            native_index = native_indexes[symbol]
            symbol_close = symbol_close.loc[native_index]
            simulated_paths[symbol] = symbol_close

        return simulated_paths

    frame = pd.DataFrame(close, copy=False).astype(float).dropna(how="all")
    close_map = {symbol: frame[symbol].dropna() for symbol in frame.columns}
    return generate_monte_carlo_paths(close_map, n_paths=n_paths, seed=seed, block_length=block_length)


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


# Plot the spread of Monte Carlo wealth paths together with the average path and a 95% envelope.
def plot_monte_carlo_wealth(wealth_paths, title="Monte Carlo Wealth", log_scale=True):
    import matplotlib.pyplot as plt
    import seaborn as sns

    wealth_paths = pd.DataFrame(wealth_paths, copy=False)

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

    if log_scale:
        ax.set_yscale("log")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Wealth")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig, ax
