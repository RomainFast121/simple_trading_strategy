from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf


# Download raw market data from Yahoo Finance and normalize the output columns.
def fetch_data(ticker, start, end, interval="1d", auto_adjust=True, progress=False):
    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=progress,
    )

    if data.empty:
        raise ValueError(f"No Yahoo Finance data returned for {ticker}.")

    data.index = pd.to_datetime(data.index)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = pd.MultiIndex.from_tuples(
            [(str(level_0).lower(), str(level_1)) for level_0, level_1 in data.columns]
        )
    else:
        data = data.rename(columns=str.lower)

    return data


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
def summarize_returns(strategy_returns, fee_cost=None, summary_meta=None, active_mask=None):
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

    summary_data["wealth"] = np.exp(summary_data["net_log_return"].cumsum())
    summary_data["cum_fees"] = summary_data["fee_cost"].cumsum()
    summary_data["running_peak"] = summary_data["wealth"].cummax()
    summary_data["drawdown"] = (summary_data["wealth"] / summary_data["running_peak"]) - 1.0

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

    summary_values = dict(summary_meta or {})
    summary_values.update(
        {
            "total_pnl": summary_data["wealth"].iloc[-1] - 1 if not summary_data.empty else np.nan,
            "total_fees": summary_data["fee_cost"].sum(),
            "max_drawdown": summary_data["drawdown"].min() if not summary_data.empty else np.nan,
            "winrate": win_rate,
            "average_return_factor": average_return_factor,
            "sharpe_ratio_annualized": sharpe,
        }
    )

    return summary_data, pd.DataFrame([summary_values])


# Turn asset returns and positions into a full performance dataframe and summary metrics.
def calculate_performance(returns, positions, fees=0.0, log_return=False, summary_meta=None):
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
        strategy_returns=data["net_strategy_return"],
        fee_cost=data["fee_cost"],
        summary_meta=summary_meta,
        active_mask=data["position_prev"] != 0,
    )

    for column in ["net_log_return", "wealth", "cum_fees", "running_peak", "drawdown"]:
        data[column] = summary_data[column]
    return data, summary


# Plot a single wealth curve in a seaborn-style chart.
def plot_wealth(wealth, title="Strategy Wealth", log_scale=True):
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


# Bootstrap historical simple returns to generate synthetic close paths for one or many tickers.
def generate_monte_carlo_paths(close, n_paths=250, seed=None):
    rng = np.random.default_rng(seed)

    if isinstance(close, pd.Series):
        close = pd.Series(close, copy=False).astype(float).dropna()
        simple_returns = close.pct_change().dropna()

        if simple_returns.empty:
            raise ValueError("Not enough price history to generate Monte Carlo paths.")

        sampled_returns = rng.choice(
            simple_returns.to_numpy(),
            size=(len(simple_returns), n_paths),
            replace=True,
        )
        compounded_paths = np.vstack(
            [np.ones(n_paths), np.cumprod(1.0 + sampled_returns, axis=0)]
        )
        simulated_close = close.iloc[0] * compounded_paths
        columns = [f"path_{path_id:04d}" for path_id in range(1, n_paths + 1)]
        return pd.DataFrame(simulated_close, index=close.index, columns=columns)

    close = pd.DataFrame(close, copy=False).astype(float).dropna(how="all")
    path_frames = {}

    for ticker in close.columns:
        ticker_close = close[ticker].dropna()
        ticker_paths = generate_monte_carlo_paths(ticker_close, n_paths=n_paths, seed=rng.integers(0, 10**9))
        path_frames[ticker] = ticker_paths.reindex(close.index)

    return pd.concat(path_frames, axis=1)


# Aggregate Monte Carlo path metrics and attach a confidence interval for the mean estimate.
def summarize_monte_carlo_results(
    path_summaries,
    metric_columns,
    confidence=0.95,
    summary_meta=None,
):
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
):
    simulated_paths = generate_monte_carlo_paths(close, n_paths=n_paths, seed=seed)
    path_summaries = []
    wealth_paths = {}

    if isinstance(simulated_paths.columns, pd.MultiIndex):
        path_names = simulated_paths.columns.get_level_values(1).unique()
        for path_name in path_names:
            path_close = simulated_paths.xs(path_name, axis=1, level=1)
            path_data, path_summary = evaluator(path_close)
            summary_row = path_summary.iloc[0].to_dict()
            summary_row["path"] = path_name
            path_summaries.append(summary_row)
            wealth_paths[path_name] = path_data["portfolio"]["wealth"]
    else:
        for path_name in simulated_paths.columns:
            path_close = simulated_paths[path_name]
            path_data, path_summary = evaluator(path_close)
            summary_row = path_summary.iloc[0].to_dict()
            summary_row["path"] = path_name
            path_summaries.append(summary_row)
            wealth_paths[path_name] = path_data["wealth"]

    path_summaries = pd.DataFrame(path_summaries)
    wealth_paths = pd.DataFrame(wealth_paths, index=simulated_paths.index)
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
    wealth_paths = pd.DataFrame(wealth_paths, copy=False)

    if wealth_paths.empty:
        raise ValueError("No Monte Carlo wealth paths available to plot.")

    lower_band = wealth_paths.quantile(0.025, axis=1)
    upper_band = wealth_paths.quantile(0.975, axis=1)
    mean_wealth = wealth_paths.mean(axis=1)

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    for column in wealth_paths.columns:
        ax.plot(
            wealth_paths.index,
            wealth_paths[column],
            color="steelblue",
            alpha=0.05,
            linewidth=0.8,
        )

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
