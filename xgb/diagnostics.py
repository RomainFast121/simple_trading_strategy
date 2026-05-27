from pathlib import Path
import os

import numpy as np
import pandas as pd


FEATURE_GROUPS = {
    "lagged_returns": [
        "lag_log_return_1",
        "lag_log_return_2",
        "lag_log_return_3",
        "lag_log_return_6",
        "lag_log_return_12",
        "lag_log_return_24",
    ],
    "momentum": ["cumret_3", "cumret_6", "cumret_12"],
    "volatility": [
        "rv_6",
        "rv_12",
        "rv_24",
        "mean_abs_return_6",
        "mean_abs_return_12",
        "mean_abs_return_24",
    ],
    "normalized_returns": ["zscore_return_12", "zscore_return_24"],
    "intraday_time": ["bar_of_day_sin", "bar_of_day_cos"],
    "market_context": ["lag_eq_market_return", "lag_cross_sectional_dispersion"],
    "relative_cross_section": ["stock_minus_market_lag", "lag_rank_pct"],
}


def clean_figure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for figure in path.glob("*.png"):
        figure.unlink()
    return path


def _prepare_matplotlib(path):
    cache_dir = Path(path).parent / ".mplconfig"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


def comparison_summary(strategy):
    rows = []
    if not strategy.summary.empty:
        row = strategy.summary.iloc[0]
        rows.append(
            {
                "strategy": "XGB",
                "yearly_factor": row.get("yearly_factor"),
                "sharpe_ratio_annualized": row.get("sharpe_ratio_annualized"),
                "max_drawdown": row.get("max_drawdown"),
                "turnover": row.get("turnover"),
                "total_fees": row.get("total_fees"),
                "selected_threshold_quantile": row.get("selected_threshold_quantile"),
                "selected_vol_lookback": row.get("selected_vol_lookback"),
            }
        )
    if isinstance(strategy.buy_and_hold_summary, pd.DataFrame) and not strategy.buy_and_hold_summary.empty:
        row = strategy.buy_and_hold_summary.iloc[0]
        rows.append(
            {
                "strategy": "B&H equal weight",
                "yearly_factor": row.get("yearly_factor"),
                "sharpe_ratio_annualized": row.get("sharpe_ratio_annualized"),
                "max_drawdown": row.get("max_drawdown"),
                "turnover": row.get("turnover"),
                "total_fees": row.get("total_fees"),
                "selected_threshold_quantile": np.nan,
                "selected_vol_lookback": strategy.vol_lookback,
            }
        )
    return pd.DataFrame(rows)


def _portfolio_series(frame, column):
    if frame.empty:
        return pd.Series(dtype=float)
    if isinstance(frame.columns, pd.MultiIndex):
        if ("portfolio", column) in frame.columns:
            return frame[("portfolio", column)].dropna()
        return pd.Series(dtype=float)
    if column in frame.columns:
        return frame[column].dropna()
    return pd.Series(dtype=float)


def _format_date_axis(ax):
    import matplotlib.dates as mdates

    locator = mdates.MonthLocator(interval=3)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")


def plot_return_histogram(frame, column, path, title, xlabel):
    _prepare_matplotlib(path)
    import matplotlib.pyplot as plt

    values = (
        pd.to_numeric(pd.Series(frame[column], copy=False), errors="coerce").dropna()
        if column in frame
        else pd.Series(dtype=float)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    if values.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.hist(values, bins=100)
        ax.text(
            1.0,
            1.0,
            f"mean = {values.mean():.6f}\nstd = {values.std(ddof=0):.6f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_prediction_scatter(frame, path, prediction_col="prediction", target_col="target_next_log_return"):
    _prepare_matplotlib(path)
    import matplotlib.pyplot as plt

    sample_source = frame[[prediction_col, target_col]].apply(pd.to_numeric, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(6, 6))
    if sample_source.empty or sample_source[prediction_col].nunique() < 2 or sample_source[target_col].nunique() < 2:
        ax.text(0.5, 0.5, "No regression fit available", ha="center", va="center", transform=ax.transAxes)
    else:
        sample = sample_source.sample(min(5000, len(sample_source)), random_state=42)
        ax.scatter(sample[prediction_col], sample[target_col], s=5, alpha=0.3)
        x = sample[prediction_col].to_numpy()
        y = sample[target_col].to_numpy()
        slope, intercept = np.polyfit(x, y, deg=1)
        fitted = slope * x + intercept
        ss_res = np.sum((y - fitted) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot
        x_grid = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_grid, slope * x_grid + intercept, color="red", linewidth=2, label=f"fit R2={r2:.4f}")
        ax.legend(loc="upper right")
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
    ax.axvline(0.0, color="black", linewidth=1, alpha=0.6)
    ax.set_title("Prediction vs realized next return")
    ax.set_xlabel("prediction")
    ax.set_ylabel("realized next log return")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_equity_comparison(strategy, path):
    _prepare_matplotlib(path)
    import matplotlib.pyplot as plt

    xgb_wealth = pd.to_numeric(_portfolio_series(strategy.data, "wealth"), errors="coerce").dropna()
    bh_wealth = pd.to_numeric(_portfolio_series(strategy.buy_and_hold_data, "wealth"), errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if xgb_wealth.empty and bh_wealth.empty:
        ax.text(0.5, 0.5, "No equity curves", ha="center", va="center", transform=ax.transAxes)
    if not xgb_wealth.empty:
        ax.plot(xgb_wealth.index, xgb_wealth / xgb_wealth.iloc[0], label="XGB")
    if not bh_wealth.empty:
        ax.plot(bh_wealth.index, bh_wealth / bh_wealth.iloc[0], label="B&H equal weight")
    ax.set_title("Equity curves")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("normalized wealth")
    ax.legend(loc="upper left")
    _format_date_axis(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_combined_diagnostics(strategy, path):
    _prepare_matplotlib(path)
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    returns = pd.to_numeric(
        _portfolio_series(strategy.buy_and_hold_data, "net_strategy_return"),
        errors="coerce",
    ).dropna()
    ax = axes[0, 0]
    if returns.empty:
        ax.text(0.5, 0.5, "No B&H returns", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.hist(returns, bins=100, color="steelblue", alpha=0.85)
        ax.text(
            1.0,
            1.0,
            f"mean = {returns.mean():.6f}\nstd = {returns.std(ddof=0):.6f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )
    ax.set_title("B&H equal-weight return distribution")
    ax.set_xlabel("net return")

    ax = axes[0, 1]
    sample_source = strategy.predictions[["prediction", "target_next_log_return"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()
    if sample_source.empty or sample_source["prediction"].nunique() < 2 or sample_source["target_next_log_return"].nunique() < 2:
        ax.text(0.5, 0.5, "No regression fit available", ha="center", va="center", transform=ax.transAxes)
    else:
        sample = sample_source.sample(min(5000, len(sample_source)), random_state=42)
        ax.scatter(sample["prediction"], sample["target_next_log_return"], s=5, alpha=0.3)
        x = sample["prediction"].to_numpy(dtype=float)
        y = sample["target_next_log_return"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, deg=1)
        fitted = slope * x + intercept
        ss_res = np.sum((y - fitted) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot
        x_grid = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_grid, slope * x_grid + intercept, color="red", linewidth=2, label=f"fit R2={r2:.4f}")
        ax.legend(loc="upper right")
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
    ax.axvline(0.0, color="black", linewidth=1, alpha=0.6)
    ax.set_title("Prediction vs realized next return")
    ax.set_xlabel("prediction")
    ax.set_ylabel("next log return")

    ax = axes[1, 0]
    xgb_wealth = pd.to_numeric(_portfolio_series(strategy.data, "wealth"), errors="coerce").dropna()
    bh_wealth = pd.to_numeric(_portfolio_series(strategy.buy_and_hold_data, "wealth"), errors="coerce").dropna()
    if xgb_wealth.empty and bh_wealth.empty:
        ax.text(0.5, 0.5, "No equity curves", ha="center", va="center", transform=ax.transAxes)
    else:
        if not xgb_wealth.empty and not bh_wealth.empty:
            end = min(xgb_wealth.index.max(), bh_wealth.index.max())
            xgb_wealth = xgb_wealth.loc[:end]
            bh_wealth = bh_wealth.loc[:end]
        if not xgb_wealth.empty:
            ax.plot(xgb_wealth.index, xgb_wealth / xgb_wealth.iloc[0], label="XGB")
        if not bh_wealth.empty:
            ax.plot(bh_wealth.index, bh_wealth / bh_wealth.iloc[0], label="B&H equal weight")
        ax.legend(loc="upper left")
    ax.set_title("Equity curves")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("normalized wealth")
    _format_date_axis(ax)

    ax = axes[1, 1]
    schedule = strategy.parameter_schedule.copy()
    if schedule.empty:
        ax.text(0.5, 0.5, "No threshold schedule", ha="center", va="center", transform=ax.transAxes)
    else:
        x = schedule["split_id"].astype(str)
        ax.plot(x, schedule["applied_threshold_quantile"], marker="o", label="threshold quantile")
        ax.set_ylabel("threshold quantile")
        ax2 = ax.twinx()
        ax2.plot(x, schedule["applied_vol_lookback"], marker="s", color="darkorange", label="vol lookback")
        ax2.set_ylabel("vol lookback bars")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper left")
    ax.set_title("Expanding validation parameter path")
    ax.set_xlabel("applied split")

    fig.suptitle("XGB strategy diagnostics", fontsize=14)
    fig.savefig(path)
    plt.close(fig)


def build_diagnostics(strategy, save=True, show=False):
    figures = {}
    figure_dir = Path(strategy.output_dir) / "figures"
    if save:
        clean_figure_dir(figure_dir)

    if save:
        paths = {"diagnostics": figure_dir / "xgb_diagnostics.png"}
        plot_combined_diagnostics(strategy, paths["diagnostics"])
        figures = {name: str(path) for name, path in paths.items() if path.exists()}

    report = {
        "comparison": comparison_summary(strategy),
        "walkforward_metrics": strategy.walkforward_metrics.copy(),
        "parameter_schedule": strategy.parameter_schedule.copy(),
        "split_performance": strategy.split_performance.copy(),
        "parameter_sweep": strategy.parameter_sweep.copy(),
        "figures": figures,
    }
    strategy.diagnostics_report = report

    if show:
        try:
            from IPython.display import Image, display

            display(report["comparison"])
            display(report["parameter_schedule"])
            display(report["split_performance"])
            for path in figures.values():
                display(Image(filename=path))
        except Exception:
            pass
    return report
