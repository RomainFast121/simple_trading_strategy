# Strategies

This repository is organized so each strategy can live in its own package while common backtest, import, and analysis logic stays in a shared utility module.

Current files:
- `momentum/`: momentum strategy package
- `xgb/`: compact cross-sectional XGBoost walk-forward strategy package
- `ensemble/`: 50/50 assembling strategy built from existing strategy output files
- `utils.py`: shared import, panel-building, performance, plotting, and Monte Carlo helpers
- `requirements.txt`: Python dependencies for the project

## Momentum Strategy

The current strategy is a moving-average momentum strategy with volatility targeting.

It can work with:
- Yahoo Finance symbols through `ticker`
- Binance symbols through `crypto`
- one single sleeve
- a basket mixing stocks and crypto

At least one of `ticker` or `crypto` must be provided. If both are empty, the class raises:
- `select at least one ticker`

### Inputs

The `MomentumStrategy` constructor takes the data-loading inputs:

- `ticker`: one Yahoo ticker or a list of Yahoo tickers
- `crypto`: one Binance symbol or a list of Binance symbols
- `start`: start date for the data
- `end`: end date for the data
- `tf`: logical timeframe used by the strategy, such as `1d`
- `hour`: optional hour used when building one observation per day from hourly bars
- `hour_timezone`: timezone used for that hour selection

The strategy parameters are passed later when calling `run(...)` or `run_monte_carlo(...)`:

- `bias`: if `False`, the signal is `+1 / -1`; if `True`, short signals become `0`
- `MA`: moving-average window length
- `fees`: transaction cost per unit of turnover
- `target_vol`: target annualized volatility used in the position scaling
- `vol_window`: rolling window used for the recent volatility estimate
- `init_amount`: starting wealth used for the wealth curve

Binance symbols must use the exchange format expected by `ccxt`.

This split is intentional:
- data settings are defined once
- raw data can be fetched once and reused
- strategy settings can be changed repeatedly without downloading data again

## File Structure

### `utils.py`

This file contains reusable functions that can be shared by future strategies.

Main functions:

- `normalize_symbol_input(...)`
  Turns a symbol input into a clean list.

- `fetch_data(...)`
  Fetches one native dataframe per sleeve.
  Yahoo symbols are downloaded through `yfinance`.
  Crypto symbols are downloaded through Binance with `ccxt`.

- `fetch_yahoo_symbol(...)`
  Downloads one Yahoo symbol and normalizes the OHLCV columns.

- `fetch_binance_symbol(...)`
  Downloads one Binance symbol with chunked `fetch_ohlcv(...)` calls so longer histories can be assembled from the exchange limit.

- `build_daily_snapshot_from_hourly(...)`
  Selects one hourly bar per local day, using the bar that starts at the chosen hour.

- `log_return(close)`
  Converts a close price series into log returns.

- `extract_close_map(...)`
  Extracts one close series per sleeve from raw strategy inputs.

- `build_close_panel(...)`
  Converts fetched close series into one stacked panel with `timestamp`, `trade_date`, `SYMBOL`, and `close`.

- `add_close_return_targets(...)`
  Builds same-symbol close-to-close returns and next-bar targets from the close panel.

- `build_xgb_features(...)`
  Creates the close-based lag, rolling, volatility, z-score, cross-sectional, and optional intraday timing features used by the XGB strategy.

- `make_walkforward_splits(...)`
  Creates expanding or rolling walk-forward splits on unique trade dates.

- `tune_xgb_model(...)`
  Runs the XGBoost tuning search on the train and tune splits, maximizing filtered sign accuracy at the configured metric filter quantile rather than R².

- `apply_xgb_cross_sectional_positions(...)`
  Converts XGB predictions into thresholded, volatility-scaled cross-sectional positions.

- `estimate_periods_per_year(index)`
  Estimates the natural annualization frequency from timestamp spacing.

- `rolling_annualized_vol(log_returns, window, min_periods)`
  Computes annualized rolling volatility from log returns.

- `calculate_performance(...)`
  Takes a sleeve return series plus a sleeve position series and computes sleeve-level turnover, fees, net returns, wealth, drawdown, and summary metrics.

- `calculate_buy_and_hold_baseline(...)`
  Builds a historical equal-weight, rebalanced B&H benchmark on the strategy evaluation calendar, including initial allocation and rebalance fees.

- `combine_sleeve_frames(...)`
  Merges already-built sleeves on the union of timestamps and creates the total portfolio path.

- `generate_monte_carlo_paths(...)`
  Builds synthetic close paths from rolling mean and volatility estimates taken from historical log returns, then applies empirical standardized shocks.

- `calculate_monte_carlo_performance(...)`
  Re-runs the evaluator on each synthetic path and aggregates the results.

- `plot_wealth(...)`
  Plots the strategy wealth curve together with an optional buy-and-hold benchmark.

- `plot_monte_carlo_wealth(...)`
  Plots all Monte Carlo wealth paths, their mean path, a confidence envelope, and an optional historical buy-and-hold benchmark.

### `momentum/`

This package contains only the momentum-specific parts while keeping the public import simple:

```python
from momentum import MomentumStrategy
```

Main methods:

- `fetch_data()`
  Downloads and stores one native raw dataframe per sleeve.

- `_build_single_ticker_frame(close)`
  Builds the momentum-specific columns for one sleeve:
  - close
  - simple return
  - log return
  - moving average
  - signal
  - recent volatility
  - raw position

- `_evaluate_single_ticker(close, ticker_name)`
  Computes the full sleeve performance on the sleeve's native calendar.

- `_evaluate_multi_ticker(close_map)`
  Builds every sleeve independently, then merges them into one portfolio only after the sleeve-local work is complete.

- `run()`
  Runs the historical backtest using the strategy parameters passed to that call and adds `B&H_yearly_factor` and `B&H_max_drawdown` to the summary.

- `run_monte_carlo(...)`
  Runs Monte Carlo on synthetic paths using the strategy parameters passed to that call, with an optional rolling estimation window, and appends the historical `B&H_yearly_factor` and `B&H_max_drawdown` to the Monte Carlo summary.

- `plot_wealth()`
  Plots the real wealth curve with the historical B&H benchmark and drawdown rectangles for both curves.

- `plot_monte_carlo()`
  Plots the Monte Carlo wealth spread with the historical B&H benchmark.

### `xgb/`

This package contains the XGBoost walk-forward strategy while keeping the public import simple:

```python
from xgb import XGBStrategy
```

Main methods:

- `fetch_data(force=False, save=True)`
  Downloads Yahoo/Binance data through the same import layer as the momentum strategy, stores it on `s.raw_data`, and caches reusable raw and feature panels under `local_outputs/xgb/current/` by default. If assets, dates, timeframe, or hour filters change, the cache is rebuilt in the same folder instead of creating a new config-specific folder.

- `run_walkforward(...)`
  Loads cached data/features when possible, builds non-forward-looking close-return features, creates chronological train/tune/test splits, fits train-only scaling and optional train-only PCA, tunes a conservative `xgboost.XGBRegressor` search on the tune split, and saves reusable OOS predictions plus split metrics.

- `use_feature_groups(include=None, exclude=None)`
  Selects feature families for the next walk-forward run without rebuilding raw features. This is useful after ablation, for example `s.use_feature_groups(exclude=["relative_cross_section"])`.

- `run_threshold(...)`
  Reuses cached predictions, searches absolute-signal threshold quantiles and volatility lookbacks through expanding validation over prior walk-forward splits, builds volatility-scaled positions after thresholding, evaluates the OOS backtest, and adds `B&H_yearly_factor` and `B&H_max_drawdown` to the summary.

- `run(...)`
  Runs the full pipeline: data if needed, then walk-forward, then threshold/backtest.

- `plot_wealth()`
  Plots the real XGB wealth curve with the historical B&H benchmark.

- `make_diagnostics(save=True, show=False)`
  Builds compact ML-style diagnostics: B&H return histogram, prediction scatter with linear fit/R², XGB versus B&H equity comparison, summary comparison table, threshold schedule, and the candidate sweep. Saved figures live under `local_outputs/xgb/current/figures/`, and metadata is consolidated in `xgb_run_diagnostics.json`.

- `run_feature_ablation(...)`
  Runs leave-one-feature-family-out walk-forward diagnostics and stores the compact result on `s.feature_ablation`.

Short example:

```python
from xgb import XGBStrategy

s = XGBStrategy(**RUN_CONFIG)
s.fetch_data(force=False, save=True)
s.run_walkforward(**WALKFORWARD_CONFIG)
s.run_threshold(**BACKTEST_CONFIG)
print(s.summary)
s.make_diagnostics(save=True, show=True)
s.plot_wealth()
```

## Assembling Strategy

The assembling strategy combines existing strategy CSV outputs without re-running the underlying models:

```python
from ensemble import AssemblingStrategy

s = AssemblingStrategy(
    strategy_paths={
        "momentum": "momentum_result.csv",
        "xgb": "local_outputs/xgb/current/strategy_data.csv",
    },
)
s.run()
print(s.summary_table())
print(s.latest_positions())
s.plot_wealth()
```

It checks symbols, inferred timeframe, index overlap, and close-price agreement. If indexes differ, it raises a warning and uses only timestamps where every component strategy is present and active. Positions are combined first, then fees are charged from the net position change, so overlapping trades do not pay duplicated turnover. The summary and default wealth plot include each component strategy, the assembled strategy, and B&H on the same common window.

The default `combination_method="weighted_average"` uses equal weights unless custom `weights` are passed. Future combination methods can be added behind the same hook, or passed as a callable that receives `component_positions` and `weights`.

Local outputs and notebooks are intentionally ignored by Git:

- `local_outputs/`
- `local_notebooks/`
- `*.ipynb`

## Import Logic

### Yahoo sleeves

Symbols passed through `ticker` are fetched from Yahoo Finance.

### Crypto sleeves

Symbols passed through `crypto` are fetched from Binance through `ccxt`.

The Binance import follows a chunked OHLCV workflow:
- request data in repeated batches
- move `since` forward by one timeframe after each batch
- stop when the requested end date is reached
- drop duplicate timestamps
- keep the requested date range only

Each sleeve stays on its own native calendar after import.

## Daily Timing Logic

This point matters for execution assumptions.

When `tf='1d'` and `hour` is not provided:
- Yahoo sleeves use native daily Yahoo bars
- crypto sleeves use native daily Binance bars

When `tf='1d'` and `hour` is provided:
- Yahoo sleeves fetch hourly bars and select the bar that starts at the chosen local hour
- crypto sleeves do the same
- that selected hourly bar becomes the one daily observation used by the strategy

Date-style `end` inputs are treated as inclusive in this repository:
- the fetch layer tries to include bars from the requested end date as well
- this avoids dropping the current day just because the upstream data providers interpret `end` as exclusive

Example interpretation:
- if the chosen hour is `09:00`, the strategy uses the `09:00 -> 10:00` bar
- the signal is built from that completed bar
- because positions are shifted by one bar in the backtest, the strategy cannot trade on that same information bar
- the earliest tradeable point is from the next selected bar onward

If `hour_timezone` is an IANA timezone such as `Europe/Zurich`, daylight saving time is handled through that timezone conversion before the hour selection is applied.

## Strategy Logic

For each sleeve, the momentum logic is built in this order:

1. Fetch raw data on the sleeve's native calendar.
2. Compute simple returns and log returns from the close price.
3. Compute the moving average.
4. Build the signal:
   - if `close > moving average`, signal = `+1`
   - otherwise, signal = `-1`
5. If `bias=True`, replace `-1` with `0`.
6. Compute recent annualized volatility over the chosen rolling window.
7. Compute the raw position:
   `signal * target_vol / recent_vol`
8. Shift the position by one period when converting sleeve returns into strategy returns.
9. Deduct fees from sleeve turnover.
10. Compound the sleeve net returns into the sleeve wealth curve.

Here, `position` is the portfolio exposure applied to the asset return at each step.

## XGBoost Strategy

The second strategy is a cross-sectional XGBoost return-prediction strategy built on the same fetched close data interface as momentum.

The constructor takes the same data-loading inputs as momentum:
- `ticker`
- `crypto`
- `start`
- `end`
- `tf`
- `hour`
- `hour_timezone`

The main parameters are passed to `run(...)`:
- `fees`
- `target_vol`
- `vol_lookback`
- `vol_lookback_min`
- `vol_lookback_max`
- `vol_lookback_step`
- `leverage_cap`
- `threshold_quantile`
- `threshold_quantile_min`
- `threshold_quantile_max`
- `threshold_quantile_step`
- `init_amount`
- `train_size`
- `tune_size`
- `test_size`
- `step_size`
- `mode`
- `refit_on_train_plus_tune`
- `model_params`
- `tuning_grid`
- `max_trials`
- `random_state`
- `pca_enabled`
- `pca_n_components`
- `tuning_min_side_balance`
- `metric_filter_quantile`

The XGB grid, walk-forward windows, filtered-signal settings, and backtest parameters are intentionally notebook-owned. The public Python code expects these values to be passed explicitly.
The tuner selects parameters by filtered sign accuracy at `metric_filter_quantile`. R² is still reported as a diagnostic, but it does not choose the model. `tuning_min_side_balance` can be used as a guardrail against one-sided top-signal sets. When tune-split scores are effectively tied, the tuner chooses deterministically and reports the selected model's train-versus-tune filtered-sign-accuracy gap in `s.walkforward_metrics`.

For experiments, put private grids and thresholds at the top of `xgb_application.ipynb`. That notebook is ignored by Git, so working search spaces do not need to be committed.

The threshold grid is expressed as quantiles of absolute prediction size. Ties are resolved by stable ranking so constant predictions do not accidentally activate the full split. This is more stable than basis points when changing timeframe, asset universe, or model scale.
`leverage_cap` caps the absolute volatility-scaled position weight per asset before portfolio aggregation.

The XGB workflow is:

1. Fetch raw market data through Yahoo and Binance exactly like momentum.
2. Extract close prices and build one stacked panel by `timestamp` and `SYMBOL`.
3. Compute close-to-close returns and next-bar targets from `close`.
4. Create close-based lag, rolling, volatility, z-score, and cross-sectional features from past information only.
5. Run expanding walk-forward train / tune / test splits.
6. Scale features on the train split only, optionally fit PCA on the train split only, and tune XGBoost by filtered sign accuracy on train versus tune.
7. Refit the chosen model and predict on the out-of-sample test split.
8. Concatenate all out-of-sample predictions across splits.
9. Use expanding validation on prior splits to select `threshold_quantile` and `vol_lookback` from their search ranges.
10. Convert predictions into thresholded, volatility-scaled cross-sectional positions using the split-specific selected parameters.
11. Feed those positions into the same shared portfolio and performance layer used by momentum.

The XGB strategy is intentionally cross-sectional:
- predictions are compared across assets at each timestamp
- positions are built from those relative signals
- the final portfolio still follows the same active-asset weighting logic as momentum after the sleeve positions are known

For `threshold_quantile` and `vol_lookback`, the XGB strategy keeps:
- one default value used on the first walk-forward split
- one search range used on later splits through expanding validation

So the final run stores both:
- the selected parameter path by split in `s.parameter_schedule`
- the full candidate sweep in `s.parameter_sweep`
- a single JSON report in `local_outputs/xgb/current/xgb_run_diagnostics.json`

The XGB strategy does not include Monte Carlo in this repository.

## B&H Comparison

The B&H comparison is a simple equal-wealth benchmark. It does not use target volatility, rolling volatility windows, signals, thresholds, or model outputs.

For each evaluated period, the benchmark:

1. Uses the same evaluated close calendar as the strategy.
2. Allocates equal wealth to each asset.
3. Rebalances back to equal weights on each bar.
4. Charges fees on the initial allocation and on each rebalance turnover.

The benchmark is intentionally plain. If a scaled comparison is needed later, scale the final benchmark return stream separately rather than changing the B&H construction.

## Sleeve-First Construction Rule

This is a core design rule of the repository.

Each sleeve is fully built on its own native timestamps before any cross-source merge happens.

That means:
- moving averages are computed on the sleeve's real history only
- volatility is computed on the sleeve's real history only
- signals are computed on the sleeve's real history only
- raw positions are computed on the sleeve's real history only

No forward-fill is allowed before those sleeve-local steps are finished.

This avoids distorting:
- moving averages
- volatility estimates
- signal timing
- turnover

## Portfolio Merge Logic

Only after every sleeve has been fully constructed do the sleeves get merged together.

The merge works like this:

1. Build the union of all timestamps across all sleeves.
2. Reindex each sleeve on that union.
3. Forward-fill sleeve `close` and sleeve `position`.
4. Treat a sleeve as active when its carried position is non-zero.
5. Count the number of active sleeves at each timestamp.
6. Scale active sleeves by `1 / active_sleeves_t`.
7. Sum the weighted sleeve return contributions into one portfolio return stream.
8. Compute the final portfolio wealth, fees, drawdown, and summary from that total return stream.

Important consequence:
- if stock sleeves still carry non-zero positions through the weekend, they remain part of the active portfolio
- their carried prices do not move while the market is closed, so their weekend return stays zero
- crypto sleeves continue to move on weekend timestamps
- crypto sleeves do not absorb the stock capital just because stocks have no fresh bars

Forward-fill happens only at this portfolio stage.

## Performance Metrics

The summary includes:

- `yearly_factor`: geometric annual return factor implied by the full compounded wealth path
- `B&H_yearly_factor`: geometric annual return factor of the comparable B&H benchmark
- `total_fees`: total fee cost paid by the strategy
- `max_drawdown`: worst peak-to-trough decline of the wealth curve
- `B&H_max_drawdown`: worst peak-to-trough decline of the comparable B&H benchmark
- `B&H_sharpe_ratio_annualized`: comparable B&H Sharpe using arithmetic mean return divided by arithmetic return volatility, with both annualized
- `winrate`: share of profitable active periods
- `average_return_factor`: geometric average return factor during active periods
- `sharpe_ratio_annualized`: annualized Sharpe based on normal net returns, with flat periods counted as zero return

For a basket, these metrics are computed from the final total portfolio wealth path, not by averaging sleeve summary rows.

Interpretation of `yearly_factor`:
- if `yearly_factor = 1.10`, that means an average annual multiplication factor of `1.10`
- over `N` years, `init_amount * yearly_factor ** N` gives the ending wealth implied by that average geometric annual factor

## Monte Carlo Analysis

The Monte Carlo module does not use the single realized path directly.

Instead it:
- estimates rolling log-return mean and volatility from history
- rescales empirical standardized shocks with those time-varying parameters
- rebuilds synthetic close paths
- reruns the strategy on each path
- aggregates the resulting metrics

### Multi-Source Monte Carlo

For a basket:
- each sleeve gets its own synthetic path
- each sleeve is evaluated on its own synthetic native path first
- only after sleeve-local evaluation are the sleeves merged into a total portfolio
- the same post-construction merge rule is used as in the historical backtest

So Monte Carlo stays aligned with the real backtest architecture:
- sleeve first
- merge later

The Monte Carlo generator is built to stay closer to market structure than an iid daily shuffle:
- local drift is estimated from rolling historical log returns
- local volatility is estimated from rolling historical log returns
- shock shapes come from empirical standardized residuals rather than a fully synthetic Gaussian draw
- basket simulations preserve cross-asset shock structure by drawing residual rows jointly across sleeves

An optional `block_length` can be passed to `run_monte_carlo(...)`:
- smaller values react faster to local changes in drift and volatility
- larger values smooth the parameter estimates more strongly
- if `block_length` is not provided, the code chooses one automatically from the sample size

### Confidence Intervals

Monte Carlo summary metrics are reported with confidence intervals because the estimated average depends on how many paths are generated.

The interval reported in the summary is a confidence interval for the estimated mean metric across simulated paths.

## Installation

Create and activate a local environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## How To Run

Typical workflow:

1. Import `MomentumStrategy` or `XGBStrategy`.
2. Instantiate it with the data-source settings you want.
3. Call `fetch_data()` once if you want to reuse the same market data across many parameter sets.
4. Call `run(...)`, passing the strategy parameters there.
5. For momentum only, you can also call `run_monte_carlo(...)`.
6. Repeat `run(...)` with different strategy parameters without fetching again.
7. Read the output from:
   - `s.summary`
   - `s.data`
8. For momentum only, additional Monte Carlo outputs are:
   - `s.monte_carlo_summary`
   - `s.monte_carlo_path_summaries`
   - `s.monte_carlo_wealth`
9. Export any dataframe with pandas if needed, for example with `.to_csv(...)`.
10. Use `plot_wealth()` for both strategies, and `plot_monte_carlo()` for momentum only.

The README intentionally does not prescribe specific parameter values.

## Data Structure Note

When one sleeve is used:
- `s.data` is a standard dataframe for that sleeve

When several sleeves are used:
- `s.data` uses grouped columns
- each sleeve has its own block of columns
- there is also a `portfolio` block containing the total portfolio metrics

So each asset key gives one sleeve block, and `s.data['portfolio']` gives the total portfolio wealth, drawdown, fees, and returns.

## Future Strategies

This structure is meant to scale.

If a new strategy is added later, the idea is:
- keep shared logic in `utils.py`
- create one new folder per strategy
- keep strategy-specific signal logic in that strategy package
- keep shared backtest and plotting logic reusable
