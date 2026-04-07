# Strategies

This repository is organized so each strategy can keep its own file while sharing common tools through a central utility module.

Current files:
- [momentum.py](momentum.py): momentum strategy class and strategy-specific logic
- [utils.py](utils.py): shared functions for data loading, returns, performance, plotting, and Monte Carlo analysis
- [requirements.txt](requirements.txt): Python dependencies for the project

## Momentum Strategy

The current strategy is a moving-average momentum strategy with target-volatility scaling.

It works for:
- one single ticker
- several tickers

When several tickers are used:
- the same parameters are applied to every ticker
- each ticker gets weight `1 / number_of_tickers`
- the final summary is computed from the total equal-weight portfolio wealth

### Inputs

The `MomentumStrategy` class takes:

- `ticker`: one ticker string or a list of tickers
- `start`: start date for the data
- `end`: end date for the data
- `bias`: if `False`, the signal is `+1 / -1`; if `True`, short signals become `0`
- `tf`: Yahoo Finance interval such as `1d`
- `MA`: moving average window length
- `fees`: transaction cost per unit of turnover
- `target_vol`: target annualized volatility for the position sizing
- `vol_window`: rolling window used for the recent volatility estimate
- `init_amount`: starting wealth used for the wealth curve
- `hour`: optional hour used only when you want a daily series built from hourly bars
- `hour_timezone`: timezone used for that hour selection, for example `UTC`, `UTC+1`, or an IANA timezone name; defaults to `UTC`

The constructor is intentionally explicit now.
There are no strategy defaults in the class signature for these tuning inputs, so each run has to state its choices directly.

## File Structure

### `utils.py`

This file contains reusable functions that can be shared by future strategies.

Main functions:

- `fetch_data(...)`
  Downloads raw Yahoo Finance data and normalizes the columns for one ticker or several tickers.
  If `tf='1d'` and no hour is provided, it uses Yahoo daily bars.
  If `tf='1d'` and an hour is provided, it builds a daily series from hourly bars using the bar that starts at the chosen hour in the chosen timezone.

- `build_daily_snapshot_from_hourly(data, hour, hour_timezone)`
  Selects one hourly bar per local day, using the bar that starts at the chosen hour.

- `log_return(close)`
  Converts a close price series into log returns.

- `estimate_periods_per_year(index)`
  Estimates the natural annualization frequency from timestamp spacing.

- `rolling_annualized_vol(log_returns, window, min_periods)`
  Computes annualized rolling volatility on a rolling time window.

- `calculate_performance(returns, positions, fees, log_return)`
  Takes an asset return series plus a position series and computes:
  - turnover
  - gross strategy return
  - fee cost
  - net strategy return
  - wealth
  - running peak
  - drawdown
  - summary metrics

- `summarize_returns(init_amount, strategy_returns, fee_cost, ...)`
  Turns a net strategy return stream into wealth, drawdown, and summary metrics.
  This is especially useful when combining several ticker sleeves into one total portfolio return stream.

- `plot_wealth(wealth, ...)`
  Plots a standard wealth curve in seaborn style.

- `generate_monte_carlo_paths(close, n_paths, seed)`
  Builds synthetic close paths by bootstrap-resampling historical simple returns.
  It works with one ticker or several tickers.

- `calculate_monte_carlo_performance(close, evaluator, ...)`
  Runs a strategy callback on every simulated path, stores the path-by-path results, and aggregates the statistics.

- `summarize_monte_carlo_results(...)`
  Computes the average metric across Monte Carlo paths and adds a 95% confidence interval for that estimated average.

- `plot_monte_carlo_wealth(wealth_paths, ...)`
  Plots all simulated wealth paths, the average path, and a 95% envelope so the spread is visible.

### `momentum.py`

This file contains only the strategy-specific parts.

Main methods:

- `fetch_data()`
  Downloads and stores the raw market data.
  If `tf='1d'` and no hour is provided, it uses Yahoo daily data.
  If `tf='1d'` and an hour is provided, it builds the daily observation series from hourly market bars through the constructor settings.

- `_build_single_ticker_frame(close)`
  Builds the momentum-specific columns for one ticker:
  - close
  - simple return
  - log return
  - moving average
  - signal
  - recent volatility
  - position size

- `_evaluate_single_ticker(close, ticker_name)`
  Computes the full performance of one ticker sleeve using `utils.calculate_performance(...)`.

- `_evaluate_multi_ticker(close_frame)`
  Combines several ticker sleeves into one equal-weight portfolio and computes the summary from total wealth.

- `run()`
  Runs the strategy on the real historical close prices.

- `run_monte_carlo(...)`
  Runs the strategy on many synthetic close paths generated from the historical return distribution.

- `plot_wealth()`
  Plots the real total wealth curve.

- `plot_monte_carlo()`
  Plots the Monte Carlo wealth spread, mean path, and 95% envelope.

## Strategy Logic

For each ticker, the momentum strategy works in this order:

1. Download raw data from Yahoo Finance.
2. Compute simple returns and log returns from the close price.
3. Compute the moving average on the close price.
4. Build the signal:
   - if `close > moving average`, signal = `+1`
   - otherwise, signal = `-1`
5. If `bias=True`, replace `-1` by `0`.
6. Compute recent annualized volatility over the chosen rolling window.
7. Compute the raw position:
   `signal * target_vol / recent_vol`
8. This means the strategy tries to scale each sleeve so its annualized volatility is closer to the chosen target.
9. Shift the position by one period when computing strategy returns.
10. Deduct fees from turnover.
11. Compound net returns into the wealth curve.

## Daily Data Timing

This point is important.

When `tf='1d'` and no `hour` is provided:
- the strategy uses Yahoo daily bars
- this is daily bar data from Yahoo

When `tf='1d'` and an `hour` is provided:
- the strategy fetches hourly bars
- it selects the bar that starts at the chosen `hour` in the chosen `hour_timezone`
- that selected hourly bar becomes the daily observation used by the strategy

So in practice:
- if `hour=15`, the strategy uses the hourly bar from `15:00` to `16:00`
- the signal is built from the values of that bar
- because the position is shifted by one bar, the strategy can only trade based on that signal from the next bar onward

So a `15:00` selection means:
- the `15:00 -> 16:00` bar is the information bar
- the strategy is not allowed to trade on that same bar
- the earliest tradeable moment is from the following selected bar onward

So `tf='1d'` can behave in two ways:
- standard daily mode if no hour is given
- one-observation-per-day hourly-snapshot mode if an hour is given

If several tickers are used:

1. The exact same logic is run independently for each ticker.
2. Each ticker sleeve produces its own net strategy return series.
3. The portfolio return is the equal-weight average of these ticker net return series.
4. The portfolio summary is then computed from this total portfolio return stream.

## Performance Metrics

The summary includes:

- `yearly_factor`: geometric annual return factor implied by the full compounded wealth path
- `total_fees`: total fee cost paid by the strategy
- `max_drawdown`: worst peak-to-trough decline of the wealth curve
- `winrate`: share of profitable active periods
- `average_return_factor`: geometric average return factor during active periods
- `sharpe_ratio_annualized`: annualized Sharpe based on normal net returns, with flat periods counted as zero return

For a basket of tickers, these metrics are computed from the total equal-weight portfolio wealth and return stream, not by averaging each ticker summary row separately.

Interpretation of `yearly_factor`:
- if `yearly_factor = 1.10`, that means an average annual growth factor of 10%
- over `N` years, `init_amount * yearly_factor ** N` gives the ending wealth implied by that average geometric annual rate

## Monte Carlo Analysis

The Monte Carlo module answers a different question from the historical backtest.

Instead of using the one realized close path, it:
- takes the historical simple return distribution
- resamples returns with replacement
- reconstructs synthetic close paths
- reruns the strategy on every path
- measures the distribution of outcomes

### Multi-Ticker Monte Carlo

This part is important when several tickers are used.

For a basket of several tickers:
- each ticker gets its own resampled synthetic return path
- one Monte Carlo path means one synthetic path for every ticker in the basket
- the strategy is rerun on the full basket path
- the equal-weight portfolio return is then recomputed from those synthetic ticker sleeves

So Monte Carlo remains coherent at the total portfolio level, not just ticker by ticker in isolation.

### What The Monte Carlo Output Means

The Monte Carlo summary returns the same style of metrics as the standard backtest, but each metric is now:
- the average across all simulated paths
- plus a 95% confidence interval for that estimated average

This matters because the average from Monte Carlo depends on how many paths are generated.

For example:
- `yearly_factor` is the average annual return factor across the simulated paths
- `yearly_factor_ci_lower` and `yearly_factor_ci_upper` give a 95% confidence interval for that average estimate

The Monte Carlo wealth plot shows:
- every individual simulated wealth path in the background
- the average wealth path
- a 95% envelope across paths to visualize the spread

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

1. Import `MomentumStrategy`.
2. Instantiate it by explicitly passing all required parameters.
3. Call `run()` for the historical backtest or `run_monte_carlo(...)` for the Monte Carlo analysis.
4. Read the output from:
   - `s.summary`
   - `s.data`
   - `s.monte_carlo_summary`
   - `s.monte_carlo_path_summaries`
   - `s.monte_carlo_wealth`
5. Export any dataframe with pandas if needed, for example with `.to_csv(...)`.
6. Use `plot_wealth()` or `plot_monte_carlo()` for the visual outputs.

The README intentionally does not prescribe specific parameter values.
The class interface is explicit, so the user chooses every setting directly when creating the strategy object.

## Data Structure Note

When you use one ticker:
- `s.data` is a standard dataframe for that ticker

When you use several tickers:
- `s.data` uses grouped columns
- each ticker has its own block of columns
- there is also a `portfolio` block containing the total equal-weight portfolio metrics

So for example:
- `s.data['TICKER_NAME']` gives one ticker sleeve details
- `s.data['portfolio']` gives the total portfolio wealth, drawdown, fees, and returns

## Future Strategies

This structure is meant to scale.

If a new strategy is added later, the idea is:
- keep shared logic in `utils.py`
- create one new file per strategy
- keep strategy-specific signal logic in the strategy file
- keep shared backtest and Monte Carlo logic reusable

Typical future additions could be:
- `mean_reversion.py`
- `breakout.py`
- `pairs_trading.py`
