# Strategies

This repository is organized so each strategy can keep its own file while sharing common tools through a central utility module.

Current files:
- [momentum.py](momentum.py): momentum strategy class and strategy-specific logic
- [utils.py](utils.py): shared functions for data loading, returns, performance, plotting, and Monte Carlo analysis
- [requirements.txt](requirements.txt): Python dependencies for the project

## Momentum Strategy

The current strategy is a moving-average momentum strategy with target-volatility scaling.

It works for:
- one single ticker such as `SPY`
- several tickers such as `['SPY', 'QQQ', 'IWM']`

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
- `target_vol`: target annualized volatility for the position sizing, for example `0.10` for 10%

## File Structure

### `utils.py`

This file contains reusable functions that can be shared by future strategies.

Main functions:

- `fetch_data(...)`
  Downloads raw Yahoo Finance data and normalizes the columns for one ticker or several tickers.

- `log_return(close)`
  Converts a close price series into log returns.

- `estimate_periods_per_year(index)`
  Estimates the natural annualization frequency from timestamp spacing.

- `rolling_annualized_vol(log_returns, window, min_periods)`
  Computes annualized rolling volatility on a time window such as `365D` or `30D`.

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

- `summarize_returns(strategy_returns, fee_cost, ...)`
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
6. Compute recent annualized volatility over the last `30D`.
7. Compute the raw position:
   `signal * target_vol / recent_vol`
8. This means the strategy tries to scale each sleeve so its annualized volatility is closer to the chosen target.
9. Shift the position by one period when computing strategy returns.
10. Deduct fees from turnover.
11. Compound net returns into the wealth curve.

If several tickers are used:

1. The exact same logic is run independently for each ticker.
2. Each ticker sleeve produces its own net strategy return series.
3. The portfolio return is the equal-weight average of these ticker net return series.
4. The portfolio summary is then computed from this total portfolio return stream.

## Performance Metrics

The summary includes:

- `total_pnl`: final compounded wealth minus one
- `total_fees`: total fee cost paid by the strategy
- `max_drawdown`: worst peak-to-trough decline of the wealth curve
- `winrate`: share of profitable active periods
- `average_return_factor`: geometric average return factor during active periods
- `sharpe_ratio_annualized`: annualized Sharpe based on normal net returns, with flat periods counted as zero return

For a basket of tickers, these metrics are computed from the total equal-weight portfolio wealth and return stream, not by averaging each ticker summary row separately.

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

For a basket such as `['SPY', 'QQQ', 'IWM']`:
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
- `total_pnl` is the average total P&L across the simulated paths
- `total_pnl_ci_lower` and `total_pnl_ci_upper` give a 95% confidence interval for that average estimate

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

Check imports:

```bash
python -c "from momentum import MomentumStrategy; from utils import fetch_data, calculate_performance, log_return; print('imports ok')"
```

Run the real backtest on one ticker:

```bash
python -c "from momentum import MomentumStrategy; s = MomentumStrategy(ticker='SPY', start='2020-01-01', end='2025-01-01', bias=True, tf='1d', MA=200, fees=0.0005, target_vol=0.10); s.run(); print(s.summary)"
```

Run the real backtest on several tickers:

```bash
python -c "from momentum import MomentumStrategy; s = MomentumStrategy(ticker=['SPY', 'QQQ', 'IWM'], start='2020-01-01', end='2025-01-01', bias=True, tf='1d', MA=200, fees=0.0005, target_vol=0.10); s.run(); print(s.summary)"
```

Run the Monte Carlo summary on one ticker:

```bash
python -c "from momentum import MomentumStrategy; s = MomentumStrategy(ticker='SPY', start='2020-01-01', end='2025-01-01', bias=True, tf='1d', MA=200, fees=0.0005, target_vol=0.10); s.run_monte_carlo(n_paths=250, seed=42); print(s.monte_carlo_summary)"
```

Run the Monte Carlo summary on several tickers:

```bash
python -c "from momentum import MomentumStrategy; s = MomentumStrategy(ticker=['SPY', 'QQQ', 'IWM'], start='2020-01-01', end='2025-01-01', bias=True, tf='1d', MA=200, fees=0.0005, target_vol=0.10); s.run_monte_carlo(n_paths=250, seed=42); print(s.monte_carlo_summary)"
```

Save the Monte Carlo summary to CSV:

```bash
python -c "from momentum import MomentumStrategy; s = MomentumStrategy(ticker=['SPY', 'QQQ', 'IWM'], start='2020-01-01', end='2025-01-01', bias=True, tf='1d', MA=200, fees=0.0005, target_vol=0.10); s.run_monte_carlo(n_paths=250, seed=42); s.monte_carlo_summary.to_csv('monte_carlo_summary.csv', index=False); print('saved monte_carlo_summary.csv')"
```

Plot the real total wealth:

```bash
python -c "from momentum import MomentumStrategy; s = MomentumStrategy(ticker=['SPY', 'QQQ', 'IWM'], start='2020-01-01', end='2025-01-01', bias=True, tf='1d', MA=200, fees=0.0005, target_vol=0.10); s.run(); s.plot_wealth()"
```

Plot the Monte Carlo total wealth spread:

```bash
python -c "from momentum import MomentumStrategy; s = MomentumStrategy(ticker=['SPY', 'QQQ', 'IWM'], start='2020-01-01', end='2025-01-01', bias=True, tf='1d', MA=200, fees=0.0005, target_vol=0.10); s.run_monte_carlo(n_paths=250, seed=42); s.plot_monte_carlo()"
```

## Data Structure Note

When you use one ticker:
- `s.data` is a standard dataframe for that ticker

When you use several tickers:
- `s.data` uses grouped columns
- each ticker has its own block of columns
- there is also a `portfolio` block containing the total equal-weight portfolio metrics

So for example:
- `s.data['SPY']` gives the SPY sleeve details
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
