# Strategies

This repository is organized so each strategy can keep its own file while sharing common tools through a central utility module.

Current files:
- [momentum.py](momentum.py): momentum strategy class and strategy-specific logic
- [utils.py](utils.py): shared functions for data loading, returns, performance, plotting, and Monte Carlo analysis
- [requirements.txt](requirements.txt): Python dependencies for the project

## Momentum Strategy

The current strategy is a moving-average momentum strategy with volatility scaling.

The goal is to:
- download historical price data from Yahoo Finance
- compute returns and a moving average
- generate a long or short signal from price versus moving average
- optionally force the short side to zero when `bias=True`
- scale exposure with long-term versus recent volatility
- include turnover-based fees
- report standard performance metrics
- compare the real path with Monte Carlo resampled paths

### Inputs

The `MomentumStrategy` class takes:

- `ticker`: Yahoo Finance symbol such as `SPY`
- `start`: start date for the data
- `end`: end date for the data
- `bias`: if `False`, the signal is `+1 / -1`; if `True`, short signals become `0`
- `tf`: Yahoo Finance interval such as `1d`
- `MA`: moving average window length
- `fees`: transaction cost per unit of turnover
- `leverage`: multiplier applied to the final position size

### File Structure

#### `utils.py`

This file contains reusable functions that can be shared by future strategies.

Main functions:

- `fetch_data(...)`
  Downloads raw Yahoo Finance data and normalizes the columns.

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

- `plot_wealth(wealth, ...)`
  Plots a standard wealth curve in seaborn style.

- `generate_monte_carlo_paths(close, n_paths, seed)`
  Builds synthetic close paths by bootstrap-resampling the historical simple returns and compounding them from the first observed close.

- `calculate_monte_carlo_performance(close, evaluator, ...)`
  Runs a strategy callback on every simulated path, stores the path-by-path results, and aggregates the statistics.

- `summarize_monte_carlo_results(...)`
  Computes the average metric across Monte Carlo paths and adds a 95% confidence interval for that estimated average.

- `plot_monte_carlo_wealth(wealth_paths, ...)`
  Plots all simulated wealth paths, the average path, and a 95% envelope so the spread is visible.

#### `momentum.py`

This file contains only the strategy-specific parts.

Main methods:

- `fetch_data()`
  Downloads and stores the raw market data for the selected ticker and dates.

- `_build_strategy_frame(close)`
  Builds the momentum-specific columns:
  - close
  - simple return
  - log return
  - moving average
  - signal
  - annual volatility
  - recent volatility
  - position size

- `_evaluate_close_series(close)`
  Calls `_build_strategy_frame(...)`, then delegates the performance calculation to `utils.calculate_performance(...)`.

- `run()`
  Runs the strategy on the real historical close prices.

- `run_monte_carlo(...)`
  Runs the strategy on many synthetic close paths generated from the historical return distribution.

- `plot_wealth()`
  Plots the real wealth curve.

- `plot_monte_carlo()`
  Plots the Monte Carlo wealth spread, mean path, and 95% envelope.

### Strategy Logic

The momentum strategy works in this order:

1. Download raw data from Yahoo Finance.
2. Compute simple returns and log returns from the close price.
3. Compute the moving average on the close price.
4. Build the signal:
   - if `close > moving average`, signal = `+1`
   - otherwise, signal = `-1`
5. If `bias=True`, replace `-1` by `0`.
6. Compute annualized volatility over the last `365D` with a minimum of `200` observations.
7. Compute recent annualized volatility over the last `30D`.
8. Compute the raw position:
   `signal * annual_vol / recent_vol`
9. Multiply the position by `leverage`.
10. Shift the position by one period when computing strategy returns.
11. Deduct fees from turnover.
12. Compound net returns into the final wealth curve.

### Performance Metrics

The summary includes:

- `total_pnl`: final compounded wealth minus one
- `total_fees`: total fee cost paid by the strategy
- `max_drawdown`: worst peak-to-trough decline of the wealth curve
- `winrate`: share of profitable active periods only
- `average_return_factor`: geometric average return factor during active periods only
- `sharpe_ratio_annualized`: annualized Sharpe based on normal net returns, with flat periods counted as zero return

Important detail:
- `average_return_factor` is active-period only
- `sharpe_ratio_annualized` uses the full net return series so non-trading bars remain part of the annualization

## Monte Carlo Analysis

The Monte Carlo module is meant to answer a different question from the historical backtest.

Instead of using the one realized close path, it:
- takes the historical simple return distribution
- resamples returns with replacement
- reconstructs many synthetic close paths
- reruns the strategy on every path
- measures the distribution of outcomes

### What The Monte Carlo Output Means

The Monte Carlo summary returns the same style of metrics as the standard backtest, but each metric is now:
- the average across all simulated paths
- plus a 95% confidence interval for that estimated average

This matters because the average from Monte Carlo depends on how many paths are generated.

For example:
- `total_pnl` is the average total P&L across the simulated paths
- `total_pnl_ci_lower` and `total_pnl_ci_upper` give a 95% confidence interval for that average estimate

The wealth plot for Monte Carlo shows:
- every individual simulated wealth path in the background
- the average wealth path
- a 95% envelope across paths to visualize the spread

### Installation

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

### How To Run

Check imports:

```bash
python -c "from momentum import MomentumStrategy; from utils import fetch_data, calculate_performance, log_return; print('imports ok')"
```

Run the real backtest summary:

```bash
python -c "from momentum import MomentumStrategy; s = MomentumStrategy(ticker='SPY', start='2020-01-01', end='2025-01-01', bias=True, tf='1d', MA=50, fees=0.0005, leverage=1.0); s.run(); print(s.summary.to_string(index=False))"
```

Inspect the last rows of the real backtest dataframe:

```bash
python -c "from momentum import MomentumStrategy; s = MomentumStrategy(ticker='SPY', start='2020-01-01', end='2025-01-01', bias=True, tf='1d', MA=50, fees=0.0005, leverage=1.0); s.run(); print(s.data[['close', 'signal', 'annual_vol', 'recent_vol', 'position', 'net_strategy_return', 'wealth']].tail(10).to_string())"
```

Run the Monte Carlo summary:

```bash
python -c "from momentum import MomentumStrategy; s = MomentumStrategy(ticker='SPY', start='2020-01-01', end='2025-01-01', bias=True, tf='1d', MA=50, fees=0.0005, leverage=1.0); s.run_monte_carlo(n_paths=250, seed=42); print(s.monte_carlo_summary.to_string(index=False))"
```

Plot and save the real wealth curve:

```bash
python -c "from momentum import MomentumStrategy; import matplotlib.pyplot as plt; s = MomentumStrategy(ticker='SPY', start='2020-01-01', end='2025-01-01', bias=True, tf='1d', MA=50, fees=0.0005, leverage=1.0); s.run(); s.plot_wealth(); plt.savefig('momentum_wealth.png'); print('saved momentum_wealth.png')"
```

Plot and save the Monte Carlo wealth spread:

```bash
python -c "from momentum import MomentumStrategy; import matplotlib.pyplot as plt; s = MomentumStrategy(ticker='SPY', start='2020-01-01', end='2025-01-01', bias=True, tf='1d', MA=50, fees=0.0005, leverage=1.0); s.run_monte_carlo(n_paths=250, seed=42); s.plot_monte_carlo(); plt.savefig('momentum_monte_carlo_wealth.png'); print('saved momentum_monte_carlo_wealth.png')"
```

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
