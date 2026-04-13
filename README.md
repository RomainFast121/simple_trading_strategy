# Strategies

This repository is organized so each strategy can live in its own file while common backtest, import, and analysis logic stays in a shared utility module.

Current files:
- `momentum.py`: momentum strategy class and momentum-specific signal logic
- `utils.py`: shared import, performance, plotting, and Monte Carlo helpers
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

Binance symbols must use the exchange format expected by `ccxt`, for example:
- `BTC/USDT`
- `ETH/USDT`

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

- `estimate_periods_per_year(index)`
  Estimates the natural annualization frequency from timestamp spacing.

- `rolling_annualized_vol(log_returns, window, min_periods)`
  Computes annualized rolling volatility from log returns.

- `calculate_performance(...)`
  Takes a sleeve return series plus a sleeve position series and computes sleeve-level turnover, fees, net returns, wealth, drawdown, and summary metrics.

- `calculate_buy_and_hold_baseline(...)`
  Builds a historical equal-capital buy-and-hold benchmark on the same close inputs used by the strategy.

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

### `momentum.py`

This file contains only the momentum-specific parts.

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
  Runs the historical backtest using the strategy parameters passed to that call and adds buy-and-hold yearly factor and max drawdown to the summary.

- `run_monte_carlo(...)`
  Runs Monte Carlo on synthetic paths using the strategy parameters passed to that call, with an optional rolling estimation window, and appends the historical buy-and-hold yearly factor and max drawdown to the Monte Carlo summary.

- `plot_wealth()`
  Plots the real wealth curve with the historical buy-and-hold benchmark.

- `plot_monte_carlo()`
  Plots the Monte Carlo wealth spread with the historical buy-and-hold benchmark.

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
- `total_fees`: total fee cost paid by the strategy
- `max_drawdown`: worst peak-to-trough decline of the wealth curve
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

1. Import `MomentumStrategy`.
2. Instantiate it with the data-source settings you want.
3. Call `fetch_data()` once if you want to reuse the same market data across many parameter sets.
4. Call `run(...)` for the historical backtest or `run_monte_carlo(...)` for the Monte Carlo analysis, passing the strategy parameters there.
5. Repeat `run(...)` with different strategy parameters without fetching again.
6. Read the output from:
   - `s.summary`
   - `s.data`
   - `s.monte_carlo_summary`
   - `s.monte_carlo_path_summaries`
   - `s.monte_carlo_wealth`
7. Export any dataframe with pandas if needed, for example with `.to_csv(...)`.
8. Use `plot_wealth()` or `plot_monte_carlo()` for the visual outputs.

The README intentionally does not prescribe specific parameter values.

## Data Structure Note

When one sleeve is used:
- `s.data` is a standard dataframe for that sleeve

When several sleeves are used:
- `s.data` uses grouped columns
- each sleeve has its own block of columns
- there is also a `portfolio` block containing the total portfolio metrics

So for example:
- `s.data['AAPL']` gives one sleeve block
- `s.data['BTC/USDT']` gives one sleeve block
- `s.data['portfolio']` gives the total portfolio wealth, drawdown, fees, and returns

## Future Strategies

This structure is meant to scale.

If a new strategy is added later, the idea is:
- keep shared logic in `utils.py`
- create one new file per strategy
- keep strategy-specific signal logic in the strategy file
- keep shared backtest and Monte Carlo logic reusable
