# Strategies

## Momentum Strategy

This repository currently contains a momentum strategy implemented in [momentum.py](/Users/romain/Documents/Perso/projects/Tutorial/simple_strategy/momentum.py).

The goal of this strategy is to:
- download historical price data from Yahoo Finance
- generate a directional signal from a moving average
- scale the position with volatility so that recent high volatility reduces exposure
- include trading fees in the backtest
- report a compact performance summary

### Inputs

The `MomentumStrategy` class takes the following parameters:

- `ticker`: Yahoo Finance symbol to download, for example `SPY`
- `start`: start date for the historical data
- `end`: end date for the historical data
- `bias`: if `False`, the strategy can be long or short; if `True`, short signals are replaced by flat exposure
- `tf`: Yahoo Finance interval, for example `1d`
- `MA`: moving average window length
- `fees`: transaction cost applied per unit of turnover
- `leverage`: multiplier applied to the final position size

### Strategy Logic

The strategy works in this order:

1. Download market data from Yahoo Finance.
2. Compute simple returns and log returns from the close price.
3. Compute a moving average on the close price.
4. Create a signal:
   If `close > moving average`, signal = `+1`
   Otherwise, signal = `-1`
5. If `bias=True`, replace `-1` by `0` so the strategy becomes long-only.
6. Compute long-horizon annualized volatility using the last `365D` of log returns with a minimum of `200` observations.
7. Compute short-horizon annualized volatility using the last `30D` of log returns.
8. Compute the raw position size as:
   `signal * annual_vol / recent_vol`
9. Multiply the raw position by `leverage`.
10. Shift the position by one period before applying returns so the signal is not using future information.
11. Compute turnover from position changes and deduct fees from the gross strategy return.
12. Convert net simple returns to net log returns and compound them into the wealth curve.
13. Compute summary statistics such as total P&L, total fees, win rate, average return factor, and annualized Sharpe ratio.

### Function-by-Function Walkthrough

#### `__init__`

Stores the input parameters and creates two empty placeholders:

- `self.data`: the full backtest dataframe
- `self.summary`: the one-row summary dataframe

This is the main object state used by the rest of the class.

#### `fetch_data`

Downloads data from Yahoo Finance with `yfinance.download(...)`.

It also:
- adjusts prices automatically
- normalizes column names to lowercase
- converts the index to pandas datetime format

The result is stored in `self.data`.

#### `_periods_per_year`

Estimates how many observations there are in one year based on the spacing between timestamps.

This matters because annualization should adapt to the actual frequency of the data rather than assume a fixed number like `252`.

For example:
- daily data will lead to a value close to the number of daily observations per year
- intraday data would produce a much larger value

#### `_rolling_annualized_vol`

Computes rolling annualized volatility from log returns using a time-based window such as `365D` or `30D`.

It does two things:
- computes the rolling standard deviation of log returns
- estimates how many observations per year are implied by the timestamps in that rolling window

The annualized volatility is then:

`rolling_std * sqrt(periods_per_year)`

This is done dynamically so the scaling remains coherent even if the data frequency changes.

#### `prepare_data`

Builds the main strategy dataframe.

It creates:
- `return`: simple asset return
- `log_return`: log asset return
- `ma`: moving average of the close price
- `signal`: `+1` or `-1`, then optionally `0` instead of `-1` if `bias=True`
- `annual_vol`: annualized volatility over the last `365D`
- `recent_vol`: annualized volatility over the last `30D`
- `position`: final scaled exposure before execution

The position formula is:

`signal * annual_vol / recent_vol`

This means:
- if recent volatility is high relative to long-term volatility, position size gets smaller
- if recent volatility is low relative to long-term volatility, position size gets larger

The idea is to keep risk exposure more stable through time.

#### `_compute_strategy`

Transforms the prepared signals and positions into a backtest result.

Main steps:
- shift the position by one bar with `position_prev`
- compute the asset simple return from the log return
- compute gross strategy return as:
  `position_prev * asset_simple_return`
- compute turnover as the absolute change in position
- compute fee cost as:
  `turnover * fees`
- compute net strategy return after fees
- convert net simple return into net log return with `log1p`
- compound the net log returns into `wealth`

It also computes the summary table.

Summary metrics:
- `total_pnl`: final compounded performance, equal to final wealth minus one
- `total_fees`: sum of all fee costs
- `max_drawdown`: worst peak-to-trough decline observed on the compounded wealth curve
- `winrate`: proportion of profitable active periods
- `average_return_factor`: average multiplicative return factor during active periods only
- `sharpe_ratio_annualized`: annualized Sharpe ratio based on net simple returns, with non-trading periods counted as zero return

Important implementation detail:
- `average_return_factor` is computed only on periods where the strategy actually had a non-zero position
- `sharpe_ratio_annualized` is computed on the full return series, with flat periods counted as zero return, so annualization stays standard without overstating sparse strategies

#### `run`

Simple wrapper that runs the full pipeline.

In practice, this is the main method to call:

```python
s.run()
```

#### `plot_wealth`

Plots the compounded wealth curve using seaborn styling and a logarithmic y-axis.

The log scale is useful because it makes compounded growth and drawdowns easier to compare visually across time.

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

Check the import:

```bash
python -c "from momentum import MomentumStrategy; print('import ok')"
```

Run the summary:

```bash
python -c "from momentum import MomentumStrategy; s = MomentumStrategy(ticker='SPY', start='2020-01-01', end='2025-01-01', bias=True, tf='1d', MA=50, fees=0.0005, leverage=3.0); s.run(); print(s.summary.to_string(index=False))"
```

Inspect the last rows of the backtest dataframe:

```bash
python -c "from momentum import MomentumStrategy; s = MomentumStrategy(ticker='SPY', start='2020-01-01', end='2025-01-01', bias=True, tf='1d', MA=50, fees=0.0005, leverage=3.0); s.run(); print(s.data[['close', 'signal', 'annual_vol', 'recent_vol', 'position', 'net_strategy_return', 'wealth']].tail(10).to_string())"
```

Generate and save the chart:

```bash
python -c "from momentum import MomentumStrategy; import matplotlib.pyplot as plt; s = MomentumStrategy(ticker='SPY', start='2020-01-01', end='2025-01-01', bias=True, tf='1d', MA=50, fees=0.0005, leverage=3.0); s.run(); s.plot_wealth(); plt.savefig('wealth.png'); print('saved wealth.png')"
```

### Outputs

After running the strategy:

- `self.data` contains the full bar-by-bar dataset
- `self.summary` contains the one-row performance summary

### Notes for Future Strategies

This README is organized so more strategies can be added later as separate sections, for example:

- `## Mean Reversion Strategy`
- `## Breakout Strategy`
- `## Pairs Trading Strategy`

That way the repository can keep one document with multiple strategy descriptions, each with:
- objective
- inputs
- logic
- function-by-function explanation
- usage example
