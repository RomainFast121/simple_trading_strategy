# Systematic Strategy Research

This public repository contains presentation material only. Strategy source
code, research notebooks, configurations, data, operational records, and the
underlying replication evidence are maintained privately.

## Public reports

### Ensemble

- [Stage 1](https://romainfast121.github.io/simple_trading_strategy/ensemble/presentation/stage_1_initial_review/outputs/stage_1_report.html)
- [Stage 2](https://romainfast121.github.io/simple_trading_strategy/ensemble/presentation/stage_2_nda_review/outputs/stage_2_report.html)
- [Stage 3](https://romainfast121.github.io/simple_trading_strategy/ensemble/presentation/stage_3_paid_pilot/outputs/stage_3_report.html?v=fe5fe64)

### Momentum

- [Stage 1](https://romainfast121.github.io/simple_trading_strategy/mom_crowding/presentation/stage_1_initial_review/outputs/stage_1_report.html)
- [Stage 3](https://romainfast121.github.io/simple_trading_strategy/mom_crowding/presentation/stage_3_paid_pilot/outputs/stage_3_report.html)

### Market neutral

- [Stage 1](https://romainfast121.github.io/simple_trading_strategy/market-neutral/presentation/stage_1_initial_review/outputs/stage_1_report.html)
- [Stage 3](https://romainfast121.github.io/simple_trading_strategy/market-neutral/presentation/stage_3_paid_pilot/outputs/stage_3_report.html)

### Opening-range breakout

- [Stage 1](https://romainfast121.github.io/simple_trading_strategy/ORB/presentation/stage_1_initial_review/outputs/stage_1_report.html)
- [Stage 3](https://romainfast121.github.io/simple_trading_strategy/ORB/presentation/stage_3_paid_pilot/outputs/stage_3_report.html?v=fe5fe64)

## What each stage shows

**Stage 1** introduces the strategy, its rationale, and its main development and
out-of-sample results. Each report shows the equity curve against a simple
market benchmark, annualized return and volatility, Sharpe ratio, maximum
drawdown, positive-month frequency, and market correlation. Rolling Sharpe and
benchmark-regression diagnostics help show whether performance is reasonably
persistent and whether it comes from something beyond broad market exposure.
It is the quickest way to understand the idea and judge whether the evidence is
worth exploring further.

**Stage 2** follows the ensemble after launch. It compares the frozen model,
the return implied by the positions actually held, and the live account using
compact cumulative return, average daily return, drawdown, observation count,
and win-rate summaries. While the live sample is still short, it also places
the model result within the distribution of same-length windows from the
one-year out-of-sample period. Annualized return, Sharpe, and Calmar are added
only after enough daily observations exist to make them meaningful. The
underlying positions and daily operating files remain private.

**Stage 3** is the deeper due-diligence view. It focuses on the post-freeze
record rather than the development sample. Alongside return, Sharpe, drawdown,
Calmar, and market-correlation metrics, it examines the return distribution,
fixed-window consistency, the fraction of positive periods, fee sensitivity,
empirical VaR and CVaR, and trading capacity relative to market volume. This is
the report intended for a closer assessment of robustness, implementation risk,
and whether the strategy remains investable beyond its headline performance.
