# Systematic Strategy Research

This public repository contains presentation material only. Strategy source
code, research notebooks, configurations, data, operational records, and the
underlying replication evidence are maintained privately.

## Public reports

### Ensemble

- [Stage 1](https://romainfast121.github.io/simple_trading_strategy/ensemble/presentation/stage_1_initial_review/outputs/stage_1_report.html?v=4651c8b9b9b0)
- [Stage 2](https://romainfast121.github.io/simple_trading_strategy/ensemble/presentation/stage_2_nda_review/outputs/stage_2_report.html?v=4f2fe3b3907e)
- [Stage 3](https://romainfast121.github.io/simple_trading_strategy/ensemble/presentation/stage_3_paid_pilot/outputs/stage_3_report.html?v=323133f2984b)
- [Stage 4](https://romainfast121.github.io/simple_trading_strategy/ensemble/presentation/stage_4_capacity_analysis/outputs/stage_4_report.html?v=c4316307ad19)

### Momentum

- [Stage 1](https://romainfast121.github.io/simple_trading_strategy/mom_crowding/presentation/stage_1_initial_review/outputs/stage_1_report.html?v=46d1c4526b67)
- [Stage 3](https://romainfast121.github.io/simple_trading_strategy/mom_crowding/presentation/stage_3_paid_pilot/outputs/stage_3_report.html?v=be7a5fcd284c)

### Market neutral

- [Stage 1](https://romainfast121.github.io/simple_trading_strategy/market-neutral/presentation/stage_1_initial_review/outputs/stage_1_report.html?v=4a6e2be2f925)
- [Stage 3](https://romainfast121.github.io/simple_trading_strategy/market-neutral/presentation/stage_3_paid_pilot/outputs/stage_3_report.html?v=78e5e1542795)

### Opening-range breakout

- [Stage 1](https://romainfast121.github.io/simple_trading_strategy/ORB/presentation/stage_1_initial_review/outputs/stage_1_report.html)
- [Stage 3](https://romainfast121.github.io/simple_trading_strategy/ORB/presentation/stage_3_paid_pilot/outputs/stage_3_report.html?v=fe5fe64)

## What each stage shows

**Stage 1** introduces the strategy, its rationale, and its main development and
out-of-sample results. The reports show the equity curve against a simple
market benchmark, annualized return and volatility, Sharpe ratio, maximum
drawdown, positive-month frequency, and market correlation. The sleeve and
ensemble reviews use independently development-calibrated post-only execution
scenarios, with modeled fills, maker fees, and funding. Benchmark-regression
diagnostics help show whether performance comes from more than broad market exposure.
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
Earlier observations retain their original execution assumptions; the new
post-only rule applies only to subsequent model targets and recorded trades.

**Stage 3** is the deeper due-diligence view. It focuses on the post-freeze
record rather than the development sample. Alongside return, Sharpe, drawdown,
Calmar, and market-correlation metrics, it examines the return distribution,
fixed-window consistency, and empirical VaR and CVaR. Execution and funding
diagnostics accompany the maker-fill scenarios; the ensemble also includes
fee sensitivity under immediate fills, with 5 bps as the taker reference, and
capacity relative to market volume. This is
the report intended for a closer assessment of robustness, implementation risk,
and whether the strategy remains investable beyond its headline performance.

**Stage 4** explores the ensemble's capacity at institutional size. Starting
with a compounded 10M reference account, it compares gradual execution over
12, 18, and 24 hours and periodic withdrawals of excess capital. The simulation
includes transaction fees, funding, and a research-based estimate of market
impact. Alongside annual return, withdrawals, Sharpe, drawdown, and rolling
consistency, it reports participation in hourly volume and estimated execution
costs. This helps show how slower trading and account growth affect performance
at scale; it retains market-order transaction costs rather than assuming
institutional post-only fills. Its 5 bps transaction charge is the explicit
taker/direct-fill assumption because maker fill conditions are difficult to
justify at that size. It is a modeled capacity study, not a guarantee
of executable returns.
