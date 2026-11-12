# Systematic Strategy Research

This public repository contains presentation material only. Strategy source
code, research notebooks, configurations, data, operational records, and the
underlying replication evidence are maintained privately.

## Main changes

- 17 August 2026: Revised the position-cap logic.
- 2 October 2026: Switched to post-only orders and updated the transaction-fee assumption.
- 13 October 2026: Added a control layer to the sleeves before ensemble aggregation.
- 19 October 2026: Added noise detection for market neutral.
- 3 November 2026: Changed the momentum reference.

## Public ensemble reports

- [Stage 1](https://romainfast121.github.io/simple_trading_strategy/ensemble/presentation/stage_1_initial_review/outputs/stage_1_report.html?v=7573d8b42fc2)
- [Stage 2](https://romainfast121.github.io/simple_trading_strategy/ensemble/presentation/stage_2_nda_review/outputs/stage_2_report.html?v=a24d15fd1a14)
- [Stage 3](https://romainfast121.github.io/simple_trading_strategy/ensemble/presentation/stage_3_paid_pilot/outputs/stage_3_report.html?v=6043dfca5f4f)
- [Stage 4](https://romainfast121.github.io/simple_trading_strategy/ensemble/presentation/stage_4_capacity_analysis/outputs/stage_4_report.html?v=384e29af43dc)

Assumed fees:

- Transaction fees: 2 bp maker order fee in Stages 1 to 3. The simulation checks whether the price crosses the order; if it does, it assumes a full fill. Stage 4 assumes 5 bp taker fees.
- Funding fees: 1 bp applied to the amount held during the day.

## What each stage shows

**Stage 1:** In-sample results compared with the one-year holdout period (out of sample, OOS). Includes equity curves, rolling 90-day Sharpe ratios, regressions against buy-and-hold and each sleeve, and core metrics.

**Stage 2:** Model and live-account comparison since live execution began on 3 August 2026.

**Stage 3:** Similar to Stage 1, but covers OOS onward and is updated daily.

**Stage 4:** Models price impact using an assumed formula and tests its effect on a 10 million USDT account.
