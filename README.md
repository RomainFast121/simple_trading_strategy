# Systematic Digital-Asset Strategy Research

This repository contains a systematic research and paper-tracking codebase for a multi-sleeve digital-asset strategy. It is designed around a simple institutional requirement: once a research configuration is frozen, future performance must be measurable without changing the strategy logic, parameters, data alignment, or historical result presentation.

The current research freeze is dated **1 June 2026**. The research configuration was optimized on **26 May 2026**, then frozen on 1 June 2026 so the Git timestamp, configuration fingerprints, and documented results align cleanly. This is not presented as live proof yet. It is the historical baseline that should remain fixed while a future paper-trading period is evaluated after the freeze.

The README is intentionally written as a central review document. It explains the system, the organization of the codebase, the evaluation logic, and the frozen headline metrics, while avoiding disclosure of the tradable universe, exact evaluation ranges, implementation-sensitive thresholds, or detailed parameterization. The only disclosed parameter is the portfolio risk target used to distinguish the two risk profiles below.

## Executive Summary

The project combines three independent strategy families into an ensemble:

- **Momentum**: a systematic trend-following sleeve.
- **XGB**: a walk-forward supervised-learning sleeve.
- **Mean reversion**: a contrarian sleeve based on short-term overextension.

The ensemble evaluates these sleeves both individually and in combined portfolio form. The objective is not to rely on one signal, but to combine return streams with different behavior and low component correlation.

The codebase supports:

- reproducible research runs;
- saved local strategy outputs;
- portfolio-level position assembly;
- fee-aware net performance;
- drawdown and recovery analysis;
- component correlation reporting;
- restart-date reporting for future paper/live monitoring.

## Freeze And Forward Test Protocol

The commercial logic is the following:

1. Freeze the research configuration and code on **1 June 2026**.
2. Record the historical research baseline in this document.
3. Do not change parameters, strategy logic, or alignment rules after the freeze.
4. Continue producing positions and performance using the same code path.
5. Later, use **2 June 2026** as the restart date in the ensemble notebook and report the post-freeze period separately.

This creates a clean separation between:

- **Research baseline**: what the frozen configuration achieved historically.
- **Forward paper period**: what the same configuration delivered after the freeze.

The forward period should be reported separately. It should not replace, blend into, or overwrite the frozen research table below.

For paper-tracking reports, `METRICS_RESTART_DATE` should be set to **2026-06-02**. This starts reporting from the first bar after the 1 June 2026 freeze and keeps the frozen research snapshot separate from subsequent observations.

## Frozen Research Snapshots

The tables below record two frozen risk profiles visible in the ensemble notebook at the research freeze. Results are net of modeled fees.

The first profile is the lower-risk research configuration. The volatility-targeted ensemble uses a **30% portfolio target volatility**, while the other ensemble variants use a **40% portfolio target volatility**. This is the more conservative presentation for allocators who prefer lower drawdown and lower turnover, while keeping the same underlying sleeve logic.

| Strategy | Yearly factor | Sharpe | Win rate | Max drawdown | Max DD duration days | Max DD recovery days | 10% DD count | 10% DD frequency days | Avg 10% DD duration days | Avg 10% DD recovery days | Total fees | Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Momentum | 1.4102 | 1.0173 | 0.4970 | -0.2353 | 111.9583 |  | 5 | 102.1917 | 46.4000 | 32.5000 | 0.0451 | 0.1821 |
| XGB | 1.4909 | 1.1589 | 0.3128 | -0.2874 | 90.0000 | 142.0417 | 4 | 127.7396 | 55.2396 | 65.0000 | 0.1153 | 0.4658 |
| Mean reversion | 1.3028 | 1.1966 | 0.2938 | -0.1093 | 113.0417 | 43.0000 | 2 | 255.4792 | 68.5208 | 25.0000 | 0.0628 | 0.2538 |
| Ensemble mean | 2.9986 | 2.3504 | 0.5304 | -0.2260 | 63.0000 |  | 6 | 85.1597 | 27.0000 | 23.6083 | 0.1741 | 0.7036 |
| Ensemble volatility target | 2.0969 | 1.8555 | 0.5324 | -0.2122 | 23.0000 | 48.0000 | 3 | 170.3194 | 22.3194 | 20.3333 | 0.1703 | 0.6879 |
| Ensemble rolling Sharpe | 3.3409 | 2.2397 | 0.5445 | -0.2712 | 30.0000 |  | 9 | 56.7731 | 21.1065 | 17.6250 | 0.1879 | 0.7594 |
| Buy and hold | 0.7159 | -0.1882 | 0.5000 | -0.5683 | 140.0417 |  | 5 | 102.1917 | 47.4000 | 32.5000 | 0.0042 | 0.0168 |

The second profile is the higher-growth research configuration. It uses a **50% portfolio target volatility** for the ensemble variants. This version accepts a larger drawdown budget in exchange for stronger historical compounding.

| Strategy | Yearly factor | Sharpe | Win rate | Max drawdown | Max DD duration days | Max DD recovery days | 10% DD count | 10% DD frequency days | Avg 10% DD duration days | Avg 10% DD recovery days | Total fees | Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Momentum | 1.4102 | 1.0173 | 0.4970 | -0.2353 | 111.9583 |  | 5 | 102.1917 | 46.4000 | 32.5000 | 0.0451 | 0.1821 |
| XGB | 1.4909 | 1.1589 | 0.3128 | -0.2874 | 90.0000 | 142.0417 | 4 | 127.7396 | 55.2396 | 65.0000 | 0.1153 | 0.4658 |
| Mean reversion | 1.3028 | 1.1966 | 0.2938 | -0.1093 | 113.0417 | 43.0000 | 2 | 255.4792 | 68.5208 | 25.0000 | 0.0628 | 0.2538 |
| Ensemble mean | 3.7087 | 2.3330 | 0.5304 | -0.2779 | 63.0000 |  | 7 | 72.9940 | 24.8571 | 21.0069 | 0.2167 | 0.8755 |
| Ensemble volatility target | 2.8142 | 1.8767 | 0.5324 | -0.2878 | 72.0000 | 48.0000 | 9 | 56.7731 | 23.2222 | 13.2500 | 0.2619 | 1.0581 |
| Ensemble rolling Sharpe | 4.0656 | 2.2501 | 0.5445 | -0.3000 | 68.9583 |  | 9 | 56.7731 | 22.7731 | 17.0000 | 0.2302 | 0.9302 |
| Buy and hold | 0.7159 | -0.1882 | 0.5000 | -0.5683 | 140.0417 |  | 5 | 102.1917 | 47.4000 | 32.5000 | 0.0042 | 0.0168 |

The point of presenting both profiles is not to retrofit the research after the fact. It is to show that the same underlying architecture can be packaged at different risk budgets before the forward paper period begins.

## Configuration Fingerprints

The exact strategy configurations are intentionally not published in this repository. Instead, the public repository records cryptographic fingerprints of the two private freeze JSON files.

The hashes are computed from canonical JSON with sorted keys and compact separators, using `scripts/hash_freeze_config.py`. This makes the hash independent of whitespace and key order. It still changes if any economically relevant frozen configuration value changes.

| Profile | Private config file | SHA-256 |
|---|---|---|
| Lower risk | `private_freeze_configs/2026-06-01_lower_risk_config.json` | `3d63a413a4e570a3b288cef8c29a433f7e0ddd8235267a89a46e3cd6b184e8b2` |
| High growth | `private_freeze_configs/2026-06-01_high_growth_config.json` | `e375ee3cda633426ba04538116d849c66c13ca72a52473b85a1bebd72c2bd422` |

The restart date used for paper-tracking metrics is deliberately excluded from these hashes. So are future paper-tracking end dates, output paths, diagnostics flags, notebook execution state, notebook outputs, and local cached data. This means that the forward paper period can be extended later without changing the frozen research fingerprint.

To verify a private config later:

```bash
python3 scripts/hash_freeze_config.py private_freeze_configs/2026-06-01_lower_risk_config.json
python3 scripts/hash_freeze_config.py private_freeze_configs/2026-06-01_high_growth_config.json
```

If the same private JSON content is used later, the hash will match exactly. If any frozen strategy parameter, asset universe, research freeze date, sampling setting, model setting, or ensemble risk-budget setting changes, the hash will change.

## Component Return Correlation

The matrix below is the Pearson correlation of aligned net component strategy returns at the frozen snapshot. It applies to both risk profiles because it is measured at the component-return level, before the ensemble risk budget is selected.

|  | Momentum | XGB | Mean reversion |
|---|---:|---:|---:|
| Momentum | 1.0000 | -0.0289 | -0.2481 |
| XGB | -0.0289 | 1.0000 | -0.1503 |
| Mean reversion | -0.2481 | -0.1503 | 1.0000 |

This matters commercially because the ensemble result should not be judged only on standalone sleeve performance. Low or negative correlation between sleeves is one of the main reasons the combined portfolio can improve risk-adjusted behavior.

## Strategy Architecture

The system is organized as a small research stack rather than a single notebook.

**Momentum**

The momentum sleeve captures directional persistence. It converts trend information into signed exposure, applies risk scaling, and produces a saved strategy output with positions, returns, fees, wealth, and drawdown statistics.

**XGB**

The machine-learning sleeve uses a chronological walk-forward process. Training, validation, model selection, and out-of-sample evaluation are separated in time. Transformations are fit only on the relevant training segment, then applied forward.

**Mean Reversion**

The mean-reversion sleeve is a contrarian strategy. It looks for short-term stretched moves and converts them into exposure in the opposite direction, with risk controls and fee-aware performance reporting.

**Ensemble**

The ensemble loads the component outputs, aligns them on a shared evaluation index, combines positions into portfolio-level exposure, charges fees on actual exposure changes, and reports both component and ensemble performance.

The current ensemble views are:

- **Ensemble mean**: equal combination of component positions.
- **Ensemble volatility target**: portfolio exposure adjusted for realized risk.
- **Ensemble rolling Sharpe**: component allocation based on recent realized sleeve performance.

## Metric Definitions

All strategy metrics are computed from net strategy returns after modeled trading fees.

**Yearly factor**

The yearly factor is geometric, not an arithmetic average of period returns. It is computed from final wealth:

```text
yearly_factor = (final_wealth / initial_wealth) ^ (1 / elapsed_years)
```

A yearly factor of `1.50` means the wealth curve compounded at roughly `+50%` per year over the evaluated period.

**Annualized Sharpe**

The Sharpe ratio is calculated from simple net strategy returns:

```text
annualized_mean = mean(period_returns) * periods_per_year
annualized_volatility = std(period_returns) * sqrt(periods_per_year)
sharpe = annualized_mean / annualized_volatility
```

No risk-free rate is subtracted.

**Win rate**

Win rate is the share of active periods with a positive net strategy return. Periods with no active exposure are not treated as trading wins.

**Max drawdown**

Max drawdown is the worst percentage decline of the wealth curve from its running peak.

**Max drawdown duration**

This is the elapsed time from the previous wealth peak to the maximum drawdown trough.

**Max drawdown recovery**

This is the elapsed time from the maximum drawdown trough until the wealth curve recovers the previous peak. If the peak is not recovered by the end of the sample, the field is left blank.

**10% drawdown count**

This counts complete or still-open drawdown episodes where the wealth curve falls at least 10% from its prior peak.

**10% drawdown frequency**

This is the total elapsed evaluation time divided by the number of 10% drawdown episodes. It is an average spacing measure, not the amount of time spent in drawdown.

**Average 10% drawdown duration**

This is the average time from peak to trough across 10% drawdown episodes.

**Average 10% drawdown recovery**

This is the average time from trough to recovery across recovered 10% drawdown episodes. Unrecovered episodes are not included in the recovery average.

**Total fees**

Total fees are the cumulative modeled fee drag over the evaluated period, expressed in return units before multiplying by initial capital.

**Turnover**

Turnover is the average absolute change in portfolio exposure per period. For combined portfolios, it is calculated on the actual portfolio-level exposure change, not by naively adding unweighted sleeve turnover.

## Alignment And Leakage Controls

The codebase is structured to reduce accidental lookahead bias:

- Strategy returns use previous positions, so exposure decided at one period is applied to the next period's return.
- Latest positions are labeled as instructions for the next period.
- Machine-learning training, validation, and test segments are chronological.
- Data transformations are fit on training data only, then applied forward.
- Model and rule selection are based on information available before the evaluated out-of-sample segment.
- Ensemble metrics can be restarted from a future date without changing historical positions or recomputing research-period results.
- Fees are charged on changes in actual exposure.

These controls are necessary but not sufficient to prove robustness. The purpose of the freeze protocol is to make the next step testable: after time has passed, the same configuration can be evaluated on data that was not available when the freeze was made.

## Codebase Organization

```text
momentum/          Momentum sleeve implementation
xgb/               Walk-forward machine-learning sleeve
mean_reversion/    Mean-reversion sleeve implementation
ensemble/          Portfolio assembly, comparison, restart metrics
utils.py           Shared performance, drawdown, plotting, and backtest utilities
requirements.txt   Python dependencies
```

Strategy notebooks sit at the project root and act as the visible research control layer. They define experiment configuration, run the sleeves, and display the summary tables. Reusable logic lives in the Python packages above.

Generated outputs are kept outside Git under a local output structure:

```text
local_outputs/momentum/current/
local_outputs/xgb/current/
local_outputs/mean_reversion/current/
local_outputs/ensemble/current/
```

This avoids versioning large generated files while keeping reruns predictable.

## Audit Trail

The intended evidence chain is:

1. Commit the code and README at the freeze point.
2. Keep the frozen research configuration unchanged.
3. Later, rerun the same notebooks and confirm that the frozen table can still be reproduced.
4. Set the paper-tracking restart date to **2026-06-02** and report post-freeze metrics separately.

The Git commit timestamp provides the external time marker for the freeze. The post-freeze results should be evaluated from data that was not available when that commit was made.

## Confidentiality Boundary

This public-facing README deliberately avoids disclosing:

- the exact traded instruments;
- exact parameter values, except the disclosed portfolio risk targets used to distinguish the two frozen profiles;
- exact research or evaluation date ranges;
- exact signal thresholds;
- detailed feature recipes;
- any venue, execution, or production deployment assumptions.

Those details can be reviewed separately under the appropriate confidentiality framework. The purpose of this document is to explain what the system is, how it is evaluated, and how the frozen result can be audited later.
