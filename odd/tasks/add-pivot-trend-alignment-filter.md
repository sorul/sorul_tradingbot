# Add PivotTrend alignment filter

## Objective
Require PivotTrend entries to match the prior four-hour M5 price direction.

## Why
Notebook analysis showed that retaining only aligned trades improved net points in every observed year and materially improved 2025 profit factor and drawdown.

## Scope
- `sorul_tradingbot/strategy/private/pivot_trend.py`
- `tests/test_pivot_trend.py`
- This task document

## Constraints
- BUY: prior completed close above the close 48 M5 bars earlier.
- SELL: prior completed close below the close 48 M5 bars earlier.
- Use only information available before the order; do not introduce look-ahead.
- Preserve current behavior otherwise.
- Do not stage or commit the user's modified `pivot_trend.ipynb`.

## Tasks
- [x] T1 — Add the causal alignment gate to PivotTrend entry decisions. Route: delegated.
- [x] T2 — Add focused tests proving aligned long/short pass and counter-trend long/short rejection. Route: delegated.
- [x] T3 — Run focused checks and record the observed result. Route: delegated.

## Acceptance criteria
- An aligned BUY/SELL remains eligible when all existing entry conditions pass.
- A counter-trend BUY/SELL is rejected before opening an order.
- The filter uses 48 completed M5 bars, matching notebook analysis.

## Delivery
- Forecast: under 400 authored lines.
- Strategy: ask-on-risk.
- Work-unit commit: `268438f feat(strategy): align pivot trend entries`.
- Rollback boundary: revert `268438f` to remove the alignment gate and its focused tests without affecting the notebook.

## Progress
- T1: `PivotTrend.indicator` compares `closes[-1]` (the last completed close) with `closes[-49]` (48 completed M5 bars earlier). The open-only current row remains excluded through `ohlc.close[:-1]`; warmup requires 49 closed bars plus the current row.
- T2: `tests/test_pivot_trend.py` verifies aligned BUY/SELL orders are emitted and counter-trend BUY/SELL breakouts return no order. The aligned cases set deliberately contradictory current-row close values, proving the filter does not inspect that unclosed value.
- T3 verification: `poetry run pytest tests/test_pivot_trend.py -q` — `4 passed in 0.77s`.
- Runtime harness: N/A. The indicator tests exercise the strategy boundary directly; a full simulator rerun is the user's next validation step.
- `pivot_trend.ipynb` remains user-owned and unstaged.
