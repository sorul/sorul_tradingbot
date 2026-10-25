# Add PivotTrend break-even protection

## Objective
Move an open PivotTrend position's stop to its entry price after a completed M5 close reaches +1 initial ATR in the trade's favour.

## Why
Completed-close excursion analysis found that about half of eventual losing trades reached +1 ATR in every observed year (2022–2025), making this a causal, single-threshold exit-management hypothesis.

## Scope
- `sorul_tradingbot/strategy/private/pivot_trend.py`
- Relevant focused tests
- This task document

## Constraints
- Preserve the user's restored baseline by keeping the rejected alignment-entry filter absent.
- Trigger only after a completed close, never current open-only bar high/low.
- Measure +1 ATR using the ATR at entry, retained per ticket.
- Move the stop only in the favourable direction, to entry price; existing pivot trailing remains able to tighten it further.
- Do not stage or commit the user-owned modified notebook.

## Tasks
- [x] T1 — Implement per-order entry-ATR tracking and causal break-even stop modification. Route: delegated.
- [x] T2 — Add focused tests for long/short activation, no early activation, and no use of the current open-only candle. Route: delegated.
- [x] T3 — Run focused checks and record observed evidence. Route: delegated.

## Acceptance criteria
- An eligible BUY/SELL moves to break-even only after a prior completed close has travelled +1 entry ATR in its direction.
- The current open-only candle cannot activate break-even.
- No rejected trend-alignment gate remains in the strategy.

## Delivery
- Forecast: under 400 authored lines.
- Strategy: ask-on-risk.
- Implementation: `22333c7 feat(strategy): add pivot trend break-even`.
- Corrections: `0de96be fix(strategy): prune pending break-even atr state` and
  `77aab14 fix(strategy): retain cross-symbol pending atr state`.

## Progress
- `poetry run pytest tests/test_pivot_trend.py -q` — 8 passed in 0.62s during final independent verification.
- The user-owned `pivot_trend.ipynb` remains unmodified by this task.
- Follow-up correction: prune ATR records keyed by magic when their order is no
  longer present in the strategy's open-order view. This preserves delayed fills
  that remain visible as open orders while removing rejected/absent records.
- Cross-symbol correction: pending ATR pruning gathers this strategy's order
  magics across every symbol, so processing one symbol does not erase another
  still-visible pending order.
- Follow-up correction: determine pending-order magic retention across every
  symbol owned by this strategy, so processing one symbol cannot erase another
  symbol's delayed-fill ATR record.
