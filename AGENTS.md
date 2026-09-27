# AGENTS.md

## Project

`sorul_tradingbot` is a Python 3.11 / Poetry Forex bot built on the local
editable `../tradeo` dependency.  The live entry point is
`sorul_tradingbot/main_forex.py`; `ForexExecutable` orchestrates MetaTrader 5,
and strategies live under `sorul_tradingbot/strategy/private/`.  The simulator
is under `sorul_tradingbot/strategy/simulator/`.

## Safety boundaries

- Treat changes to live execution, order management, strategies, and MT5
  infrastructure as financial-safety work. Do not start Docker, MT5, or
  `make run_forex` unless the user explicitly authorizes it.
- Never read, print, commit, or modify `.env` contents. Do not alter
  `metatrader/`, simulator data/outputs, or logs without explicit approval.
- Preserve unrelated Git state. In particular, do not restore or stage the
  existing deletion of `sorul_tradingbot/strategy/private/pivot_trend.ipynb`,
  and leave untracked `.gga` untouched.
- Do not run `make tag`, push, create tags, regenerate dependency lockfiles, or
  change the local `../tradeo` dependency without explicit authorization.

## Trading invariants

- Keep strategy ownership isolated by symbol and `strategy_name`/order comment.
  A mismatch in `strategy_factory.py` can close open orders, so retain backward
  compatibility when changing strategy names or factory resolution.
- Preserve causal simulation: indicators may use closed candles; on the active
  candle the simulator exposes only its open. Do not use its high, low, or close
  to decide an entry.
- Stops must be finite, positive, and on the protective side of the entry.
  Trailing stops may tighten but never loosen. Maintain conservative intrabar
  handling: when both stop and target can occur in one candle, resolve to stop;
  gaps exit at the executable opening price.
- Keep the stale-history and duplicate-entry fail-safes in live execution unless
  a focused test demonstrates the replacement behaviour.

## Development and verification

- Use Poetry and Python 3.11. Preserve the repository style: two-space
  indentation and 80-character lines (`config/tox.ini`).
- For Python changes, run the applicable checks: `make flake8` and `make test`.
  Report any check that is skipped or cannot run.
- For strategy changes affecting entry, stop loss, take profit, trailing,
  break-even, gaps, or order management, add deterministic tests. Cover BUY and
  SELL symmetry where applicable, using synthetic OHLC data or
  `SimulatedMTClient`; do not treat a backtest result as proof of live safety.
- Keep simulator changes causal and add focused coverage in
  `tests/test_simulator.py`. Strategy tests may live alongside their private
  strategy as `test_*.py`.
- Do not introduce dependencies or broad refactors without explaining the
  trade-off and obtaining approval.

## Repository workflow

- Keep changes narrow and avoid touching generated, private, or ignored assets
  unless they are explicitly in scope. Ignored paths may still contain important
  local state.
- Before committing, confirm the diff excludes unrelated work. Use Conventional
  Commit messages and never add AI attribution.
