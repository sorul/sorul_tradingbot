# Clean PivotTrend notebook

## Objective
Make the PivotTrend analysis notebook's execution section self-contained, ordered, and easy to re-run for a different payload.

## Problem
The four current analysis cells depend on notebook state, use inconsistent metric names, and mix data preparation with reports.

## Scope
- `sorul_tradingbot/strategy/private/pivot_trend.ipynb`
- This task document

## Constraints
- Keep P&L expressed as raw `pnl_points`; do not perform FX or EUR conversion.
- Use `print(df.to_markdown(index=False))` for tabular outputs.
- Preserve causal features: every feature is shifted to completed bars before entry.

## Tasks
- [x] T1 — Consolidate the four execution cells into clearly headed, self-contained preparation and reporting cells. Route: delegated (notebook preparation and edit).
- [x] T2 — Verify the notebook executes the cleaned sequence against the existing 2022–2024 payload and produces markdown tables. Route: delegated verification.

## Acceptance criteria
- A user can change one payload path, execute the cells top-to-bottom, and get the regime, alignment, and baseline comparison tables.
- No cell depends on an accidental column such as `net_points` being present on individual trades.

## Verification
- Run the execution cells programmatically where practical, or report an exact limitation.

## Progress
- T1 complete: cells 4–7 now use one configurable `payload_path`, raw `pnl_points`, causal shifted features, and explicit step headings. Every DataFrame report uses `print(df.to_markdown(index=False))`.
- T2 complete: `poetry run python /tmp/verify_pivot_notebook.py` executed cells 4–7 against `pivot_trend_2026-09-21_22-02`; it printed the regime, alignment, and baseline comparison tables and ended with `NOTEBOOK_EXECUTION_OK`.
- JSON validation: `python -c` loaded `pivot_trend.ipynb` successfully.

## Delivery
- Forecast: under 400 authored lines.
- Strategy: ask-on-risk.
- Work unit: notebook cleanup and reproducible execution validation.
- Verification: `poetry run python /tmp/verify_pivot_notebook.py` — passed (`NOTEBOOK_EXECUTION_OK`).
- Runtime harness: N/A; this notebook is an offline analysis artifact.
- Rollback boundary: revert `sorul_tradingbot/strategy/private/pivot_trend.ipynb` and this task document only.
