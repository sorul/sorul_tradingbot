from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from tradeo.ohlc import OHLC

from sorul_tradingbot.strategy.private.pivot_trend import PivotTrend


def _ohlc(*, closed_start, closed_penultimate, closed_last, entry, current_close):
  """Build 49 closed M5 bars followed by the current open-only bar."""
  closes = np.linspace(closed_start, closed_penultimate, 49)
  closes[-1] = closed_last
  all_closes = np.append(closes, current_close)
  opens = np.append(closes, entry)
  index = pd.date_range('2025-01-02 09:00', periods=50, freq='5min', tz='UTC')
  frame = pd.DataFrame(
      {
          'open': opens,
          'high': np.maximum(opens, all_closes) + 2.0,
          'low': np.minimum(opens, all_closes) - 2.0,
          'close': all_closes,
          'volume': 100,
      },
      index=index,
  )
  return OHLC(frame)


def _strategy_with_pivots(maxima, minima):
  strategy = PivotTrend(SimpleNamespace(open_orders=[]))
  strategy._pivots = lambda _prices, kind: maxima if kind == 'max' else minima
  return strategy


@pytest.mark.parametrize(
    ('side', 'closed_start', 'closed_penultimate', 'closed_last', 'entry',
     'current_close', 'maxima', 'minima'),
    [
        (
            'BUY', 100.0, 105.0, 111.0, 110.0, 1.0,
            [(108.0, 40), (106.0, 20)], [(95.0, 35), (94.0, 15)],
        ),
        (
            'SELL', 110.0, 105.0, 99.0, 100.0, 1_000.0,
            [(108.0, 40), (110.0, 20)], [(101.0, 35), (103.0, 15)],
        ),
    ],
)
def test_indicator_allows_breakout_aligned_with_prior_trend(
    side, closed_start, closed_penultimate, closed_last, entry, current_close,
    maxima, minima,
):
  ohlc = _ohlc(
      closed_start=closed_start,
      closed_penultimate=closed_penultimate,
      closed_last=closed_last,
      entry=entry,
      current_close=current_close,
  )
  strategy = _strategy_with_pivots(maxima, minima)

  order = strategy.indicator(ohlc, 'SP500', ohlc.datetime[-1])

  assert order is not None
  assert order.order_type.buy is (side == 'BUY')


@pytest.mark.parametrize(
    ('closed_start', 'closed_penultimate', 'closed_last', 'entry', 'maxima',
     'minima'),
    [
        (
            120.0, 105.0, 111.0, 110.0,
            [(108.0, 40), (106.0, 20)], [(95.0, 35), (94.0, 15)],
        ),
        (
            90.0, 105.0, 99.0, 100.0,
            [(108.0, 40), (110.0, 20)], [(101.0, 35), (103.0, 15)],
        ),
    ],
)
def test_indicator_rejects_breakout_against_prior_trend(
    closed_start, closed_penultimate, closed_last, entry, maxima, minima,
):
  ohlc = _ohlc(
      closed_start=closed_start,
      closed_penultimate=closed_penultimate,
      closed_last=closed_last,
      entry=entry,
      current_close=500.0,
  )
  strategy = _strategy_with_pivots(maxima, minima)

  order = strategy.indicator(ohlc, 'SP500', ohlc.datetime[-1])

  assert order is None
