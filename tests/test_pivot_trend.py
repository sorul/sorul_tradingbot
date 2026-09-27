from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from tradeo.ohlc import OHLC

from sorul_tradingbot.strategy.private.pivot_trend import PivotTrend
from sorul_tradingbot.strategy.simulator.simulator import SimulatedMTClient


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


def _filled_order_for_break_even(buy):
  client = SimulatedMTClient()
  maxima = [(108.0, 40), (106.0, 20)]
  minima = [(95.0, 35), (94.0, 15)]
  if buy:
    ohlc = _ohlc(
        closed_start=100.0, closed_penultimate=105.0, closed_last=111.0,
        entry=110.0, current_close=110.0,
    )
  else:
    maxima = [(108.0, 40), (110.0, 20)]
    minima = [(101.0, 35), (103.0, 15)]
    ohlc = _ohlc(
        closed_start=110.0, closed_penultimate=105.0, closed_last=99.0,
        entry=100.0, current_close=100.0,
    )
  strategy = PivotTrend(client)
  strategy._pivots = lambda _prices, kind: maxima if kind == 'max' else minima
  order = strategy.indicator(ohlc, 'SP500', ohlc.datetime[-1])
  client.set_now(ohlc.datetime[-1])
  client.create_new_order(order)
  strategy.handle_filled_orders(order)
  strategy._entry_atrs[order.ticket] = 1.0
  return strategy, order, ohlc


@pytest.mark.parametrize('buy', [True, False])
def test_break_even_moves_stop_after_completed_close_reaches_entry_atr(buy):
  strategy, order, ohlc = _filled_order_for_break_even(buy)
  ohlc.datetime = np.array([
      value + pd.Timedelta(minutes=5) for value in ohlc.datetime
  ])
  ohlc.close[-2] = order.price + (1.0 if buy else -1.0)
  strategy._pivots = lambda _prices, _kind: []

  strategy.indicator(ohlc, 'SP500', ohlc.datetime[-1])

  assert order.stop_loss == order.price


@pytest.mark.parametrize('buy', [True, False])
def test_break_even_ignores_early_or_current_open_only_movement(buy):
  strategy, order, ohlc = _filled_order_for_break_even(buy)
  initial_stop = order.stop_loss
  ohlc.datetime = np.array([
      value + pd.Timedelta(minutes=5) for value in ohlc.datetime
  ])
  ohlc.close[-2] = order.price + (0.99 if buy else -0.99)
  ohlc.close[-1] = order.price + (100.0 if buy else -100.0)
  ohlc.high[-1] = order.price + 100.0
  ohlc.low[-1] = order.price - 100.0
  strategy._pivots = lambda _prices, _kind: []

  strategy.indicator(ohlc, 'SP500', ohlc.datetime[-1])

  assert order.stop_loss == initial_stop


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
def test_indicator_allows_confirmed_breakout_regardless_of_prior_trend(
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
