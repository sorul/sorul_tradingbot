# flake8: noqa
from datetime import date

import pandas as pd
from tradeo.ohlc import OHLC

from sorul_tradingbot.strategy.private.volume_19 import Volume
from sorul_tradingbot.strategy.simulator.simulator import SimulatedMTClient


def _volume(levels_period='weekly'):
  return Volume(SimulatedMTClient(), levels_period=levels_period)


def _ohlc():
  return OHLC(
      pd.DataFrame(
          {
              'open': [100.0, 101.0, 110.0, 111.0],
              'high': [102.0, 103.0, 112.0, 113.0],
              'low': [99.0, 100.0, 109.0, 110.0],
              'close': [101.0, 102.0, 111.0, 112.0],
              'volume': [10.0, 20.0, 30.0, 40.0],
          },
          index=pd.to_datetime(
              [
                  '2026-05-26 15:00:00',
                  '2026-05-26 16:00:00',
                  '2026-05-27 15:00:00',
                  '2026-05-27 16:00:00',
              ]
          ),
      ),
      volume_column_name='volume',
  )


def test_daily_levels_are_grouped_by_session_date():
  levels = _volume('daily')._calculate_poc_vah_val_by_period(
      _ohlc(),
      session_start='14:30',
      session_end='20:00',
      levels_period='daily',
  )

  assert set(levels) == {date(2026, 5, 26), date(2026, 5, 27)}


def test_weekly_levels_are_grouped_by_monday_week_start():
  levels = _volume('weekly')._calculate_poc_vah_val_by_period(
      _ohlc(),
      session_start='14:30',
      session_end='20:00',
      levels_period='weekly',
  )

  assert set(levels) == {date(2026, 5, 25)}


def test_levels_period_is_validated():
  try:
    _volume('monthly')
  except ValueError as exc:
    assert 'levels_period must be one of: daily, weekly' in str(exc)
  else:
    raise AssertionError('Expected invalid levels_period to raise ValueError')
