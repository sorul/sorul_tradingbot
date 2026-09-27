# flake8: noqa
from datetime import datetime, timezone

import pandas as pd
import pytest

from tradeo.order import (
    ImmutableOrderDetails,
    MutableOrderDetails,
    Order,
    OrderPrice,
    OrderType,
)

from sorul_tradingbot.strategy.simulator.simulator import (
    SimulatedMTClient,
    StrategySimulator,
    _load_data,
    _orders_to_dataframe,
    _summarize,
)


def _order(
    *,
    buy: bool,
    entry: float,
    stop_loss: float,
    take_profit: float,
    market: bool = True,
) -> Order:
  opened_at = datetime(2026, 5, 21, 18, 5, tzinfo=timezone.utc)
  return Order(
      MutableOrderDetails(
          OrderPrice(
              price=entry,
              stop_loss=stop_loss,
              take_profit=take_profit,
          )
      ),
      ImmutableOrderDetails(
          symbol='SP500',
          order_type=OrderType(buy=buy, market=market),
          magic=str(round(opened_at.timestamp())),
          comment='test',
      ),
  )


def test_place_break_even_closes_buy_at_current_price_when_market_below_entry():
  mt_client = SimulatedMTClient()
  now = datetime(2026, 5, 22, 6, 10, tzinfo=timezone.utc)
  mt_client.set_now(now)
  order = _order(buy=True, entry=100.0, stop_loss=90.0, take_profit=120.0)
  mt_client.create_new_order(order)
  mt_client.set_market_snapshot('SP500', bid=95.0, ask=95.0)

  mt_client.place_break_even(order)

  assert mt_client.open_orders == []
  assert len(mt_client.closed_trades) == 1
  trade = mt_client.closed_trades[0]
  assert trade.exit_price == 95.0
  assert trade.pnl == -5.0
  assert trade.result == 'stop_loss'


def test_place_break_even_closes_sell_at_current_price_when_market_above_entry():
  mt_client = SimulatedMTClient()
  now = datetime(2026, 5, 22, 6, 10, tzinfo=timezone.utc)
  mt_client.set_now(now)
  order = _order(buy=False, entry=100.0, stop_loss=110.0, take_profit=80.0)
  mt_client.create_new_order(order)
  mt_client.set_market_snapshot('SP500', bid=105.0, ask=105.0)

  mt_client.place_break_even(order)

  assert mt_client.open_orders == []
  assert len(mt_client.closed_trades) == 1
  trade = mt_client.closed_trades[0]
  assert trade.exit_price == 105.0
  assert trade.pnl == -5.0
  assert trade.result == 'stop_loss'


def test_place_break_even_keeps_order_open_with_real_client_break_even_price():
  mt_client = SimulatedMTClient()
  now = datetime(2026, 5, 22, 6, 10, tzinfo=timezone.utc)
  mt_client.set_now(now)
  order = _order(buy=False, entry=100.0, stop_loss=110.0, take_profit=80.0)
  mt_client.create_new_order(order)
  mt_client.set_market_snapshot('SP500', bid=95.0, ask=95.0)

  mt_client.place_break_even(order)

  assert mt_client.open_orders == [order]
  assert mt_client.closed_trades == []
  assert order.stop_loss == 99.9999


def test_closed_trade_keeps_initial_prices_and_tracks_buy_excursions():
  mt_client = SimulatedMTClient()
  opened_at = datetime(2026, 5, 22, 16, 0, tzinfo=timezone.utc)
  mt_client.set_now(opened_at)
  order = _order(
      buy=True,
      entry=100.0,
      stop_loss=90.0,
      take_profit=120.0,
  )
  mt_client.create_new_order(order)
  order._mutable_details._prices.stop_loss = 100.0001

  mt_client.evaluate_positions(
      pd.Series({'high': 121.0, 'low': 97.0}),
      opened_at,
  )

  trade = mt_client.closed_trades[0]
  assert trade.initial_stop_loss == 90.0
  assert trade.initial_take_profit == 120.0
  assert trade.initial_reward_risk == 2.0
  assert trade.max_adverse_excursion == 3.0
  assert trade.max_favorable_excursion == 21.0

  closed_df, _ = _orders_to_dataframe([trade], [])
  exported = closed_df.iloc[0]
  assert exported.initial_stop_loss == 90.0
  assert exported.initial_take_profit == 120.0
  assert exported.initial_reward_risk == 2.0
  assert exported.max_adverse_excursion == 3.0
  assert exported.max_favorable_excursion == 21.0


def test_closed_trade_tracks_sell_excursions():
  mt_client = SimulatedMTClient()
  opened_at = datetime(2026, 5, 22, 16, 0, tzinfo=timezone.utc)
  mt_client.set_now(opened_at)
  order = _order(
      buy=False,
      entry=100.0,
      stop_loss=110.0,
      take_profit=80.0,
  )
  mt_client.create_new_order(order)

  mt_client.evaluate_positions(
      pd.Series({'high': 106.0, 'low': 79.0}),
      opened_at,
  )

  trade = mt_client.closed_trades[0]
  assert trade.max_adverse_excursion == 6.0
  assert trade.max_favorable_excursion == 21.0


@pytest.mark.parametrize(
    ('close_price', 'expected_entry_price'),
    [
        (109.0, 101.5),
        (95.0, 101.5),
        (101.5, 101.5),
    ],
)
def test_market_order_without_price_uses_candle_open(
    close_price,
    expected_entry_price,
):
  class MarketStrategy:
    point = 0.1

    def indicator(self, ohlc, symbol, now):
      del ohlc, symbol, now
      return _order(
          buy=True,
          entry=0.0,
          stop_loss=90.0,
          take_profit=120.0,
      )

    def check_order_viability(self, order, date):
      del order, date
      return True

  mt_client = SimulatedMTClient()
  strategy = MarketStrategy()
  now = pd.Timestamp('2026-05-22 16:15:00', tz='UTC')
  data = pd.DataFrame(
      {
          'open': [101.5],
          'high': [110.0],
          'low': [100.0],
          'close': [close_price],
          'volume': [100],
      },
      index=[now],
  )
  simulator = StrategySimulator(
      strategy,
      data=data,
      symbol='SP500',
      mt_client=mt_client,
      show_progress=False,
  )

  simulator.run()

  assert mt_client.open_orders[0].price == pytest.approx(expected_entry_price)
  assert mt_client.get_active_orders()[0].entry_price == pytest.approx(
      expected_entry_price
  )


class _MarketOrderStrategy:
  """Create one market order and expose the candles seen by the signal."""

  def __init__(
      self,
      mt_client,
      *,
      buy=True,
      entry=0.0,
      market=True,
      stop_loss=90.0,
      take_profit=120.0,
  ):
    self.mt_client = mt_client
    self.buy = buy
    self.entry = entry
    self.market = market
    self.stop_loss = stop_loss
    self.take_profit = take_profit
    self.indicator_candles = []
    self.manage_calls = 0

  def indicator(self, ohlc, symbol, now):
    del symbol, now
    self.indicator_candles.append(
        (
            float(ohlc.open[-1]),
            float(ohlc.high[-1]),
            float(ohlc.low[-1]),
            float(ohlc.close[-1]),
        )
    )
    if len(self.indicator_candles) != 1:
      return None
    return _order(
        buy=self.buy,
        entry=self.entry,
        stop_loss=self.stop_loss,
        take_profit=self.take_profit,
        market=self.market,
    )

  def check_order_viability(self, order, date):
    del order, date
    return True

  def handle_filled_orders(self, order):
    del order
    self.manage_calls += 1

  def handle_pending_orders(self, order):
    del order


def _run_market_strategy(strategy, bars):
  mt_client = strategy.mt_client
  simulator = StrategySimulator(
      strategy,
      data=pd.DataFrame(
          bars,
          index=pd.date_range('2026-05-22 16:15:00', periods=len(bars), freq='5min', tz='UTC'),
      ),
      symbol='SP500',
      mt_client=mt_client,
      show_progress=False,
  )
  simulator.run()
  return mt_client


def test_gap_through_buy_stop_fills_at_executable_bar_open():
  strategy = _MarketOrderStrategy(SimulatedMTClient())

  mt_client = _run_market_strategy(
      strategy,
      [
          {'open': 100.0, 'high': 105.0, 'low': 95.0, 'close': 101.0, 'volume': 1},
          {'open': 85.0, 'high': 95.0, 'low': 80.0, 'close': 90.0, 'volume': 1},
      ],
  )

  trade = mt_client.closed_trades[0]
  assert trade.result == 'stop_loss'
  assert trade.exit_price == 85.0


def test_gap_through_sell_stop_fills_at_executable_bar_open():
  strategy = _MarketOrderStrategy(
      SimulatedMTClient(), buy=False, stop_loss=110.0, take_profit=80.0
  )

  mt_client = _run_market_strategy(
      strategy,
      [
          {'open': 100.0, 'high': 105.0, 'low': 95.0, 'close': 99.0, 'volume': 1},
          {'open': 115.0, 'high': 120.0, 'low': 110.0, 'close': 116.0, 'volume': 1},
      ],
  )

  trade = mt_client.closed_trades[0]
  assert trade.result == 'stop_loss'
  assert trade.exit_price == 115.0


def test_market_entry_is_protected_by_its_own_bar_stop():
  strategy = _MarketOrderStrategy(SimulatedMTClient())

  mt_client = _run_market_strategy(
      strategy,
      [
          {'open': 100.0, 'high': 105.0, 'low': 89.0, 'close': 101.0, 'volume': 1},
      ],
  )

  trade = mt_client.closed_trades[0]
  assert trade.entry_price == 100.0
  assert trade.result == 'stop_loss'
  assert trade.exit_price == 90.0


def test_market_entry_is_protected_by_its_own_bar_target():
  strategy = _MarketOrderStrategy(SimulatedMTClient())

  mt_client = _run_market_strategy(
      strategy,
      [
          {'open': 100.0, 'high': 121.0, 'low': 95.0, 'close': 101.0, 'volume': 1},
      ],
  )

  trade = mt_client.closed_trades[0]
  assert trade.result == 'take_profit'
  assert trade.exit_price == 120.0


def test_entry_bar_with_stop_and_target_hit_chooses_conservative_stop():
  strategy = _MarketOrderStrategy(SimulatedMTClient())

  mt_client = _run_market_strategy(
      strategy,
      [
          {'open': 100.0, 'high': 121.0, 'low': 89.0, 'close': 101.0, 'volume': 1},
      ],
  )

  trade = mt_client.closed_trades[0]
  assert trade.result == 'stop_loss'
  assert trade.exit_price == 90.0


def test_management_trailing_stop_applies_before_next_bar_intrabar_range():
  class TrailingStrategy(_MarketOrderStrategy):
    def handle_filled_orders(self, order):
      self.manage_calls += 1
      order._mutable_details._prices.stop_loss = 105.0

  strategy = TrailingStrategy(SimulatedMTClient())

  mt_client = _run_market_strategy(
      strategy,
      [
          {'open': 100.0, 'high': 104.0, 'low': 96.0, 'close': 102.0, 'volume': 1},
          {'open': 110.0, 'high': 112.0, 'low': 100.0, 'close': 111.0, 'volume': 1},
      ],
  )

  trade = mt_client.closed_trades[0]
  assert strategy.manage_calls == 1
  assert trade.result == 'stop_loss'
  assert trade.exit_price == 105.0


@pytest.mark.parametrize('close_price', [109.0, 95.0, 101.5])
def test_signal_sees_only_current_bar_open_and_market_entry_uses_it(close_price):
  strategy = _MarketOrderStrategy(SimulatedMTClient())

  mt_client = _run_market_strategy(
      strategy,
      [
          {
              'open': 101.5,
              'high': 110.0,
              'low': 100.0,
              'close': close_price,
              'volume': 1,
          },
      ],
  )

  assert strategy.indicator_candles == [(101.5, 101.5, 101.5, 101.5)]
  assert mt_client.get_active_orders()[0].entry_price == 101.5


def test_market_entry_with_explicit_price_uses_executable_bar_open():
  strategy = _MarketOrderStrategy(SimulatedMTClient(), entry=98.0)

  mt_client = _run_market_strategy(
      strategy,
      [
          {'open': 101.5, 'high': 110.0, 'low': 100.0, 'close': 109.0, 'volume': 1},
      ],
  )

  assert mt_client.get_active_orders()[0].entry_price == 101.5


def test_pending_order_keeps_its_requested_price():
  strategy = _MarketOrderStrategy(
      SimulatedMTClient(), entry=98.0, market=False
  )

  mt_client = _run_market_strategy(
      strategy,
      [
          {'open': 101.5, 'high': 110.0, 'low': 100.0, 'close': 109.0, 'volume': 1},
      ],
  )

  assert mt_client.get_active_orders()[0].entry_price == 98.0


def test_load_data_keeps_all_candles_on_finish_day(tmp_path):
  csv_path = tmp_path / 'candles.csv'
  csv_path.write_text(
      'datetime;open;high;low;close;volume\n'
      '2026-05-22 00:00:00;1;2;0;1;10\n'
      '2026-05-22 23:55:00;1;2;0;1;10\n'
      '2026-05-23 00:00:00;1;2;0;1;10\n',
      encoding='utf-8',
  )

  data = _load_data(csv_path, finish_date=pd.Timestamp('2026-05-22'))

  assert list(data.index) == [
      pd.Timestamp('2026-05-22 00:00:00'),
      pd.Timestamp('2026-05-22 23:55:00'),
  ]


def test_summary_reports_raw_price_points_not_eur(capsys, tmp_path):
  class Trade:
    pnl = 12.5

  summary = _summarize([Trade()], 'SP500', tmp_path / 'candles.csv')

  assert 'Net: 12.50 points' in summary
  assert 'EUR' not in summary
  assert 'points' in capsys.readouterr().out
