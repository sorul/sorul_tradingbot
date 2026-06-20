# flake8: noqa
from datetime import datetime, timezone

from tradeo.order import (
    ImmutableOrderDetails,
    MutableOrderDetails,
    Order,
    OrderPrice,
    OrderType,
)

from sorul_tradingbot.strategy.simulator.simulator import SimulatedMTClient


def _order(*, buy: bool, entry: float, stop_loss: float, take_profit: float) -> Order:
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
          order_type=OrderType(buy=buy, market=True),
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
