"""Confirmed pivot breakouts with ATR-buffered structural stops.

Designed for M5 input, but no timeframe or trading-session restriction is
imposed. Positions can stay open overnight. ATR is the arithmetic mean of
the last ``atr_period`` true ranges, not Wilder smoothing. Lots are fixed,
so monetary risk varies with stop distance. There is no profit target.

The last OHLC row is the entry candle: only its open is used. All indicators
use closed candles. In the existing simulator, entry-bar stops are not checked
and trailing changes take effect at the next position evaluation. Backtests
retain those simulator limitations; this strategy does not change execution.
"""
from datetime import datetime
from pathlib import Path
import sys
from typing import Optional

import numpy as np
from tradeo.mt_client import MT_Client
from tradeo.ohlc import OHLC
from tradeo.order import (
    ImmutableOrderDetails, MutableOrderDetails, Order, OrderPrice, OrderType,
)
from tradeo.strategies.strategy import Strategy
from tradeo.trading_methods import get_pivots
from tradeo.utils import create_magic_number


class PivotTrend(Strategy):
  """Trade HH/HL or LH/LL structure followed by a closed-candle breakout."""

  def __init__(
      self,
      mt_client: MT_Client,
      pivot_left: int = 10,
      pivot_right: int = 4,
      atr_period: int = 14,
      atr_buffer: float = 0.5,
      lots: float = 0.01,
  ) -> None:
    """Configure pivot confirmation, volatility buffer and fixed lot size."""
    super().__init__(strategy_name='PivotTrend v0.1', mt_client=mt_client)
    for name, value in (
        ('pivot_left', pivot_left), ('pivot_right', pivot_right),
        ('atr_period', atr_period),
    ):
      if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    if not np.isfinite(atr_buffer) or atr_buffer < 0:
      raise ValueError('atr_buffer must be finite and non-negative')
    if not np.isfinite(lots) or lots <= 0:
      raise ValueError('lots must be finite and positive')
    self.pivot_left = pivot_left
    self.pivot_right = pivot_right
    self.atr_period = atr_period
    self.atr_buffer = atr_buffer
    self.lots = lots
    self._last_bars: dict[str, datetime] = {}
    self._broken_pivots: dict[tuple[str, bool], datetime] = {}
    self._stop_pivots: dict[str, datetime] = {}
    self._requested_stops: dict[int, float] = {}
    self._entry_atrs_by_magic: dict[str, float] = {}
    self._entry_atrs: dict[int, float] = {}
    self._break_even_tickets: set[int] = set()

  def indicator(
      self, ohlc: OHLC, symbol: str, now_date: datetime,
  ) -> Optional[Order]:
    """Manage stops and return at most one signal per symbol and new bar."""
    _ = now_date
    warmup = max(self.pivot_left + self.pivot_right + 1, self.atr_period + 1)
    if len(ohlc) < warmup + 1:
      return None
    bar = ohlc.datetime[-1]
    if symbol in self._last_bars and bar <= self._last_bars[symbol]:
      return None
    highs, lows, closes = ohlc.high[:-1], ohlc.low[:-1], ohlc.close[:-1]
    entry = float(ohlc.open[-1])
    if not np.isfinite([highs, lows, closes]).all() or not np.isfinite(entry):
      return None
    if entry <= 0 or np.any(highs < lows):
      return None
    self._last_bars[symbol] = bar
    true_ranges = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])),
    )
    atr = float(np.mean(true_ranges[-self.atr_period:]))
    if not np.isfinite(atr) or atr <= 0:
      return None
    maxima = self._pivots(highs, 'max')
    minima = self._pivots(lows, 'min')
    self._trail_orders(symbol, entry, atr, maxima, minima, ohlc.datetime[:-1])
    self._place_break_even_orders(symbol, float(closes[-1]))
    if len(maxima) < 2 or len(minima) < 2:
      return None
    bullish = maxima[0][0] > maxima[1][0] and minima[0][0] > minima[1][0]
    bearish = maxima[0][0] < maxima[1][0] and minima[0][0] < minima[1][0]
    buy = bullish and closes[-2] <= maxima[0][0] < closes[-1]
    sell = bearish and closes[-2] >= minima[0][0] > closes[-1]
    if not (buy or sell):
      return None
    buy = bool(buy)
    pivot = maxima[0] if buy else minima[0]
    pivot_time = ohlc.datetime[:-1][pivot[1]]
    key = (symbol, buy)
    if self._broken_pivots.get(key) == pivot_time:
      return None
    # Consume even skipped breakouts: an occupied symbol is not a delayed entry.
    self._broken_pivots[key] = pivot_time
    opposing = minima[0] if buy else maxima[0]
    stop = opposing[0] + (-1 if buy else 1) * self.atr_buffer * atr
    if not self._valid_stop(entry, stop, buy) or self._own_orders(symbol):
      return None
    self._stop_pivots[symbol] = ohlc.datetime[:-1][opposing[1]]
    order = Order(
        MutableOrderDetails(OrderPrice(price=entry, stop_loss=float(stop)),
                            lots=self.lots),
        ImmutableOrderDetails(
            symbol=symbol, order_type=OrderType(buy=buy, market=True),
            magic=create_magic_number(), comment=self.strategy_name,
        ),
    )
    self._entry_atrs_by_magic[order.magic] = atr
    return order

  def _pivots(self, prices: np.ndarray, kind: str) -> list:
    """Return the two latest fully confirmed pivots, most recent first."""
    return get_pivots(
        prices, left=self.pivot_left, right=self.pivot_right,
        n_pivot=2, max_min=kind,
    )

  def _own_orders(self, symbol: str) -> list[Order]:
    """Isolate this strategy's positions from other strategies."""
    return [order for order in self.mt_client.open_orders
            if order.symbol == symbol and order.comment == self.strategy_name]

  @staticmethod
  def _valid_stop(price: float, stop: float, buy: bool) -> bool:
    """Reject non-finite, non-positive and wrong-side protection."""
    return bool(np.isfinite([price, stop]).all() and price > 0 and stop > 0
                and (stop < price if buy else stop > price))

  def _trail_orders(
      self, symbol: str, entry: float, atr: float,
      maxima: list, minima: list, dates: np.ndarray,
  ) -> None:
    """Tighten on new opposing pivots, never on changing ATR alone."""
    active_tickets = {order.ticket for order in self.mt_client.open_orders}
    self._requested_stops = {
        ticket: stop for ticket, stop in self._requested_stops.items()
        if ticket in active_tickets
    }
    self._entry_atrs = {
        ticket: atr for ticket, atr in self._entry_atrs.items()
        if ticket in active_tickets
    }
    self._break_even_tickets.intersection_update(active_tickets)
    active_magics = {order.magic for order in self._own_orders(symbol)}
    self._entry_atrs_by_magic = {
        magic: atr for magic, atr in self._entry_atrs_by_magic.items()
        if magic in active_magics
    }
    for order in self._own_orders(symbol):
      if not order.order_type.market:
        continue
      buy = order.order_type.buy
      pivots = minima if buy else maxima
      if not pivots:
        continue
      pivot, index = pivots[0]
      pivot_time = dates[index]
      previous_pivot = self._stop_pivots.get(symbol)
      if previous_pivot is not None and pivot_time <= previous_pivot:
        continue
      self._stop_pivots[symbol] = pivot_time
      stop = float(pivot + (-1 if buy else 1) * self.atr_buffer * atr)
      previous = self._requested_stops.get(order.ticket, order.stop_loss)
      previous = max(previous, order.stop_loss) if buy else min(
          previous, order.stop_loss,
      )
      improves = stop > previous if buy else stop < previous
      if improves and self._valid_stop(entry, stop, buy):
        self._modify_stop(order, stop)
        self._requested_stops[order.ticket] = stop

  def _place_break_even_orders(self, symbol: str, close: float) -> None:
    """Move eligible positions to entry after a completed favourable close."""
    if not np.isfinite(close):
      return
    for order in self._own_orders(symbol):
      if not order.order_type.market or order.ticket in self._break_even_tickets:
        continue
      entry_atr = self._entry_atrs.get(order.ticket)
      if entry_atr is None or not np.isfinite(entry_atr) or entry_atr <= 0:
        continue
      buy = order.order_type.buy
      moved_favourably = (
          close >= order.price + entry_atr if buy
          else close <= order.price - entry_atr
      )
      if not moved_favourably:
        continue
      previous = self._requested_stops.get(order.ticket, order.stop_loss)
      previous = max(previous, order.stop_loss) if buy else min(
          previous, order.stop_loss,
      )
      stop = float(order.price)
      improves = stop > previous if buy else stop < previous
      if improves:
        self._modify_stop(order, stop)
        self._requested_stops[order.ticket] = stop
      self._break_even_tickets.add(order.ticket)

  def _modify_stop(self, order: Order, stop: float) -> None:
    """Use broker commands live and the simulator's mutable order storage."""
    modify = getattr(self.mt_client, 'send_modify_order_command', None)
    if callable(modify):
      modify(order.ticket, MutableOrderDetails(
          OrderPrice(stop_loss=stop, take_profit=order.take_profit),
          lots=order.lots, expiration=order.expiration,
      ))
      return
    # Source identity also supports the simulator CLI running as __main__.
    module = sys.modules.get(type(self.mt_client).__module__)
    simulator_path = Path(__file__).resolve().parents[1] / 'simulator/simulator.py'
    client_type = getattr(module, 'SimulatedMTClient', None)
    if (client_type is None or not isinstance(self.mt_client, client_type)
        or Path(getattr(module, '__file__', '')).resolve() != simulator_path):
      raise TypeError('Client does not support stop modification')
    order._mutable_details._prices.stop_loss = stop

  def check_order_viability(
      self, order: Order, min_risk_profit: float = 1.5,
      date: Optional[datetime] = None, **kwargs,
  ) -> bool:
    """Allow target-free orders without the base reward/risk or hour filter."""
    _ = min_risk_profit, date, kwargs
    return (order.comment == self.strategy_name
            and not self._own_orders(order.symbol)
            and self._valid_stop(order.price, order.stop_loss,
                                 order.order_type.buy))

  def handle_filled_orders(self, order: Order, **kwargs) -> None:
    """Associate a filled order with the ATR used when its signal was made."""
    _ = kwargs
    entry_atr = self._entry_atrs_by_magic.pop(order.magic, None)
    if entry_atr is not None:
      self._entry_atrs[order.ticket] = entry_atr
