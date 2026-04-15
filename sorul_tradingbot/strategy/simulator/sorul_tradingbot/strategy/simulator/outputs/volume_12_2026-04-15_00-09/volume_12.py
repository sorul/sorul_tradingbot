"""Volume Strategy."""
from typing import Union, Optional, Dict, Tuple
import pandas as pd
from datetime import datetime, time, date
from tradeo.trading_methods import calculate_poc_vah_val, calculate_heikin_ashi
from tradeo.ohlc import OHLC
from tradeo.utils import create_magic_number
from tradeo.strategies.strategy import Strategy
from tradeo.mt_client import MT_Client
from tradeo.config import Config
from tradeo.order import (
    Order, MutableOrderDetails, ImmutableOrderDetails, OrderPrice, OrderType
)
from tradeo.log import log
from zoneinfo import ZoneInfo


class Volume(Strategy):
  """Strategy based on tendency and previous benefits.

  Volume v0.12:
  - Si hay una orden abierta en el mismo dia, no se abre una nueva.
  - Se mejora la logica de entrada basandose en VAH y VAL.

  """
  def __init__(self, mt_client: MT_Client, poc_ratio_threshold: float = 0.75):
    """Initialize the attributes."""
    super().__init__(strategy_name='Volume v0.12', mt_client=mt_client)
    if not 0.0 <= poc_ratio_threshold <= 1.0:
      raise ValueError('poc_ratio_threshold must be between 0.0 and 1.0')
    self.take_profit_size = 300
    self.stop_loss_size = 300
    self.point = 0.1
    # Ratio of closed candles required above/below the POC level.
    self.poc_ratio_threshold = poc_ratio_threshold

  def indicator(
      self,
      ohlc: OHLC,
      symbol: str,
      now_date: datetime,
  ) -> Union[Order, None]:
    """Return an order if the strategy is triggered."""
    start_time, close_time = self._get_session_times(now_date)
    start = start_time.strftime('%H:%M')
    close = close_time.strftime('%H:%M')

    if symbol == 'SP500' and (start_time <= now_date.time() <= close_time):
      # Transform the Japan ticks to Heikin Ashi format
      ohlc = calculate_heikin_ashi(ohlc)
      daily_levels = calculate_poc_vah_val(
          ohlc, session_start=start, session_end=close
      )
      previous_date = self._get_previous_available_date(now_date, daily_levels)
      if previous_date and previous_date in daily_levels:
        vah = daily_levels[previous_date]['VAH']
        poc = daily_levels[previous_date]['POC']
        val = daily_levels[previous_date]['VAL']
        return self._check_and_open_trade(
            ohlc=ohlc,
            symbol=symbol,
            vah=vah,
            poc=poc,
            val=val,
            start_time=start_time,
            close_time=close_time,
            now_date=now_date,
        )

  def _get_previous_available_date(
      self, current_date: datetime, daily_levels: Dict
  ) -> Optional[date]:
    """Find the most recent date in daily_levels before the current date."""
    previous_date = pd.Timestamp(current_date) - pd.Timedelta(days=1)
    while previous_date.date() not in daily_levels:
      previous_date -= pd.Timedelta(days=1)
      if previous_date < pd.Timestamp(current_date) - pd.Timedelta(days=5):
        return None
    return previous_date.date()

  def _get_session_times(self, now_date: datetime) -> Tuple[time, time]:
    """Return UTC session times adjusted for New York daylight savings."""
    ny_tz = ZoneInfo('America/New_York')
    utc_tz = ZoneInfo('UTC')
    if now_date.tzinfo is None:
      aware_now = now_date.replace(tzinfo=utc_tz)
    else:
      aware_now = now_date.astimezone(utc_tz)
    now_ny = aware_now.astimezone(ny_tz)
    dst = now_ny.dst()
    is_dst = bool(dst and dst.total_seconds())
    if is_dst:
      return time(14, 30), time(20, 0)
    return time(15, 30), time(21, 0)

  def _check_and_open_trade(
      self,
      ohlc: OHLC,
      symbol: str,
      vah: float,
      poc: float,
      val: float,
      start_time: time,
      close_time: time,
      now_date: datetime,
  ) -> Optional[Order]:
    buy, sell = False, False

    ohlc_today = self._calculate_today_ohlc(
        ohlc=ohlc,
        start_time=start_time,
        close_time=close_time,
        now_date=now_date,
    )
    if len(ohlc_today.open) < 2:
      return None

    # 📝 Note: [-1] candle is not closed yet, use [-2] for last closed candle.

    # Check for buys
    buy = self._check_for_buy(ohlc=ohlc_today, vah=vah, poc=poc)
    if buy:
      entry_price = ohlc_today.close[-1]
      take_profit = entry_price + self.take_profit_size * self.point
      stop_loss = entry_price - self.stop_loss_size * self.point
    # Check for sells
    sell = self._check_for_sell(ohlc=ohlc_today, val=val, poc=poc)
    if sell:
      entry_price = ohlc_today.close[-1]
      take_profit = entry_price - self.take_profit_size * self.point
      stop_loss = entry_price + self.stop_loss_size * self.point
      sell = True
    # Return order if buy or sell conditions met
    if buy or sell:
      magic = create_magic_number()
      return Order(
          MutableOrderDetails(
              OrderPrice(take_profit=take_profit, stop_loss=stop_loss),
              lots=0.01
          ),
          ImmutableOrderDetails(
              symbol=symbol,
              order_type=OrderType(buy=buy, market=True),
              magic=magic,
              comment=self.strategy_name
          )
      )

  def _calculate_today_ohlc(
      self,
      ohlc: OHLC,
      start_time: time,
      close_time: time,
      now_date: datetime,
  ) -> OHLC:
    """Calculate today's OHLC data."""
    ohcl_df = ohlc.to_dataframe()
    now = pd.Timestamp(now_date)
    index_tz = ohcl_df.index.tz  # type: ignore
    if index_tz is not None:
      if now.tzinfo is None:
        now = now.tz_localize(index_tz)
      else:
        now = now.tz_convert(index_tz)
    elif now.tzinfo is not None:
      now = now.tz_convert(None)
    today_tp = now.normalize()
    start_dt = today_tp + pd.Timedelta(
        hours=start_time.hour, minutes=start_time.minute
    )
    close_dt = today_tp + pd.Timedelta(
        hours=close_time.hour, minutes=close_time.minute
    )
    df_today = ohcl_df[(ohcl_df.index >= start_dt) &
                       (ohcl_df.index <= close_dt)]
    return OHLC(df_today)

  def _check_for_buy(
      self,
      ohlc: OHLC,
      vah: float,
      poc: float,
  ) -> bool:
    """Check for buy conditions."""
    opens = ohlc.open
    closes = ohlc.close
    highs = ohlc.high
    lows = ohlc.low

    vah_touched = any(l <= vah <= h for h, l in zip(highs, lows))
    total_closed = len(closes[:-1])
    candles_above_poc = sum(1 for c in closes[:-1] if c > poc)
    above_poc = (
        total_closed > 0 and
        (candles_above_poc / total_closed) >= self.poc_ratio_threshold
    )
    opened_above_vah = opens[0] > vah
    green_candle = closes[-2] > opens[-2]
    if above_poc and opened_above_vah and vah_touched and green_candle:
      buy = True
    else:
      buy = False
    log.debug(
        f'Buy: {buy}, vah: {vah}, vah_touched: {vah_touched}, above_poc: {above_poc}, opened_above_vah: { opened_above_vah}, green_candle: {green_candle}'
    )
    return buy

  def _check_for_sell(
      self,
      ohlc: OHLC,
      val: float,
      poc: float,
  ) -> bool:
    """Check for sell conditions."""
    opens = ohlc.open
    closes = ohlc.close
    highs = ohlc.high
    lows = ohlc.low

    val_touched = any(l <= val <= h for h, l in zip(highs, lows))
    total_closed = len(closes[:-1])
    candles_below_poc = sum(1 for c in closes[:-1] if c < poc)
    below_poc = (
        total_closed > 0 and
        (candles_below_poc / total_closed) >= self.poc_ratio_threshold
    )
    opened_below_val = opens[0] < val
    red_candle = closes[-2] < opens[-2]
    if below_poc and opened_below_val and val_touched and red_candle:
      sell = True
    else:
      sell = False
    log.debug(
        f'Sell: {sell}, val: {val}, val_touched: {val_touched}, below_poc: {below_poc}, opened_below_val: { opened_below_val}, red_candle: {red_candle}'
    )
    return sell

  def check_order_viability(
      self,
      order: Order,
      min_risk_profit: float = 0.9,
  ) -> bool:
    """Check if the order is viable."""
    _ = min_risk_profit
    symbol = order.symbol
    order_type = order.order_type
    current_dt = getattr(self.mt_client, 'current_time', None)
    if current_dt is None:
      today = datetime.now(Config.broker_timezone)
    else:
      if current_dt.tzinfo is None:
        today = Config.broker_timezone.localize(current_dt)
      else:
        today = current_dt.astimezone(Config.broker_timezone)

    for o in self.mt_client.open_orders:
      a = o.symbol == symbol
      b = o.order_type == order_type
      c = o.comment == self.strategy_name
      if a and b and c:
        return False

    for t in self.mt_client.historical_trades:
      if (
          t.execution_time is not None and
          t.execution_time.date() == today.date()
      ):
        return False

    return True
