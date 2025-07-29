"""Forex Event Handler."""
from tradeo.event_handlers.event_handler import EventHandler
from tradeo.mt_client import MT_Client
from tradeo.config import Config
from tradeo.ohlc import OHLC
from tradeo.strategies.strategy import Strategy
from datetime import datetime
from typing import List

from sorul_tradingbot.strategy.private.volume_07 import Volume as Volume07
from sorul_tradingbot.strategy.private.volume_08 import Volume as Volume08
from sorul_tradingbot.strategy.private.volume_09 import Volume as Volume09


class ForexEventHandler(EventHandler):
  """This class only provides 'on_historical_data' event holder."""

  def __init__(self):
    """Initialize the attributes."""
    super().__init__('ForexEventHandler')

  def on_historical_data(self, mt_client: MT_Client, symbol: str,
                         data: OHLC) -> None:
    """Handle the return of GET_HISTORICAL_DATA command."""
    now_date = datetime.now(Config.utc_timezone)
    strategies: List[Strategy] = [
        Volume07(mt_client),
        Volume08(mt_client),
        Volume09(mt_client),
    ]
    for strategy in strategies:
      possible_order = strategy.indicator(data, symbol, now_date)
      if possible_order and strategy.check_order_viability(possible_order):
        mt_client.create_new_order(possible_order)
