"""Forex Event Handler."""
from tradeo.event_handlers.event_handler import EventHandler
from tradeo.mt_client import MT_Client
from tradeo.config import Config
from tradeo.ohlc import OHLC
from datetime import datetime

from sorul_tradingbot.strategy.private.tnt import TNT


class ForexEventHandler(EventHandler):
  """This class only provides 'on_historical_data' event holder."""

  def __init__(self):
    """Initialize the attributes."""
    super().__init__('BasicEventHandler')

  def on_historical_data(
          self,
          mt_client: MT_Client,
          symbol: str,
          data: OHLC
  ) -> None:
    """Handle the return of GET_HISTORICAL_DATA command."""
    now_date = datetime.now(Config.utc_timezone)
    strategy = TNT()
    possible_order = strategy.indicator(data, symbol, now_date)
    if possible_order and strategy.check_order_viability(
            mt_client, possible_order):
      mt_client.create_new_order(possible_order)
