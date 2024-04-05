"""Forex executable."""
from tradingbot.executable.executable import Executable
from tradingbot.utils import get_remaining_symbols, reboot_mt
from tradingbot.config import Config
from tradingbot.mt_client import MT_Client
from tradingbot.log import log
from tradingbot.files import Files
import tradingbot.files as f
from datetime import datetime, timedelta
from random import randrange
import traceback

from sorul_tradingbot.event_handler import ForexEventHandler
from sorul_tradingbot.strategy.private.tnt import TNT


class ForexExecutable(Executable):
  """Forex executable."""

  def entry_point(self):
    """Entry point of the forex bot."""
    if not self.is_locked() and self.check_time_viability():
      mt_client = MT_Client(event_handler=ForexEventHandler())
      try:
        self.main(mt_client)
      except Exception:  # noqa
        # Finish the bot
        self.finish(mt_client)
        # Log the error
        log.error(traceback.format_exc())

  def main(self, mt_client: MT_Client):
    """Execute forex bot."""
    # First of all, we lock the execution of the forex bot.
    # To prevent another execution to run at the same time.
    f.lock(Files.FOREX_LOCK)

    # Start the MT Client
    mt_client.start()

    # Clean all files
    mt_client.clean_all_command_files()
    mt_client.clean_all_historical_files()
    mt_client.clean_messages()
    f.reset_successful_symbols_file()

    # Dates
    utc_date = datetime.now(Config.utc_timezone)
    local_date = datetime.now(Config.local_timezone)

    # The instant of time that executed this main
    execution_time = timedelta(
        minutes=utc_date.minute % 5,
        seconds=utc_date.second,
        microseconds=utc_date.microsecond
    )

    # Send profit message
    self._send_profit_message(mt_client, local_date)

    # Send commands to obtain the historical data
    [
        mt_client.get_historical_data(s, Config.timeframe)
        for s in Config.symbols
    ]

    # Send commands to obtain bid/ask
    mt_client.subscribe_symbols(Config.symbols)

    # Handle the existing trades
    self.handle_trades(mt_client)

    # Process the result of "get_historical_data"
    self.handle_new_historical_data(mt_client, utc_date, execution_time)

    # Finish the main
    self.finish(mt_client)

  def handle_trades(self, mt_client: MT_Client) -> None:
    """Handle the existing trades."""
    orders = mt_client.check_open_orders()
    strategy = TNT()
    for order in orders:
      if order.order_type.pending:
        strategy.handle_pending_orders(mt_client, order)
      elif order.order_type.market:
        strategy.handle_filled_orders(mt_client, order)

    len_orders = len(orders)
    message = f'Number of open orders: {len_orders}'
    if len_orders > 900:
      log.warning(message)
    else:
      log.debug(message)

  def handle_new_historical_data(
          self,
          mt_client: MT_Client,
          utc_date: datetime,
          execution_time: timedelta
  ) -> None:
    """Handle the new historical data."""
    # Initialize the remaining symbols
    rs = get_remaining_symbols()

    # The execution will take up to 4 minutes
    stop_condition = utc_date - execution_time + timedelta(minutes=4)

    while len(rs) > 0 and datetime.now(Config.utc_timezone) < stop_condition:
      # Get randomly the next symbol
      next_symbol = rs[randrange(len(rs))]

      # Check if JSON data is available to trigger the event
      mt_client.check_historical_data(next_symbol)

      # Update the remaining symbols
      rs = get_remaining_symbols()

    # Check if there are remaining symbols to process
    if len(rs) > 0:
      log.warning(f'{len(rs)} remaining symbols to process.')
    else:
      f.reset_consecutive_times_down_file()

    # Check if MT needs to restart
    self._check_mt_needs_to_restart(len(rs))

  def _send_profit_message(
          self, mt_client: MT_Client, local_date: datetime) -> bool:
    """Get the balance of the account."""
    balance = mt_client.get_balance()
    last_balance = f.get_last_balance()
    difference = balance - last_balance
    emoji = '🚀' if difference >= 0 else '☔'
    message_condition = local_date.hour % 12 == 0 and local_date.minute == 5
    if message_condition:
      log.info(f'{emoji} {difference:.2f} €')
      return True
    else:
      return False

  def _check_mt_needs_to_restart(self, n_remaining_symbols: int) -> None:
    """Check if MT needs to restart."""
    ctd = f.get_consecutive_times_down()
    symbols_len = len(Config.symbols)
    if n_remaining_symbols > int(symbols_len / 2) and ctd > 4:
      reboot_mt()
    else:
      f.increment_consecutive_times_down()

  def is_locked(self) -> bool:
    """Return True if the forex-bot is running."""
    if f.file_exists(Files.FOREX_LOCK.value):
      log.warning('Forex bot is locked.')
      return True
    else:
      return False

  def check_time_viability(self) -> bool:
    """Check if the forex bot is viable to run."""
    now_date = datetime.now(Config.broker_timezone)
    # Monday (0) -> Sunday (6)
    is_weekday = now_date.weekday() in [0, 1, 2, 3, 4]
    # TODO: When executions are performed with the real account,
    # we need to consider testing the removal of this condition.
    is_not_on_the_hour = now_date.minute != 0
    return is_weekday and is_not_on_the_hour

  def finish(self, mt_client: MT_Client) -> None:
    """Finish the forex bot."""
    mt_client.stop()
    f.unlock(Files.FOREX_LOCK)
