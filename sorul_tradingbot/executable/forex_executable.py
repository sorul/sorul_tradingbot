"""Forex executable."""
from tradeo.executable.executable import Executable
from tradeo.files import write_file
from tradeo.files import Files
from tradeo.config import Config
from tradeo.mt_client import MT_Client
from tradeo.strategies.strategy import Strategy
from tradeo.log import log
from tradeo.blocker import Blocker
from tradeo.utils import (
    get_consecutive_times_down,
    increment_consecutive_times_down,
    reset_consecutive_times_down, get_last_balance
)
from datetime import datetime, timedelta
from random import randrange
import traceback
import subprocess
from time import sleep
from typing import List

from sorul_tradingbot.event_handler import ForexEventHandler
from sorul_tradingbot.strategy.private.tnt import TNT
from sorul_tradingbot.strategy.private.volume import Volume


class ForexExecutable(Executable):
  """Forex executable."""

  def __init__(self):
    """Initialize the forex bot."""
    self.name = 'BasicForex'

  def entry_point(self):
    """Entry point of the forex bot."""
    if self.check_time_viability():
      mt_client = MT_Client(event_handler=ForexEventHandler())
      try:
        with Blocker(lockfile=f'/tmp/{self.name}.lock'):
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

    # Start the MT Client
    mt_client.start()

    # Clean all files
    mt_client.clean_all_command_files()
    mt_client.clean_all_historical_files()
    mt_client.clean_messages()

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
    strategies: List[Strategy] = [
        # TNT(mt_client),
        Volume(mt_client)
    ]
    for strategy in strategies:
      for order in orders:
        if order.order_type.pending:
          strategy.handle_pending_orders(order)
        elif order.order_type.market:
          strategy.handle_filled_orders(order)

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
    rs = mt_client.get_remaining_symbols()

    # The execution will take up to 4 minutes
    stop_condition = utc_date - execution_time + timedelta(minutes=4)

    while len(rs) > 0 and datetime.now(Config.utc_timezone) < stop_condition:
      # Get randomly the next symbol
      next_symbol = rs[randrange(len(rs))]

      # Check if JSON data is available to trigger the event
      mt_client.check_historical_data(next_symbol)

      # Update the remaining symbols
      rs = mt_client.get_remaining_symbols()

    # Check if there are remaining symbols to process
    if len(rs) <= int(len(Config.symbols) / 2):
      reset_consecutive_times_down()
    else:
      # Check if MT needs to restart
      self._check_mt_needs_to_restart(len(rs))

  def _send_profit_message(
          self, mt_client: MT_Client, local_date: datetime) -> bool:
    """Get the balance of the account."""
    balance = mt_client.get_balance()
    last_balance = get_last_balance()
    difference = balance - last_balance
    emoji = '🚀' if difference >= 0 else '☔'
    message_condition = local_date.hour % 12 == 0 and local_date.minute == 5
    if message_condition and balance != -1:
      log.info(f'{emoji} {difference:.2f} €')
      write_file(Files.LAST_BALANCE.value, str(balance))
      return True
    else:
      return False

  def _check_mt_needs_to_restart(self, n_remaining_symbols: int) -> None:
    """Check if MT needs to restart."""
    ctd = get_consecutive_times_down()
    symbols_len = len(Config.symbols)
    if n_remaining_symbols > int(symbols_len / 2) and ctd > 5:
      self._reboot_mt()
      reset_consecutive_times_down()
    else:
      increment_consecutive_times_down()

  def _reboot_mt(self) -> None:
    log.warning('Rebooting MetaTrader ...')

    subprocess.run(
        ['make', 'stop_docker'],
        capture_output=True, text=True
    )
    sleep(10)
    subprocess.run(
        ['make', 'start_docker'],
        capture_output=True, text=True
    )

    sleep(180)

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
    errors = mt_client.get_error_messages()
    white_list = [
        e for e in errors if e.error_type != 'WRONG_FORMAT_START_IDENTIFIER'
    ]
    if len(white_list) > 0:
      log.warning(f'Messages: {" | ".join(str(e) for e in errors)}')
    log.debug('Forex bot finished!')
