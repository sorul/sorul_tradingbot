"""Strategy factory."""
from tradeo.strategies.strategy import Strategy
from tradeo.log import log
from tradeo.mt_client import MT_Client


def strategy_factory(mt_client: MT_Client, strategy_name: str) -> Strategy:
  """Given a strategy name, return the corresponding strategy object."""
  from sorul_tradingbot.strategy.private.tnt import TNT
  from sorul_tradingbot.strategy.private.volume import Volume

  for s in [
    TNT(mt_client),
    Volume(mt_client)
  ]:
    if s.strategy_name in strategy_name or strategy_name in s.strategy_name:
      return s

  error_message = f'Not such {strategy_name} strategy!'
  log.error(error_message)
  raise Exception(error_message)
