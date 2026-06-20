"""Strategy factory."""
from typing import Optional
from tradeo.strategies.strategy import Strategy
from tradeo.log import log
from tradeo.mt_client import MT_Client


def strategy_factory(
    mt_client: MT_Client,
    strategy_name: str,
) -> Optional[Strategy]:
  """Given a strategy name, return the corresponding strategy object."""
  from sorul_tradingbot.strategy.private.tnt import TNT
  from sorul_tradingbot.strategy.private.volume_07 import Volume as Volume07
  from sorul_tradingbot.strategy.private.volume_08 import Volume as Volume08
  from sorul_tradingbot.strategy.private.volume_09 import Volume as Volume09
  from sorul_tradingbot.strategy.private.volume_10 import Volume as Volume10
  from sorul_tradingbot.strategy.private.volume_11 import Volume as Volume11
  from sorul_tradingbot.strategy.private.volume_12 import Volume as Volume12
  from sorul_tradingbot.strategy.private.volume_14 import Volume as Volume14
  from sorul_tradingbot.strategy.private.volume_15 import Volume as Volume15
  from sorul_tradingbot.strategy.private.volume_16 import Volume as Volume16
  from sorul_tradingbot.strategy.private.volume_17 import Volume as Volume17
  from sorul_tradingbot.strategy.private.volume_18 import Volume as Volume18
  from sorul_tradingbot.strategy.private.volume_19 import Volume as Volume19

  for s in [
      TNT(mt_client),
      Volume07(mt_client),
      Volume08(mt_client),
      Volume09(mt_client),
      Volume10(mt_client),
      Volume11(mt_client),
      Volume12(mt_client),
      Volume14(mt_client),
      Volume15(mt_client),
      Volume16(mt_client),
      Volume17(mt_client),
      Volume18(mt_client),
      Volume19(mt_client),
  ]:
    if s.strategy_name in strategy_name or strategy_name in s.strategy_name:
      return s

  error_message = f'Not such {strategy_name} strategy!'
  log.warning(error_message)
  return None
