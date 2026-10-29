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
  from sorul_tradingbot.strategy.private.volume_18 import Volume as Volume18
  from sorul_tradingbot.strategy.private.volume_19 import Volume as Volume19
  from sorul_tradingbot.strategy.private.volume_20 import Volume as Volume20
  from sorul_tradingbot.strategy.private.pivot_trend import PivotTrend
  from sorul_tradingbot.strategy.private.pivot_trend_2 import PivotTrend as PivotTrend2
  from sorul_tradingbot.strategy.private.pivot_trend_3 import PivotTrend as PivotTrend3

  for s in [
      TNT(mt_client),
      Volume18(mt_client),
      Volume19(mt_client),
      Volume20(mt_client),
      PivotTrend(mt_client),
      PivotTrend2(mt_client),
      PivotTrend3(mt_client),
  ]:
    if s.strategy_name in strategy_name or strategy_name in s.strategy_name:
      return s

  error_message = f'Not such {strategy_name} strategy!'
  log.warning(error_message)
  return None
