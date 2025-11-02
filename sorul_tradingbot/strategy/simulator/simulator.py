"""Simple simulator loop to backtest Tradeo strategies."""
import argparse
import importlib
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Type

import pandas as pd
from tqdm import tqdm
import yfinance as yf
from tradeo.ohlc import OHLC
from tradeo.order import Order
from tradeo.strategies.strategy import Strategy

logging.getLogger('tradeo').setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)  # si usan el root

LOGGER = logging.getLogger(__name__)


@dataclass
class ActiveOrder:
  """Lightweight container with open-order metadata."""
  order: Order
  entry_time: datetime
  entry_price: float


@dataclass
class ExecutedOrder:
  """Result of an order after closing."""
  order: Order
  entry_time: datetime
  exit_time: datetime
  entry_price: float
  exit_price: float
  result: str
  pnl: float

  @property
  def execution_time(self) -> datetime:
    """Expose execution_time to mimic Trade objects."""
    return self.exit_time


class OHLCView:
  """Lightweight view over OHLC data without copying full history."""

  def __init__(self, source: OHLC, source_df: pd.DataFrame, end: int) -> None:
    if end <= 0:
      raise ValueError('end must be positive')
    self.datetime = source.datetime[:end]
    self.open = source.open[:end]
    self.high = source.high[:end]
    self.low = source.low[:end]
    self.close = source.close[:end]
    self.volume = source.volume[:end]
    self._df = source_df
    self._end = end

  def __len__(self) -> int:
    return len(self.open)

  def to_dataframe(self) -> pd.DataFrame:
    return self._df.iloc[:self._end]


class SimulatedMTClient:
  """Very small subset of MT_Client for strategy checks."""
  def __init__(self) -> None:
    self.open_orders: List[Order] = []
    self.historical_trades: List[ExecutedOrder] = []
    self.closed_trades: List[ExecutedOrder] = []
    self._active_orders: dict[int, ActiveOrder] = {}
    self._ticket_sequence = 1
    self._now: Optional[datetime] = None
    self.current_time: Optional[datetime] = None

  def set_now(self, now: datetime) -> None:
    self._now = now
    self.current_time = now

  def create_new_order(self, order: Order) -> None:
    if self._now is None:
      raise RuntimeError('Current simulation time is not set.')
    entry_price = self._infer_entry_price(order)
    order._mutable_details._prices.price = entry_price  # type: ignore[attr-defined]
    order._immutable_details.open_time = self._now  # type: ignore[attr-defined]
    order._ticket = self._ticket_sequence  # type: ignore[attr-defined]
    self._ticket_sequence += 1
    self.open_orders.append(order)
    self._active_orders[order.ticket] = ActiveOrder(
        order=order,
        entry_time=self._now,
        entry_price=entry_price,
    )
    LOGGER.info(
        'Order opened ticket=%s side=%s entry=%.2f tp=%.2f sl=%.2f',
        order.ticket,
        'BUY' if order.order_type.buy else 'SELL',
        entry_price,
        order.take_profit,
        order.stop_loss,
    )

  def evaluate_positions(self, bar: pd.Series, now: datetime) -> None:
    """Check if any active order hits TP or SL within the provided bar."""
    to_close: List[Tuple[Order, float, str]] = []
    high, low = float(bar.high), float(bar.low)
    for ticket, meta in list(self._active_orders.items()):
      order = meta.order
      is_buy = order.order_type.buy
      tp_hit, sl_hit = self._check_hits(order, high, low, is_buy)
      exit_price: Optional[float] = None
      result: Optional[str] = None
      if tp_hit and sl_hit:
        exit_price, result = self._resolve_same_bar(
            meta.entry_price, order, is_buy
        )
      elif tp_hit:
        exit_price, result = (
            order.take_profit,
            'take_profit',
        )
      elif sl_hit:
        exit_price, result = (
            order.stop_loss,
            'stop_loss',
        )
      if exit_price is not None and result is not None:
        to_close.append((order, exit_price, result))
    for order, exit_price, result in to_close:
      self._close_order(order, exit_price, now, result)

  def _check_hits(
      self,
      order: Order,
      high: float,
      low: float,
      is_buy: bool,
  ) -> Tuple[bool, bool]:
    tp = order.take_profit
    sl = order.stop_loss
    tp_hit = False
    sl_hit = False
    if tp:
      tp_hit = high >= tp if is_buy else low <= tp
    if sl:
      sl_hit = low <= sl if is_buy else high >= sl
    return tp_hit, sl_hit

  def _resolve_same_bar(
      self,
      entry_price: float,
      order: Order,
      is_buy: bool,
  ) -> Tuple[float, str]:
    """Break ties when TP and SL are reached within same bar."""
    distance_tp = abs(order.take_profit - entry_price)
    distance_sl = abs(entry_price - order.stop_loss)
    if distance_tp <= distance_sl:
      return order.take_profit, 'take_profit'
    if is_buy:
      return order.stop_loss, 'stop_loss'
    return order.stop_loss, 'stop_loss'

  def _close_order(
      self,
      order: Order,
      exit_price: float,
      exit_time: datetime,
      result: str,
  ) -> None:
    active = self._active_orders.pop(order.ticket)
    self.open_orders = [o for o in self.open_orders if o.ticket != order.ticket]
    pnl = (
        exit_price -
        active.entry_price if order.order_type.buy else active.entry_price -
        exit_price
    )
    executed = ExecutedOrder(
        order=order,
        entry_time=active.entry_time,
        exit_time=exit_time,
        entry_price=active.entry_price,
        exit_price=exit_price,
        result=result,
        pnl=pnl,
    )
    self.historical_trades.append(executed)
    self.closed_trades.append(executed)
    LOGGER.info(
        'Order closed ticket=%s result=%s pnl=%.2f',
        order.ticket,
        result,
        pnl,
    )

  @staticmethod
  def _infer_entry_price(order: Order) -> float:
    price = order.price
    if price:
      return price
    tp, sl = order.take_profit, order.stop_loss
    if tp and sl:
      return (tp + sl) / 2
    return tp or sl or 0.0

  def get_active_orders(self) -> List[ActiveOrder]:
    """Expose snapshot of open orders metadata."""
    return list(self._active_orders.values())


class StrategySimulator:
  """Main driver to iterate bars and invoke the strategy."""
  def __init__(
      self,
      strategy: Strategy,
      data: pd.DataFrame,
      symbol: str,
      mt_client: SimulatedMTClient,
      show_progress: bool = True,
  ) -> None:
    self.strategy = strategy
    self.data = data
    self.symbol = symbol
    self.mt_client = mt_client
    self._show_progress = show_progress
    self._ohlc_source = OHLC(
        data,
        volume_column_name='volume',
    )
    preload = getattr(self.strategy, 'set_backtest_data', None)
    if callable(preload):
      try:
        preload(
            ohlc=self._ohlc_source,
            data=self.data,
            symbol=self.symbol,
        )
      except TypeError:
        preload(self._ohlc_source)

  def run(self) -> List[ExecutedOrder]:
    """Execute the simulation and return closed orders."""
    iterator = self.data.iterrows()
    if self._show_progress:
      iterator = tqdm(
          iterator, total=len(self.data), desc='Simulating', unit='bar'
      )
    for idx, (timestamp, row) in enumerate(iterator, start=1):
      now = timestamp.to_pydatetime()
      self.mt_client.set_now(now)
      self.mt_client.evaluate_positions(row, now)
      ohlc = OHLCView(self._ohlc_source, self.data, idx)
      possible_order = self.strategy.indicator(ohlc, self.symbol, now)
      if possible_order and self.strategy.check_order_viability(possible_order):
        self.mt_client.create_new_order(possible_order)
    if self._show_progress:
      tqdm.write('Simulation completed')
    return self.mt_client.closed_trades


def _load_data(
    csv_path: Path,
    start_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
  df = pd.read_csv(csv_path, parse_dates=['datetime'])
  df = df.set_index('datetime').sort_index()
  df = df[['open', 'high', 'low', 'close', 'volume']]
  if start_date is not None:
    start_ts = pd.Timestamp(start_date)
    index_tz = df.index.tz
    if index_tz is not None:
      if start_ts.tz is None:
        start_ts = start_ts.tz_localize(index_tz)
      else:
        start_ts = start_ts.tz_convert(index_tz)
    elif start_ts.tz is not None:
      start_ts = start_ts.tz_localize(None)
    df = df[df.index >= start_ts]
    if df.empty:
      raise ValueError(
          f'No candles available on or after the requested start date {start_ts.date()}.'
      )
  return df


def _load_strategy(
    module_path: str,
    class_name: str,
    mt_client: SimulatedMTClient,
) -> Strategy:
  module = importlib.import_module(module_path)
  strategy_cls: Type[Strategy] = getattr(module, class_name)
  return strategy_cls(mt_client)


def _summarize(
    trades: Iterable[ExecutedOrder],
    symbol: str,
) -> None:
  trades = list(trades)
  if not trades:
    print('No trades executed.')
    return
  wins = sum(1 for trade in trades if trade.result == 'take_profit')
  losses = sum(1 for trade in trades if trade.result == 'stop_loss')
  net = sum(trade.pnl for trade in trades)
  if symbol.upper() == 'SP500':
    try:
      net = _convert_sp500_net_to_eur(trades)
    except Exception as exc:  # pragma: no cover - defensive
      LOGGER.warning('Could not convert SP500 net to EUR: %s', exc)
  print(
      f'Trades: {len(trades)} | Wins: {wins} | Losses: {losses} | '
      f'Net: {net:.2f} EUR'
  )
  # for trade in trades:
  #   side = 'BUY' if trade.order.order_type.buy else 'SELL'
  #   print(
  #       f'{trade.entry_time.isoformat()} {side} -> {trade.exit_time.isoformat()} '
  #       f'{trade.result.upper()} pnl={trade.pnl:.2f}',
  #   )


def _convert_sp500_net_to_eur(trades: Iterable[ExecutedOrder]) -> float:
  """Convert SP500 PnL (assumed USD) into EUR using yearly EUR/USD averages."""
  yearly_rates: dict[int, float] = {}
  total_eur = 0.0
  for trade in trades:
    exit_time = trade.exit_time
    year = exit_time.year
    rate = yearly_rates.get(year)
    if rate is None:
      rate = _fetch_average_eurusd_rate(year)
      yearly_rates[year] = rate
    total_eur += trade.pnl / rate
  return total_eur


def _fetch_average_eurusd_rate(year: int) -> float:
  """Fetch the average EUR/USD close price for a given year."""
  start = datetime(year, 1, 1)
  end = datetime(year + 1, 1, 1)
  data = yf.download(
      'EURUSD=X',
      start=start,
      end=end,
      progress=False,
      auto_adjust=False,
  )
  closes = data.get('Close')
  if closes is None:
    raise ValueError('No Close column found in EURUSD data.')
  closes = closes.dropna()
  if closes.empty:
    raise ValueError(f'No EUR/USD data available for year {year}.')
  mean_value = closes.mean()
  return float(mean_value.iloc[0]) if hasattr(mean_value, 'iloc') else float(mean_value)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Simple OHLC strategy simulator')
  parser.add_argument(
      '--strategy-module',
      default='sorul_tradingbot.strategy.private.volume_12',
      help='Dotted path to the strategy module.',
  )
  parser.add_argument(
      '--strategy-class',
      default='Volume',
      help='Strategy class name to instantiate.',
  )
  parser.add_argument(
      '--data-file',
      default='sorul_tradingbot/strategy/simulator/data/sp500_5m_test.csv',
      help='CSV file with bars data.',
  )
  parser.add_argument(
      '--symbol',
      default='SP500',
      help='Symbol to pass into the strategy.',
  )
  parser.add_argument(
      '--no-progress',
    action='store_true',
      help='Disable progress bar output.',
  )
  parser.add_argument(
      '--export-file',
      default=
      'sorul_tradingbot/strategy/simulator/outputs/simulation_payload.pkl',
      help=(
          'Optional path to persist candles and trades as a pickle payload '
          'for notebook exploration.'
      ),
  )
  parser.add_argument(
      '--start-date',
      help='Start date in YYYY-MM-DD format to begin the simulation.',
  )
  return parser.parse_args(argv)


def _orders_to_dataframe(
    trades: Iterable[ExecutedOrder],
    active_orders: Iterable[ActiveOrder],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Convert executed and active orders to DataFrames ready for export."""
  executed_rows = []
  for trade in trades:
    order = trade.order
    side = 'BUY' if order.order_type.buy else 'SELL'
    executed_rows.append({
        'ticket': order.ticket,
        'symbol': order.symbol,
        'side': side,
        'entry_time': trade.entry_time,
        'exit_time': trade.exit_time,
        'entry_price': trade.entry_price,
        'exit_price': trade.exit_price,
        'take_profit': order.take_profit,
        'stop_loss': order.stop_loss,
        'result': trade.result,
        'pnl': trade.pnl,
    })
  executed_df = pd.DataFrame(executed_rows)
  active_rows = []
  for active in active_orders:
    order = active.order
    side = 'BUY' if order.order_type.buy else 'SELL'
    active_rows.append({
        'ticket': order.ticket,
        'symbol': order.symbol,
        'side': side,
        'entry_time': active.entry_time,
        'entry_price': active.entry_price,
        'take_profit': order.take_profit,
        'stop_loss': order.stop_loss,
    })
  active_df = pd.DataFrame(active_rows)
  return executed_df, active_df


def _export_results(
    export_path: Path,
    candles: pd.DataFrame,
    trades: Iterable[ExecutedOrder],
    active_orders: Iterable[ActiveOrder],
) -> None:
  """Persist simulation payload for later inspection in notebooks."""
  export_path = export_path.expanduser()
  export_path.parent.mkdir(parents=True, exist_ok=True)
  candles_df = candles.reset_index().rename(columns={'index': 'datetime'})
  executed_df, active_df = _orders_to_dataframe(trades, active_orders)
  payload = {
      'candles': candles_df,
      'closed_trades': executed_df,
      'open_trades': active_df,
  }
  with export_path.open('wb') as file_handle:
    pickle.dump(payload, file_handle)
  print(f'Exported simulation payload to {export_path}')


def main(argv: Optional[List[str]] = None) -> None:
  logging.basicConfig(
      level=logging.WARNING, format='[%(levelname)s] %(message)s'
  )
  args = parse_args(argv)
  start_date = None
  if args.start_date:
    try:
      start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    except ValueError as exc:
      raise ValueError(
          'Invalid start date format. Use YYYY-MM-DD.'
      ) from exc
  csv_path = Path(args.data_file)
  if not csv_path.exists():
    raise FileNotFoundError(f'Data file not found: {csv_path}')
  data = _load_data(csv_path, start_date)
  mt_client = SimulatedMTClient()
  strategy = _load_strategy(
      args.strategy_module, args.strategy_class, mt_client
  )
  simulator = StrategySimulator(
      strategy,
      data,
      args.symbol,
      mt_client,
      show_progress=not args.no_progress,
  )
  trades = simulator.run()
  _summarize(trades, args.symbol)
  export_file = getattr(args, 'export_file', None)
  if export_file:
    _export_results(
        Path(export_file),
        data,
        trades,
        mt_client.get_active_orders(),
    )


if __name__ == '__main__':
  main()
