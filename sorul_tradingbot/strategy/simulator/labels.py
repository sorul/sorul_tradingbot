"""Helpers to annotate simulated orders with custom labels."""
from __future__ import annotations

from typing import Optional

from tradeo.order import Order

_LABEL_ATTR = '_tb_simulation_label'


def set_simulation_label(order: Order, label: str) -> Order:
  """Attach a custom label to an order for simulator exports."""
  setattr(order, _LABEL_ATTR, label)
  return order


def get_simulation_label(order: Order) -> Optional[str]:
  """Retrieve the simulator label previously assigned to an order."""
  value = getattr(order, _LABEL_ATTR, None)
  if value is None:
    return None
  return str(value)


__all__ = ['set_simulation_label', 'get_simulation_label']
