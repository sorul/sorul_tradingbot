"""Regression coverage for published dependency declarations."""

# Pytest rewrites assertions in test modules; SCS108 targets production asserts
# that can disappear under Python optimization, so suppress it only here.

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_tradeo_is_a_published_exact_dependency() -> None:
  """Require the canonical manifest to use the exact public release."""
  pyproject = tomllib.loads((ROOT / 'pyproject.toml').read_text())
  tradeo = pyproject['tool']['poetry']['dependencies']['tradeo']

  assert tradeo == '==0.27.0'  # noqa: SCS108


def test_requirements_exports_pin_tradeo_without_local_sources() -> None:
  """Require exported requirements to avoid machine-local sources."""
  for path in ('requirements.txt', 'requirements_dev.txt'):
    lines = (ROOT / path).read_text().splitlines()
    tradeo_lines = [line for line in lines if line.lower().startswith('tradeo')]

    assert len(tradeo_lines) == 1  # noqa: SCS108
    assert tradeo_lines[0].startswith('tradeo==0.27.0')  # noqa: SCS108
    assert not any(line.startswith('-e ') for line in lines)  # noqa: SCS108
    assert not any('file://' in line for line in lines)  # noqa: SCS108
