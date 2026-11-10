"""Regression coverage for published dependency declarations."""

# Pytest rewrites assertions in test modules; SCS108 targets production asserts
# that can disappear under Python optimization, so suppress it only here.

from pathlib import Path
import re
import tomllib
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_tradeo_is_a_published_exact_dependency() -> None:
  """Require the canonical manifest to use the exact public release."""
  pyproject = tomllib.loads((ROOT / 'pyproject.toml').read_text())
  tradeo = pyproject['tool']['poetry']['dependencies']['tradeo']

  assert tradeo == '==0.27.0'  # noqa: SCS108


def test_production_requirements_pin_tradeo_without_local_sources() -> None:
  """Require exported requirements to avoid machine-local sources."""
  lines = (ROOT / 'requirements.txt').read_text().splitlines()
  tradeo_lines = [line for line in lines if line.lower().startswith('tradeo')]

  assert len(tradeo_lines) == 1  # noqa: SCS108
  assert tradeo_lines[0].startswith('tradeo==0.27.0')  # noqa: SCS108
  assert not any(line.startswith('-e ') for line in lines)  # noqa: SCS108
  assert not any('file://' in line for line in lines)  # noqa: SCS108


def test_production_requirements_match_main_poetry_lock_group() -> None:
  """Keep the production export aligned with main-group lock pins."""
  packages = tomllib.loads((ROOT / 'poetry.lock').read_text())['package']
  main_pins = {
    (_canonical_name(package['name']), str(package['version']))
    for package in packages
    if 'main' in package.get('groups', [])
  }
  assert _requirement_pins('requirements.txt') == main_pins  # noqa: SCS108


def test_dev_export_is_ignored_and_rebuilt_without_relocking() -> None:
  """Keep the local dev export disposable and derive it from the lock."""
  ignore_entries = (ROOT / '.gitignore').read_text().splitlines()
  makefile = (ROOT / 'Makefile').read_text()
  target = makefile.split('dev_requirements:\n', 1)[1].split('\n\n', 1)[0]
  commands = [line.strip() for line in target.splitlines() if line.strip()]

  assert 'poetry.lock' not in ignore_entries  # noqa: SCS108
  assert 'requirements_dev.txt' in ignore_entries  # noqa: SCS108
  assert commands[0] == 'poetry check --lock'  # noqa: SCS108
  assert any(  # noqa: SCS108
    command.startswith('poetry export --with dev') for command in commands
  )
  assert 'poetry lock' not in commands  # noqa: SCS108


def test_dependency_ci_uses_poetry_lock_without_local_dev_export() -> None:
  """Install CI dependencies from the committed lock, not local exports."""
  workflow = (ROOT / '.github/workflows/dependency-ci.yml').read_text()
  install = workflow.split(
    '      - name: Install Poetry and project dependencies\n', 1,
  )[1].split('      - name:', 1)[0]

  assert 'requirements_dev.txt' not in workflow  # noqa: SCS108
  assert 'poetry check --lock' in install  # noqa: SCS108
  assert install.index('poetry check --lock') < install.index(  # noqa: SCS108
    'poetry install --with dev --no-interaction',
  )


def test_dependabot_write_permissions_are_scoped_to_guard_job() -> None:
  """Grant content writes only to the privileged auto-merge guard."""
  workflow = yaml.safe_load(
    (ROOT / '.github/workflows/dependabot-auto-merge.yml').read_text(),
  )

  assert workflow['permissions'] == {  # noqa: SCS108
    'contents': 'read',
    'pull-requests': 'write',
  }
  assert workflow['jobs']['guard']['permissions'] == {  # noqa: SCS108
    'contents': 'write',
    'pull-requests': 'write',
  }
  assert workflow['jobs']['policy']['permissions'] == {}  # noqa: SCS108


def _canonical_name(name: str) -> str:
  return re.sub(r'[-_.]+', '-', name).lower()


def _requirement_pins(path: str) -> set[tuple[str, str]]:
  lines = [
    line.strip()
    for line in (ROOT / path).read_text().splitlines()
    if line.strip() and not line.lstrip().startswith(('#', '-'))
  ]
  pins = set()

  for line in lines:
    match = re.match(
      r'^([A-Za-z0-9][A-Za-z0-9_.-]*)(?:\[[^]]+\])?==([^;\s]+)',
      line,
    )
    assert match is not None  # noqa: SCS108
    pins.add((_canonical_name(match.group(1)), match.group(2)))

  assert len(pins) == len(lines)  # noqa: SCS108
  return pins
