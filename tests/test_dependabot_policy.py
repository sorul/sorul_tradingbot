"""Fail-closed policy checks for automatic development updates."""

from copy import deepcopy
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.github/scripts'))
from dependabot_policy import evaluate_candidate, verify_installed  # noqa: E402


def test_accepts_major_version_range_widening_with_locked_candidate() -> None:
  """Allow a major constraint widening backed by the candidate lock."""
  candidate = _candidate()
  candidate['head_pyproject']['tool']['poetry']['group']['dev']['dependencies'][
    'dev-lib'
  ] = '>=2.5,<4'
  candidate['head_lock']['metadata']['content-hash'] = 'head-hash'
  _set_locked_version(candidate['head_lock'], 'dev-lib', '3.1.0')
  candidate['metadata'][0]['updateType'] = 'version-update:semver-major'
  candidate['metadata'][0]['newVersion'] = '>=2.5,<4'
  candidate['files'] = [
    {'filename': 'pyproject.toml', 'status': 'modified'},
    {'filename': 'poetry.lock', 'status': 'modified'},
  ]
  candidate['pr']['changed_files'] = 2

  assert evaluate_candidate(candidate) == (True, 'eligible')  # noqa: SCS108


def test_accepts_grouped_dev_minor_and_major_updates() -> None:
  """Allow mixed semver levels when every grouped update is dev-only."""
  candidate = _candidate()
  candidate['base_pyproject']['tool']['poetry']['group']['dev'][
    'dependencies'
  ]['dev-helper'] = '^1.0'
  candidate['head_pyproject'] = deepcopy(candidate['base_pyproject'])
  candidate['base_lock']['package'].append(
    _package('dev-helper', '1.0.0', ['dev']),
  )
  candidate['head_lock']['package'].append(
    _package('dev-helper', '1.1.0', ['dev']),
  )
  _set_locked_version(candidate['head_lock'], 'dev-lib', '3.1.0')
  candidate['head_pyproject']['tool']['poetry']['group']['dev'][
    'dependencies'
  ]['dev-lib'] = '>=2.5,<4'
  candidate['head_pyproject']['tool']['poetry']['group']['dev'][
    'dependencies'
  ]['dev-helper'] = '^1.1'
  candidate['head_lock']['metadata']['content-hash'] = 'head-hash'
  candidate['files'] = [
    {'filename': 'pyproject.toml', 'status': 'modified'},
    {'filename': 'poetry.lock', 'status': 'modified'},
  ]
  candidate['pr']['changed_files'] = 2
  candidate['metadata'] = [
    _metadata('dev-lib', 'version-update:semver-major', '>=2.5,<4'),
    _metadata('dev-helper', 'version-update:semver-minor', '^1.1'),
  ]
  candidate['metadata'][1]['prevVersion'] = '1.0.0'

  assert evaluate_candidate(candidate) == (True, 'eligible')  # noqa: SCS108


def test_accepts_lock_only_direct_dev_update() -> None:
  """Allow a transitive lock change for a direct development update."""
  candidate = _candidate()
  _set_locked_version(candidate['head_lock'], 'dev-lib', '2.10.0')
  candidate['metadata'][0]['updateType'] = 'version-update:semver-minor'
  candidate['metadata'][0]['newVersion'] = '2.10.0'

  assert evaluate_candidate(candidate) == (True, 'eligible')  # noqa: SCS108


def test_rejects_downgrade_labeled_as_major_update() -> None:
  """Do not interpret a major-version downgrade as an eligible upgrade."""
  candidate = _candidate()
  _set_locked_version(candidate['head_lock'], 'dev-lib', '1.0.0')
  candidate['metadata'][0]['updateType'] = 'version-update:semver-major'
  candidate['metadata'][0]['newVersion'] = '1.0.0'

  eligible, reason = evaluate_candidate(candidate)

  assert not eligible  # noqa: SCS108
  assert reason == 'locked_version_not_upgrade'  # noqa: SCS108


@pytest.mark.parametrize(
  'change',
  [
    'production_metadata',
    'mixed_file',
    'missing_lock_change',
    'runtime_manifest_change',
    'source_spec',
    'marker_spec',
    'production_lock_drift',
    'lock_source',
    'unknown_metadata',
    'multiple_commits',
    'unsigned_commit',
    'stale_head',
    'stale_base',
    'pending_auto_not_cleared',
    'legacy_opt_in',
    'ambiguous_lock_version',
  ],
)
def test_rejects_unsafe_or_ambiguous_candidate(  # noqa: CFQ001
  change: str,
) -> None:
  """Reject production, mixed, unsigned, stale, and unprovable changes."""
  candidate = _candidate()

  if change == 'production_metadata':
    candidate['metadata'][0]['dependencyType'] = 'direct:production'
  elif change == 'mixed_file':
    candidate['files'].append(
      {'filename': 'requirements.txt', 'status': 'modified'},
    )
    candidate['pr']['changed_files'] = 2
  elif change == 'missing_lock_change':
    candidate['files'] = [
      {'filename': 'pyproject.toml', 'status': 'modified'},
    ]
    candidate['pr']['changed_files'] = 1
    candidate['head_pyproject']['tool']['poetry']['group']['dev'][
      'dependencies'
    ]['dev-lib'] = '>=2.5,<4'
    candidate['head_lock']['metadata']['content-hash'] = 'head-hash'
  elif change == 'runtime_manifest_change':
    candidate['head_pyproject']['tool']['poetry']['dependencies'][
      'runtime-lib'
    ] = '^2.0'
  elif change == 'source_spec':
    candidate['head_pyproject']['tool']['poetry']['group']['dev'][
      'dependencies'
    ]['dev-lib'] = {'path': '../local-package'}
    candidate['head_lock']['metadata']['content-hash'] = 'head-hash'
    candidate['files'] = [
      {'filename': 'pyproject.toml', 'status': 'modified'},
      {'filename': 'poetry.lock', 'status': 'modified'},
    ]
    candidate['pr']['changed_files'] = 2
  elif change == 'marker_spec':
    candidate['head_pyproject']['tool']['poetry']['group']['dev'][
      'dependencies'
    ]['dev-lib'] = '>=2.5,<4; python_version >= "3.11"'
    candidate['head_lock']['metadata']['content-hash'] = 'head-hash'
    candidate['files'] = [
      {'filename': 'pyproject.toml', 'status': 'modified'},
      {'filename': 'poetry.lock', 'status': 'modified'},
    ]
    candidate['pr']['changed_files'] = 2
  elif change == 'production_lock_drift':
    _set_locked_version(candidate['head_lock'], 'runtime-lib', '2.0.0')
  elif change == 'lock_source':
    candidate['head_lock']['package'][1]['source'] = {
      'type': 'legacy',
      'url': 'https://example.invalid/simple',
    }
  elif change == 'unknown_metadata':
    candidate['metadata'] = []
  elif change == 'multiple_commits':
    candidate['pr']['commits'] = 2
    candidate['commits'].append(deepcopy(candidate['commits'][0]))
  elif change == 'unsigned_commit':
    candidate['commits'][0]['commit']['verification']['verified'] = False
  elif change == 'stale_head':
    candidate['pr']['head']['sha'] = 'b' * 40
  elif change == 'stale_base':
    candidate['pr']['base']['sha'] = 'b' * 40
  elif change == 'pending_auto_not_cleared':
    candidate['auto_merge_cleared'] = False
  elif change == 'legacy_opt_in':
    candidate['enabled'] = False
  elif change == 'ambiguous_lock_version':
    candidate['head_lock']['package'].append(
      _package('dev-lib', '2.9.0', ['dev']),
    )

  eligible, reason = evaluate_candidate(candidate)
  assert not eligible  # noqa: SCS108
  assert reason != 'eligible'  # noqa: SCS108


def test_rejects_metadata_previous_version_mismatch() -> None:
  """Require metadata's old version to match the existing lock entry."""
  candidate = _candidate()
  candidate['metadata'][0]['prevVersion'] = '1.0.0'

  assert evaluate_candidate(candidate)[0] is False  # noqa: SCS108


def test_rejects_invalid_direct_dev_metadata_entry() -> None:
  """Reject metadata for a non-Python ecosystem."""
  candidate = _candidate()
  candidate['metadata'][0]['packageEcosystem'] = 'npm'

  assert evaluate_candidate(candidate)[0] is False  # noqa: SCS108


def test_verifies_installed_direct_dev_versions_against_lock() -> None:
  """Require installed direct development versions to match the lock."""
  pyproject, lock = _installed_fixture()

  assert verify_installed(  # noqa: SCS108
    pyproject,
    lock,
    {'dev-lib': '3.1.0'},
  ) == (True, 'installed_versions_match')


def test_rejects_installed_version_mismatch() -> None:
  """Reject a local environment that differs from its locked candidate."""
  pyproject, lock = _installed_fixture()

  assert verify_installed(  # noqa: SCS108
    pyproject,
    lock,
    {'dev-lib': '2.9.0'},
  ) == (False, 'installed_version_mismatch')


def _candidate() -> dict:
  sha = 'a' * 40
  return {
    'enabled': True,
    'auto_merge_cleared': True,
    'metadata_verified': True,
    'event': {
      'repository': {
        'full_name': 'owner/repository',
        'default_branch': 'master',
      },
      'pull_request': {
        'number': 7,
        'user': {'login': 'dependabot[bot]'},
        'head': {
          'sha': sha,
          'repo': {'full_name': 'owner/repository'},
        },
        'base': {'ref': 'master', 'sha': 'c' * 40},
      },
    },
    'pr': {
      'number': 7,
      'state': 'open',
      'draft': False,
      'user': {'login': 'dependabot[bot]'},
      'head': {
        'sha': sha,
        'repo': {'full_name': 'owner/repository'},
      },
      'base': {'ref': 'master', 'sha': 'c' * 40},
      'commits': 1,
      'changed_files': 1,
    },
    'files': [{'filename': 'poetry.lock', 'status': 'modified'}],
    'commits': [
      {
        'sha': sha,
        'author': {'login': 'dependabot[bot]'},
        'commit': {'verification': {'verified': True}},
      },
    ],
    'metadata': [_metadata('dev-lib', 'version-update:semver-patch', '2.10.0')],
    'base_pyproject': _pyproject(),
    'head_pyproject': _pyproject(),
    'base_lock': _lock('2.9.0', '1.0.0', 'base-hash'),
    'head_lock': _lock('2.10.0', '1.0.0', 'base-hash'),
  }


def _metadata(name: str, update_type: str, new_version: str) -> dict:
  return {
    'dependencyName': name,
    'dependencyType': 'direct:development',
    'packageEcosystem': 'pip',
    'directory': '/',
    'targetBranch': 'master',
    'updateType': update_type,
    'prevVersion': '2.9.0',
    'newVersion': new_version,
  }


def _installed_fixture() -> tuple[dict, dict]:
  pyproject = _pyproject()
  pyproject['tool']['poetry']['group']['dev']['dependencies'][
    'dev-lib'
  ] = '>=2.5,<4'
  lock = _lock('3.1.0', '1.0.0', 'head-hash')
  return pyproject, lock


def _lock(dev_version: str, runtime_version: str, content_hash: str) -> dict:
  return {
    'package': [
      _package('runtime-lib', runtime_version, ['main']),
      _package('dev-lib', dev_version, ['dev']),
    ],
    'metadata': {
      'lock-version': '2.1',
      'python-versions': '3.11.0',
      'content-hash': content_hash,
    },
  }


def _pyproject() -> dict:
  return {
    'tool': {
      'poetry': {
        'dependencies': {'runtime-lib': '^1.0'},
        'group': {'dev': {'dependencies': {'dev-lib': '^2.5'}}},
      },
    },
  }


def _package(name: str, version: str, groups: list[str]) -> dict:
  return {
    'name': name,
    'version': version,
    'groups': groups,
    'files': [],
    'dependencies': {},
  }


def _set_locked_version(lock: dict, name: str, version: str) -> None:
  for package in lock['package']:
    if package['name'] == name:
      package['version'] = version
      return
  raise AssertionError(f'Package is missing from test lock: {name}')
