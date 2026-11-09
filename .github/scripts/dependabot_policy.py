"""Fail-closed policy for trusted, development-only Dependabot updates."""

import argparse
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import tomllib

MAX_INPUT_BYTES = 2_000_000
BOT_LOGIN = 'dependabot[bot]'
BASE_BRANCH = 'master'
UPDATE_TYPES = {
  'version-update:semver-major',
  'version-update:semver-minor',
  'version-update:semver-patch',
}
SPEC_CHARACTERS = re.compile(r'^[A-Za-z0-9.*+<>=!~^, -]+$')
VERSION_PATTERN = re.compile(
  r'^[vV]?(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$'
)


class Rejected(Exception):
  """A static policy rejection code safe to display in a workflow log."""


def main() -> int:
  """Dispatch bounded policy validation or installed-version verification."""
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest='command', required=True)
  validate = subparsers.add_parser('validate')
  for name in (
    'event',
    'pull-request',
    'files',
    'commits',
    'metadata',
    'base-pyproject',
    'head-pyproject',
    'base-lock',
    'head-lock',
    'output',
  ):
    validate.add_argument(f'--{name}', required=True)
  validate.add_argument('--opt-in', required=True)
  validate.add_argument('--metadata-verified', required=True)
  subparsers.add_parser('verify-installed')
  args = parser.parse_args()
  if args.command == 'validate':
    return _validate_command(args)
  return _verify_installed_command()


def _validate_command(args: argparse.Namespace) -> int:
  try:
    candidate = {
      'enabled': args.opt_in == 'v2',
      'auto_merge_cleared': True,
      'metadata_verified': args.metadata_verified == 'success',
      'event': _read_json(args.event),
      'pr': _read_json(args.pull_request),
      'files': _read_json(args.files),
      'commits': _read_json(args.commits),
      'metadata': _read_json(args.metadata),
      'base_pyproject': _read_toml(args.base_pyproject),
      'head_pyproject': _read_toml(args.head_pyproject),
      'base_lock': _read_toml(args.base_lock),
      'head_lock': _read_toml(args.head_lock),
    }
    eligible, reason = evaluate_candidate(candidate)
  except (OSError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError):
    eligible, reason = False, 'invalid_or_ambiguous_input'
  _write_output(args.output, eligible, reason)
  sys.stdout.write(f'{reason}\n')
  return 0


def _verify_installed_command() -> int:
  failure = None
  try:
    pyproject = _read_toml('pyproject.toml')
    lock = _read_toml('poetry.lock')
    installed = {}
    for name in _dev_dependencies(pyproject):
      try:
        installed[_canonical_name(name)] = importlib.metadata.version(name)
      except importlib.metadata.PackageNotFoundError:
        failure = 'direct_dev_package_not_installed'
        break
    valid, reason = verify_installed(pyproject, lock, installed)
    if failure is None and not valid:
      failure = reason
  except (OSError, ValueError, KeyError, TypeError, tomllib.TOMLDecodeError):
    failure = 'invalid_or_ambiguous_input'
  if failure is not None:
    return _print_failure(failure)
  sys.stdout.write(f'{reason}\n')
  return 0


def _print_failure(reason: str) -> int:
  sys.stderr.write(f'{reason}\n')
  return 1


def _write_output(path: str, eligible: bool, reason: str) -> None:
  with Path(path).open('a', encoding='utf-8') as output:
    output.write(f'eligible={str(eligible).lower()}\n')
    output.write(f'reason={reason}\n')


def evaluate_candidate(candidate: dict) -> tuple[bool, str]:
  """Return whether the candidate may receive a native auto-merge request."""
  try:
    _require(candidate.get('enabled') is True, 'opt_in_disabled')
    _require(candidate.get('auto_merge_cleared') is True, 'pending_not_cleared')
    _require(candidate.get('metadata_verified') is True, 'metadata_unverified')
    event = _mapping(candidate.get('event'))
    pull_request = _mapping(event.get('pull_request'))
    repository = _mapping(event.get('repository'))
    pr = _mapping(candidate.get('pr'))
    _validate_pull_request(repository, pull_request, pr)

    files = candidate.get('files')
    _validate_files(files, pr)
    _validate_commit(candidate.get('commits'), pr)
    metadata = _validate_metadata(candidate.get('metadata'))
    base_pyproject = _mapping(candidate.get('base_pyproject'))
    head_pyproject = _mapping(candidate.get('head_pyproject'))
    changed_dependencies, direct_dev_names = _validate_manifest_metadata(
      metadata,
      base_pyproject,
      head_pyproject,
      files,
    )
    metadata_names = {item['name'] for item in metadata}

    base_lock = _mapping(candidate.get('base_lock'))
    head_lock = _mapping(candidate.get('head_lock'))
    _validate_lock_metadata(base_lock, head_lock, bool(changed_dependencies))
    base_main, base_dev = _lock_indexes(base_lock)
    head_main, head_dev = _lock_indexes(head_lock)
    _require(base_main == head_main, 'production_lock_changed')
    _validate_reported_direct_dev_updates(
      direct_dev_names,
      metadata_names,
      base_dev,
      head_dev,
    )
    _validate_updated_packages(metadata, base_dev, head_dev)
    return True, 'eligible'
  except Rejected as error:
    return False, str(error)
  except (
    KeyError,
    TypeError,
    ValueError,
    AttributeError,
    tomllib.TOMLDecodeError,
  ):
    return False, 'invalid_or_ambiguous_input'


def _validate_manifest_metadata(
  metadata: list[dict],
  base_pyproject: dict,
  head_pyproject: dict,
  files: list[dict],
) -> tuple[set[str], set[str]]:
  changed = _validate_pyproject(base_pyproject, head_pyproject)
  paths = {item['filename'] for item in files}
  _require(
    ('pyproject.toml' in paths) == bool(changed),
    'manifest_change_mismatch',
  )
  metadata_names = {item['name'] for item in metadata}
  _require(changed <= metadata_names, 'metadata_manifest_mismatch')
  direct_names = set(_normalized_keys(_dev_dependencies(head_pyproject)))
  _require(metadata_names <= direct_names, 'metadata_not_direct_dev')
  return changed, direct_names


def _validate_reported_direct_dev_updates(
  direct_names: set[str],
  metadata_names: set[str],
  base_dev: dict[str, list[dict]],
  head_dev: dict[str, list[dict]],
) -> None:
  for name in direct_names - metadata_names:
    old_versions = [item['version'] for item in base_dev.get(name, [])]
    new_versions = [item['version'] for item in head_dev.get(name, [])]
    _require(old_versions == new_versions, 'unreported_direct_dev_update')


def verify_installed(
  pyproject: dict,
  lock: dict,
  installed_versions: dict[str, str],
) -> tuple[bool, str]:
  """Require each direct dev distribution to match its unique locked version."""
  try:
    dependencies = _dev_dependencies(pyproject)
    _, dev_packages = _lock_indexes(lock)
    for name in dependencies:
      package = dev_packages.get(_canonical_name(name), [])
      _require(len(package) == 1, 'ambiguous_locked_dev_version')
      installed = installed_versions.get(_canonical_name(name))
      _require(installed is not None, 'direct_dev_package_not_installed')
      _require(installed == package[0]['version'], 'installed_version_mismatch')
    return True, 'installed_versions_match'
  except Rejected as error:
    return False, str(error)
  except (KeyError, TypeError, ValueError, AttributeError):
    return False, 'invalid_or_ambiguous_input'


def _validate_pull_request(
  repository: dict,
  event_pr: dict,
  pr: dict,
) -> None:
  repo_name = repository.get('full_name')
  _require(
    isinstance(repo_name, str)
    and bool(re.fullmatch(r'[^/]+/[^/]+', repo_name)),
    'unexpected_repository',
  )
  _require(
    repository.get('default_branch') == BASE_BRANCH,
    'unexpected_default_branch',
  )
  _require(
    event_pr.get('user', {}).get('login') == BOT_LOGIN,
    'unexpected_pr_author',
  )
  _require(pr.get('user', {}).get('login') == BOT_LOGIN, 'unexpected_pr_author')
  _require(
    pr.get('state') == 'open' and pr.get('draft') is False,
    'pr_not_ready',
  )
  _require(type(pr.get('number')) is int, 'invalid_pr_number')
  _require(pr.get('number') == event_pr.get('number'), 'stale_pr_number')

  event_head = _mapping(event_pr.get('head'))
  event_base = _mapping(event_pr.get('base'))
  current_head = _mapping(pr.get('head'))
  current_base = _mapping(pr.get('base'))
  event_head_repo = _mapping(event_head.get('repo'))
  current_head_repo = _mapping(current_head.get('repo'))
  _require(event_head_repo.get('full_name') == repo_name, 'fork_head')
  _require(current_head_repo.get('full_name') == repo_name, 'fork_head')
  _require(event_base.get('ref') == BASE_BRANCH, 'unexpected_base_branch')
  _require(current_base.get('ref') == BASE_BRANCH, 'unexpected_base_branch')

  head_sha = event_head.get('sha')
  base_sha = event_base.get('sha')
  _require(_is_sha(head_sha) and _is_sha(base_sha), 'invalid_event_sha')
  _require(current_head.get('sha') == head_sha, 'stale_head_sha')
  _require(current_base.get('sha') == base_sha, 'stale_base_sha')
  _require(
    type(pr.get('commits')) is int and pr['commits'] == 1,
    'multiple_commits',
  )


def _validate_files(files: object, pr: dict) -> None:
  _require(isinstance(files, list) and files, 'missing_changed_files')
  changed_files = pr.get('changed_files')
  _require(type(changed_files) is int, 'invalid_changed_file_count')
  _require(
    1 <= changed_files <= 2 and len(files) == changed_files,
    'changed_file_count',
  )
  filenames = []
  for item in files:
    file_info = _mapping(item)
    filename = file_info.get('filename')
    _require(
      filename in {'poetry.lock', 'pyproject.toml'}
      and file_info.get('status') == 'modified'
      and 'previous_filename' not in file_info,
      'out_of_scope_file',
    )
    filenames.append(filename)
  _require(len(set(filenames)) == len(filenames), 'duplicate_changed_file')
  _require('poetry.lock' in filenames, 'lock_not_changed')


def _validate_commit(commits: object, pr: dict) -> None:
  _require(isinstance(commits, list) and len(commits) == 1, 'multiple_commits')
  commit = _mapping(commits[0])
  author = _mapping(commit.get('author'))
  details = _mapping(commit.get('commit'))
  verification = _mapping(details.get('verification'))
  _require(commit.get('sha') == pr['head']['sha'], 'stale_commit_sha')
  _require(author.get('login') == BOT_LOGIN, 'untrusted_commit_author')
  _require(verification.get('verified') is True, 'unverified_commit')


def _validate_metadata(metadata: object) -> list[dict]:
  _require(isinstance(metadata, list) and metadata, 'unknown_metadata')
  result = []
  for item in metadata:
    entry = _mapping(item)
    name = entry.get('dependencyName')
    canonical = _canonical_name(name) if isinstance(name, str) else ''
    _require(bool(canonical), 'unknown_metadata')
    _require(
      entry.get('dependencyType') == 'direct:development',
      'not_direct_dev',
    )
    _require(entry.get('packageEcosystem') == 'pip', 'wrong_ecosystem')
    _require(entry.get('directory') == '/', 'wrong_directory')
    _require(entry.get('targetBranch') == BASE_BRANCH, 'wrong_target_branch')
    _require(entry.get('updateType') in UPDATE_TYPES, 'unknown_update_type')
    _require(
      isinstance(entry.get('prevVersion'), str)
      and bool(entry['prevVersion'].strip())
      and isinstance(entry.get('newVersion'), str)
      and bool(entry['newVersion'].strip()),
      'unknown_update_version',
    )
    result.append({**entry, 'name': canonical})
  _require(
    len({entry['name'] for entry in result}) == len(result),
    'duplicate_metadata',
  )
  return result


def _validate_pyproject(base: dict, head: dict) -> set[str]:
  base_deps = _dev_dependencies(base)
  head_deps = _dev_dependencies(head)
  base_names = _normalized_keys(base_deps)
  head_names = _normalized_keys(head_deps)
  _require(
    base_names.keys() == head_names.keys(),
    'dev_dependency_set_changed',
  )

  base_poetry = base['tool']['poetry']
  head_poetry = head['tool']['poetry']
  _require(
    not base_poetry.get('source') and not head_poetry.get('source'),
    'custom_source',
  )
  _require(
    _without_dev_dependencies(base) == _without_dev_dependencies(head),
    'non_dev_manifest_changed',
  )
  changed = {
    name
    for name in base_names
    if base_deps[base_names[name]] != head_deps[head_names[name]]
  }
  for name in changed:
    _require(
      _is_plain_version_spec(base_deps[base_names[name]])
      and _is_plain_version_spec(head_deps[head_names[name]]),
      'source_extra_or_marker_change',
    )
  return changed


def _validate_lock_metadata(
  base: dict,
  head: dict,
  manifest_changed: bool,
) -> None:
  _require(set(base) == {'metadata', 'package'}, 'invalid_base_lock')
  _require(set(head) == {'metadata', 'package'}, 'invalid_head_lock')
  base_meta = _mapping(base.get('metadata'))
  head_meta = _mapping(head.get('metadata'))
  _require(set(base_meta) == set(head_meta), 'lock_metadata_changed')
  _require(
    isinstance(base_meta.get('content-hash'), str)
    and bool(base_meta['content-hash'])
    and isinstance(head_meta.get('content-hash'), str)
    and bool(head_meta['content-hash']),
    'missing_lock_content_hash',
  )
  base_meta_without_hash = {
    key: value for key, value in base_meta.items() if key != 'content-hash'
  }
  head_meta_without_hash = {
    key: value for key, value in head_meta.items() if key != 'content-hash'
  }
  _require(
    base_meta_without_hash == head_meta_without_hash,
    'lock_format_changed',
  )
  _require(
    (base_meta['content-hash'] != head_meta['content-hash'])
    is manifest_changed,
    'lock_manifest_hash_mismatch',
  )


def _lock_indexes(lock: dict) -> tuple[dict, dict]:
  packages = lock.get('package')
  _require(isinstance(packages, list) and packages, 'invalid_lock_packages')
  main = {}
  dev: dict[str, list[dict]] = {}
  for item in packages:
    package = _mapping(item)
    name = package.get('name')
    version = package.get('version')
    groups = package.get('groups')
    _require(
      isinstance(name, str) and bool(name)
      and isinstance(version, str) and bool(version)
      and isinstance(groups, list)
      and groups,
      'invalid_lock_package',
    )
    _require('source' not in package, 'lock_source_found')
    dependencies = package.get('dependencies', {})
    _require(isinstance(dependencies, dict), 'invalid_lock_dependencies')
    _require(
      not any(
        isinstance(dependency, dict) and _contains_source_record(dependency)
        for dependency in dependencies.values()
      ),
      'lock_source_found',
    )
    normalized = _canonical_name(name)
    _require(
      all(group in {'main', 'dev'} for group in groups),
      'unknown_lock_group',
    )
    if 'main' in groups:
      _require(normalized not in main, 'ambiguous_main_package')
      record = deepcopy(package)
      record.pop('groups')
      main[normalized] = record
    if 'dev' in groups:
      dev.setdefault(normalized, []).append(package)
  return main, dev


def _validate_updated_packages(
  metadata: list[dict],
  base_dev: dict[str, list[dict]],
  head_dev: dict[str, list[dict]],
) -> None:
  for entry in metadata:
    name = entry['name']
    old_packages = base_dev.get(name, [])
    new_packages = head_dev.get(name, [])
    _require(
      len(old_packages) == 1 and len(new_packages) == 1,
      'ambiguous_locked_dev_version',
    )
    old_version = old_packages[0]['version']
    new_version = new_packages[0]['version']
    _require(old_version != new_version, 'locked_candidate_unchanged')
    old_parts = _version_parts(old_version)
    new_parts = _version_parts(new_version)
    _require(
      old_parts is not None and new_parts is not None,
      'unsupported_locked_version',
    )
    _require(old_parts != new_parts, 'unsupported_locked_version')
    _require(new_parts > old_parts, 'locked_version_not_upgrade')
    actual_type = _update_type(old_parts, new_parts)
    _require(entry['updateType'] == actual_type, 'update_type_lock_mismatch')

    metadata_old = _version_parts(entry['prevVersion'])
    metadata_new = _version_parts(entry['newVersion'])
    if metadata_old is not None:
      _require(metadata_old == old_parts, 'metadata_previous_version_mismatch')
    if metadata_new is not None:
      _require(metadata_new == new_parts, 'metadata_new_version_mismatch')


def _dev_dependencies(pyproject: dict) -> dict:
  try:
    dependencies = pyproject['tool']['poetry']['group']['dev']['dependencies']
  except (KeyError, TypeError):
    raise Rejected('missing_dev_dependencies') from None
  return _mapping(dependencies)


def _normalized_keys(dependencies: dict) -> dict[str, str]:
  result = {}
  for name in dependencies:
    _require(isinstance(name, str) and bool(name), 'invalid_dependency_name')
    normalized = _canonical_name(name)
    _require(normalized not in result, 'ambiguous_dependency_name')
    result[normalized] = name
  return result


def _without_dev_dependencies(pyproject: dict) -> dict:
  result = deepcopy(pyproject)
  try:
    del result['tool']['poetry']['group']['dev']['dependencies']
  except (KeyError, TypeError):
    raise Rejected('missing_dev_dependencies') from None
  return result


def _is_plain_version_spec(value: object) -> bool:
  return (
    isinstance(value, str)
    and bool(value.strip())
    and bool(SPEC_CHARACTERS.fullmatch(value.strip()))
    and any(character.isdigit() or character == '*' for character in value)
  )


def _contains_source_record(value: object) -> bool:
  forbidden = {'source', 'url', 'git', 'directory', 'path'}
  if isinstance(value, dict):
    return any(
      key in forbidden or _contains_source_record(child)
      for key, child in value.items()
    )
  if isinstance(value, list):
    return any(_contains_source_record(child) for child in value)
  return False


def _canonical_name(name: str) -> str:
  return re.sub(r'[-_.]+', '-', name).lower()


def _version_parts(version: str) -> tuple[int, int, int] | None:
  match = VERSION_PATTERN.fullmatch(version)
  if match is None:
    return None
  return tuple(int(value or 0) for value in match.groups())


def _update_type(old: tuple[int, int, int], new: tuple[int, int, int]) -> str:
  if old[0] != new[0]:
    return 'version-update:semver-major'
  if old[1] != new[1]:
    return 'version-update:semver-minor'
  return 'version-update:semver-patch'


def _is_sha(value: object) -> bool:
  return isinstance(value, str) and bool(re.fullmatch(r'[0-9a-f]{40}', value))


def _mapping(value: object) -> dict:
  _require(isinstance(value, dict), 'invalid_mapping')
  return value


def _require(condition: bool, reason: str) -> None:
  if not condition:
    raise Rejected(reason)


def _read_json(path: str) -> object:
  contents = _read_bounded(path)
  return json.loads(contents)


def _read_toml(path: str) -> dict:
  contents = _read_bounded(path)
  return tomllib.loads(contents)


def _read_bounded(path: str) -> str:
  data = Path(path).read_bytes()
  if len(data) > MAX_INPUT_BYTES:
    raise ValueError('input too large')
  return data.decode('utf-8')


if __name__ == '__main__':
  raise SystemExit(main())
