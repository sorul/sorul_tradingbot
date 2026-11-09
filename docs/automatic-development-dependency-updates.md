# Automatic development-dependency updates

This guide describes this repository's Poetry/Dependabot setup and a cautious
pattern to adapt elsewhere. It is not a universal copy-paste recipe. Automated
checks are evidence about a candidate, not proof that a release is benign or
that a trading application is safe to run.

## Current source of truth

| File | Role |
| --- | --- |
| `pyproject.toml` | Declares runtime dependencies under `[tool.poetry.dependencies]` and development tools under `[tool.poetry.group.dev.dependencies]`. |
| `poetry.lock` | Committed Poetry resolution used by CI and Dependabot candidate validation. Runtime and development packages share this lock. |
| `requirements.txt` | Generated production requirements export; it is not the dependency source of truth. |
| `requirements_dev.txt` | Local, ignored export including the dev group. It also includes runtime/transitive dependencies, so it is not a dev-only manifest. |
| `Makefile` | `make dev_requirements` checks the existing lock, then exports the local dev requirements without relocking. `make requirements` runs `poetry lock` before exporting production requirements; it can change the lock and is not a routine export/check command. `make flake8` and `make test` are the CI quality commands. |
| `.gitignore` | Keeps the generated `requirements_dev.txt` local while allowing `poetry.lock` to be versioned. |
| `.github/dependabot.yml` | Weekly root `pip` version updates, capped at five open version-update PRs, with separate development version/security groups covering major, minor, and patch updates. |
| `.github/workflows/dependency-ci.yml` | Unprivileged `pull_request` CI on Python 3.11.0: validates and installs the Poetry lock with dev dependencies, verifies installed direct-dev versions against that lock, then runs `make flake8` and `make test`. It does not overlay `requirements_dev.txt`. |
| `.github/workflows/dependabot-auto-merge.yml` | Privileged `pull_request_target` guard for eligible PRs to `master`. It reads candidate files as data, cancels any previous auto-merge request, and only then may request native auto-merge. |
| `.github/scripts/dependabot_policy.py` | Fail-closed checks for PR identity, metadata, changed files, Poetry manifest/lock deltas, production graph stability, and installed dev versions. |
| `.github/DEPENDABOT.md` | Short maintainer setup and safety notes. GitHub does not read this file; workflow YAML and repository settings control behavior. |

The intended dependency flow is **manifest → committed lock → install/test**.
Use `make dev_requirements` only when a local `requirements_dev.txt` export is
needed; it does not run `poetry lock`. Avoid using the generated export as the
CI overlay or editing it by hand. The production `make requirements` target
does relock, so do not treat it as a harmless export step. The `push_develop`
target also runs `poetry update`; it is not the routine for Dependabot lock
updates.

## What can request auto-merge here

Dependabot's version and security groups are separate. Both are development
only and include major, minor, and patch updates. Grouping forms a PR batch; it
does not make its contents trusted or compatible. Alerts and normal GitHub
notifications remain separate: this workflow neither dismisses alerts nor
suppresses notifications.

An auto-merge request is considered only when all of these conditions hold:

- The repository's default branch and PR base are `master`; the PR author is
  `dependabot[bot]`, and its head repository is this same repository.
- `DEPENDABOT_AUTOMERGE_ENABLED` is exactly `v2`. An older value such as
  `true` does not opt in.
- `dependabot/fetch-metadata@v3` succeeds and every item in its
  `updated-dependencies-json` is a root (`/`) `pip` direct-development update
  classified as major, minor, or patch.
- The PR has exactly one verified Dependabot commit. The metadata action
  validates only the first commit; the API commit count plus this restriction
  prevents later PR commits from being silently treated as verified.
- `poetry.lock` is modified, with at most one additional modified file:
  `pyproject.toml`. No export, production requirements, rename, or other file
  is in scope. A manifest change must be a change to existing direct-dev
  version constraints only; additions, removals, markers, extras, sources,
  runtime changes, and other manifest edits stay manual.
- When the manifest changes, every changed dependency name must appear in the
  metadata. Additional metadata entries are allowed only for existing direct-
  dev dependencies whose constraints did not change and whose locked versions
  really did update. No changed direct-dev lock version may be unexplained by
  metadata. The metadata update type must match the semver class of the actual
  old-to-new lock transition; concrete stable three-component versions in the
  metadata must match those lock versions. Constraint strings do not identify
  the resolved candidate—the lock does. The locked candidate must be a strictly
  higher stable three-component version. Unsupported lock versions and
  mismatches stay manual. A changed manifest requires a changed lock.
  Lock-only PRs may qualify if all metadata entries are eligible.
- Every package in the lock's production (`main`) graph remains identical.
  This includes packages shared between the runtime and dev groups.

The privileged workflow checks out only the trusted base SHA. It fetches the
base and candidate `pyproject.toml`/`poetry.lock` via GitHub's contents API and
parses them as bounded data; it never checks out or executes PR code. On each
Dependabot PR update, it first tries to cancel an existing native auto-merge
request and confirms it is gone. It then evaluates eligibility. Unsupported,
production, mixed, or unknown candidates do not request auto-merge and remain
available for manual review. If cancellation or the bot guard fails, the
required `policy` check fails instead of silently passing.

For eligible changes, the workflow requests `gh pr merge --auto --merge
--match-head-commit <event-head-sha>`. This is a request for GitHub's native
auto-merge, not a bypass; it uses no `--admin` flag. GitHub can merge immediately
if every required condition is already satisfied.
The candidate's actual dependencies are installed from the lock and checked
by `Dependency CI / quality`. Major dev updates can still break behavior or
pass incomplete tests; maintainers should keep human review requirements when
they want that extra gate.

## Rollout and required GitHub settings

Repository settings were not inspected or changed for this guide. Local YAML
does not prove that GitHub is enforcing the protections needed by native
auto-merge. Roll out in this order:

1. Keep the variable disabled while the implementation moves from `develop`
   to `master` through the repository's normal PR process. The first rollout
   PR cannot be required to pass the new policy job before that workflow exists
   on the default branch; use the checks already available for that rollout.
2. After the workflow is on `master`, confirm that both
   `Dependency CI / quality` and `Dependabot Auto Merge / policy` appear and
   pass on PRs. Their exact GitHub check contexts and source were not verified
   locally.
3. Enable GitHub Actions and ensure organization/repository Actions policies
   permit the actions used and the `pull_request_target` event.
   For security-update PRs, also enable the dependency graph, Dependabot alerts,
   and Dependabot security updates in the repository's security settings.
4. Enable repository auto-merge and allow merge commits, which is the method
   requested by this workflow.
5. Protect `master`, require branches to be up to date, and require both
   successful checks above from GitHub Actions. Retain any desired approvals,
   code-owner, conversation, or other review rules. Native auto-merge waits
   for required reviews and checks; this workflow does not bypass them.
6. Only after those requirements are enforced, deliberately set the repository
   variable `DEPENDABOT_AUTOMERGE_ENABLED` to exactly `v2`. The legacy value
   `true` remains disabled. If checks or strict protection cannot be confirmed,
   leave the v2 opt-in off.

These settings must be applied by a repository owner. Actual GitHub check
contexts, event policies, permissions, branch protections, and native merge
behavior have not been exercised end to end. The workflow and docs are not a
substitute for verifying those controls in the repository UI/API.

## Manual cases and troubleshooting

- Older open PRs without a `poetry.lock` change, PRs with extra files, mixed
  production/dev updates, unverified or multi-commit PRs, unknown metadata,
  unsupported version formats, and lock/metadata mismatches are manual. Ask
  Dependabot to rebase or recreate an appropriate PR when needed; do not widen
  the allowlist to clear the queue.
- If checks pass but a PR stays open, confirm both required contexts on
  `master`, branch freshness, required reviews, the selected merge method, and
  repository/organization Actions policy. A required approval still blocks
  auto-merge until it is granted.
- The `policy` check should run for ordinary PRs too. Its successful result for
  an ineligible Dependabot PR means “no auto-merge requested,” not “the update
  was approved.” Quality checks and human review remain separate.
- Security updates may be urgent. This policy allows only the same eligible
  dev-only update classes; an otherwise excluded security fix remains
  manual. Do not dismiss the alert or treat auto-merge status as resolution.
- Auto-merge does not suppress GitHub notifications. Dependabot alerts and
  notification preferences need their own review.

## Adapting to another project

Do not copy `requirements_dev.txt`, the Poetry group names, branch names, or
this helper unchanged. First identify that project's package manager,
production/development boundaries, lock format, generated exports, runtime,
and real CI commands. Then adapt the changed-file allowlist, per-dependency
metadata mapping, exact candidate-version checks, and production-graph
comparison to its ecosystem. If the ecosystem cannot distinguish these
boundaries reliably, keep updates manual. Run untrusted PR tests without
production credentials or write permissions, keep privileged automation
separate from PR code execution, and make native auto-merge wait for explicit
required checks. Passing CI cannot prove a dependency release is non-malicious
or production-safe.

## References

- [Dependabot options reference](https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference)
- [`dependabot/fetch-metadata` outputs](https://github.com/dependabot/fetch-metadata)
- [GitHub native auto-merge](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-auto-merge-for-pull-requests-in-your-repository)
- [Protected branches and required checks](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [Securely using `pull_request_target`](https://docs.github.com/en/actions/reference/security/securely-using-pull_request_target)
