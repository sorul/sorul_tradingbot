# Dependabot automation

## Scope and decision path

Dependabot updates the root `pip` ecosystem. Native auto-merge is limited to
same-repository `dependabot[bot]` pull requests targeting `master` whose every
metadata entry is a direct development dependency and a patch, minor, or major
update. The PR must change only `poetry.lock`, optionally alongside
`pyproject.toml`; production requirements and the production lock graph must
remain unchanged. Lock-only transitive updates may qualify when every reported
direct update is eligible. Production, mixed, source-based, ambiguous, or
unknown updates receive no auto-merge request and remain manual.

The trusted `pull_request_target` job reads the candidate TOML files as bounded
data from GitHub's contents API. It checks out only the trusted base SHA and
never checks out or runs PR code. The candidate must be one verified commit;
this deliberately compensates for `dependabot/fetch-metadata@v3` validating
only the first commit. Metadata is checked per updated dependency, then
reconciled with the manifest and lockfile, including unchanged production lock
entries. The ordinary `Dependency CI / quality` job installs from the lock
and verifies installed direct development versions before lint and tests.

On every Dependabot PR update, the privileged job first cancels any existing
native auto-merge request. It requests a new one only when the opt-in and all
candidate checks pass, using the event head SHA with `gh pr merge
--match-head-commit`; it never bypasses rules with `--admin`. The required
`policy` status serializes these decisions and fails when the bot guard or
auto-merge cancellation cannot be verified. It succeeds without requesting
auto-merge for ordinary PRs and unsupported Dependabot updates, so those can
still be reviewed manually.

## One-time repository setup

Do not enable the v2 opt-in until all of these are true on `master`:

1. Enable GitHub Actions and permit the `pull_request_target` event under
   organization/repository Actions policies.
2. Enable repository auto-merge and allow the merge method used here (merge
   commits).
3. Protect `master`, require branches to be up to date, and require both
   `Dependency CI / quality` and `Dependabot Auto Merge / policy` from GitHub
   Actions. Select the exact check contexts shown by GitHub; confirm they are
   passing on a PR before requiring them.
4. Only after those protections are active, set the repository variable
   `DEPENDABOT_AUTOMERGE_ENABLED` to exactly `v2`. The older value `true` is
   intentionally not accepted, so it does not activate this broader policy.

These settings must be applied by a repository owner; they are not changed by
the workflow and were not verified as part of this code change. If the
required policy check is absent or branch protection is not strict, do not set
the variable to `v2`: the workflow alone is not a merge safety boundary.

## Security and operational limits

Pull-request CI checks untrusted dependency releases and PR code. It has no
write permission and receives no project secrets; tests must not access live
trading services. The privileged workflow is isolated from that execution and
uses only the minimal read/write token permissions needed to inspect the PR,
cancel prior auto-merge, and request guarded auto-merge.

One verified Dependabot commit, metadata, a lock diff, and passing checks do
not prove a dependency release is benign. Major upgrades can change behavior;
auto-merge should be disabled while investigating suspicious releases or
when maintainers cannot establish provenance. Unsupported metadata, lock
layouts, versions, source declarations, and mixed PRs deliberately fall back
to manual review. Test success is not proof of production or financial safety.

The helper and workflows are locally tested, but actual GitHub check contexts,
branch protection, event-policy behavior, permissions, and native auto-merge
have not been exercised against the repository. Revalidate those settings in
GitHub before enabling the variable. `requirements_dev.txt` is a local ignored
export; CI and Dependabot use the committed Poetry lockfile instead.
