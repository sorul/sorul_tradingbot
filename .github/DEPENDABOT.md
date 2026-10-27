# Dependabot auto-merge

The privileged workflow only requests native auto-merge for same-repository
Dependabot pull requests whose every update is a direct development `pip`
patch/minor at `/`, and whose only changed file is `requirements_dev.txt`.
Everything else remains for manual review. It does not check out or execute PR
code.

Auto-merge is disabled unless the repository variable
`DEPENDABOT_AUTOMERGE_ENABLED` is exactly `true`. Before an owner opts in:

1. Enable GitHub auto-merge and allow merge commits for the repository.
2. Require the `Dependency CI` quality job as a status check on every target
   branch. Confirm the exact check context in GitHub's branch-protection UI;
   do not assume its displayed name.
3. Ensure these workflows are present on the repository's default branch, then
   set `DEPENDABOT_AUTOMERGE_ENABLED=true`.

No repository settings are changed by this workflow. It asks GitHub to merge
only after required checks pass and uses the event's expected head SHA; it does
not bypass required checks or use administrator privileges.
