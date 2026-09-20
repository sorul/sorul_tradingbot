# MT5 Raspberry recovery

## Objective
Restore the MetaTrader Docker service so KasmVNC on port 3000 and `terminal64.exe` start reliably on the reset Raspberry.

## Problem and evidence
- `mt5` is running and publishes port 3000, but the browser receives 502.
- Logs confirm `wine32:i386` is missing and `/config/.wine` is owned by UID/GID 1000 while the LinuxServer application user is 911:911.
- `binfmt_misc` completed successfully; it is not the current blocker.

## Scope and constraints
- Change only the MT5 Docker runtime configuration and image dependencies needed to fix the verified failures.
- User authorized a backup-first repair of the persistent Wine prefix on 2026-09-20. Create and verify a full recoverable backup before replacing only the broken Wine drive mappings; preserve AgentFiles and unrelated working-tree changes.
- Preserve unrelated working-tree changes.
- Delivery strategy: ask-on-risk. TDD: disabled for this container configuration repair; use Docker runtime verification.

## Tasks
- [x] MT5-1 (delegated; writer trigger: Dockerfile and Compose): configured `PUID=1000` and `PGID=1000`, enabled the `i386` architecture, and installed `wine32:i386` and `libwine:i386` alongside the existing amd64 Wine runtime. `docker compose config` resolved those exact PUID/PGID values and the `./metatrader:/config` bind mount. The rebuilt image is `sha256:2aa157e7852457933d89f01fe27868c637678135ba40f7776b29225823a4b3c5`; `docker compose up -d --force-recreate --no-deps mt5` recreated only `mt5`. Rollback boundary: Dockerfile and docker-compose.yaml only.
- [ ] MT5-2 (delegated; repair and verification, partial): stopped only `mt5`, made the full backup `metatrader/.wine.backup-20260920T223324Z`, then replaced only `metatrader/.wine/dosdevices/c:` and `metatrader/.wine/dosdevices/z:`. The repaired paths are symlinks `c: -> ../drive_c` and `z: -> /`. The backup is excluded from future Docker build contexts by `.dockerignore` (`metatrader/*`), measures `7.1G` versus the source's `7.5G` allocated usage, preserves the original regular `c:` (28 bytes) and `z:` (10 bytes) entries, and passed `rsync -aHAXS --numeric-ids -n --itemize-changes` with no differences (`PARITY=clean`). `mt5` is running (`exit=0`, `oom=false`, `restarts=0`) and KasmVNC remains reachable (`http_code=200`), but `terminal64.exe` is still absent. Fresh logs no longer report the prior `dosdevices/c:/windows: Not a directory` failure; they do report the independent remaining failure `wine: could not load kernel32.dll, status c0000135`. Do not modify the remaining prefix without new authorization.

## Progress
- Created 2026-09-20 after confirmed read-only diagnostics.
- MT5-1 evidence: the LinuxServer startup log reports `User UID: 1000` and `User GID: 1000`; the MT5 launcher runs as `abc` UID 1000. `dpkg-query` reports `wine32:i386=ii`, `wine64:amd64=ii`, `libwine:i386=ii`, and `libwine:amd64=ii`. The prefix owner remains `1000:1000`; it was not modified.
- MT5-2 evidence: container status `running`, health `none`, exit `0`, OOM `false`, restarts `0`; port mapping is `0.0.0.0:3000->3000/tcp` and HTTP status is 200. `terminal64.exe` has no process because of the broken persistent-prefix drive mapping above.
- MT5-2 backup and repair evidence: before editing, `metatrader/.wine/dosdevices/c:` and `z:` were UID/GID `1000:1000` regular data files (28 and 10 bytes respectively); those original entries remain in `metatrader/.wine.backup-20260920T223324Z/dosdevices/`. The backup was made with `rsync -aHAXS --numeric-ids`, preserving modes, ownership, timestamps, symlinks, ACLs, xattrs, hard links, and sparse files where supported. Its non-mutating parity check was clean. The source paths now resolve to directories: `dosdevices/c:` and `drive_c` have the same inode, and `dosdevices/c:/windows` is a directory in both host and container.
- MT5-2 runtime verification: after `docker compose start mt5`, `mt5` was `running`, `exit=0`, `oom=false`, `restarts=0`, and port 3000 returned HTTP 200. `docker exec mt5 ... ps` found no `terminal64.exe` or `wineserver` process. Current-start logs show `terminal64.exe` is installed, no longer show `init_redirects ... c:/windows: Not a directory`, and repeatedly show `wine: could not load kernel32.dll, status c0000135`.
- Rollback for the authorized mapping repair: stop only MT5 (`docker compose stop mt5`); remove the two current symlinks and restore only the original entries from `metatrader/.wine.backup-20260920T223324Z/dosdevices/c:` and `z:` with `cp -a`; then start MT5. The backup remains adjacent to the live prefix and AgentFiles were not modified.
- Next: the authorized mapping repair is complete, but MT5 remains blocked by the `kernel32.dll` Wine-prefix failure. Obtain authorization before any broader prefix reconstruction or restoration, then verify `terminal64.exe`, MT5 GUI/login, EA, and fresh AgentFiles.
