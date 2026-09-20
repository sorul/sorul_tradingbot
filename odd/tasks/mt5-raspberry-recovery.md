# MT5 Raspberry recovery

## Objective
Restore the MetaTrader Docker service so KasmVNC on port 3000 and `terminal64.exe` start reliably on the reset Raspberry.

## Problem and evidence
- `mt5` is running and publishes port 3000, but the browser receives 502.
- Logs confirm `wine32:i386` is missing and `/config/.wine` is owned by UID/GID 1000 while the LinuxServer application user is 911:911.
- `binfmt_misc` completed successfully; it is not the current blocker.

## Scope and constraints
- Change only the MT5 Docker runtime configuration and image dependencies needed to fix the verified failures.
- Preserve `./metatrader` and its AgentFiles; do not delete or recreate the persistent Wine prefix.
- Preserve unrelated working-tree changes.
- Delivery strategy: ask-on-risk. TDD: disabled for this container configuration repair; use Docker runtime verification.

## Tasks
- [x] MT5-1 (delegated; writer trigger: Dockerfile and Compose): configured `PUID=1000` and `PGID=1000`, enabled the `i386` architecture, and installed `wine32:i386` and `libwine:i386` alongside the existing amd64 Wine runtime. `docker compose config` resolved those exact PUID/PGID values and the `./metatrader:/config` bind mount. The rebuilt image is `sha256:2aa157e7852457933d89f01fe27868c637678135ba40f7776b29225823a4b3c5`; `docker compose up -d --force-recreate --no-deps mt5` recreated only `mt5`. Rollback boundary: Dockerfile and docker-compose.yaml only.
- [ ] MT5-2 (delegated; verification, blocked): KasmVNC is reachable (`curl http://localhost:3000` returned `http_code=200`), and `mt5` is `running`, with no restarts or OOM. The initial `wine32 is missing` and `wine: '/config/.wine' is not owned by you` failures are absent. However, `terminal64.exe` is not running because the persistent prefix has a broken `/config/.wine/dosdevices/c:` entry: it is a regular 28-byte file rather than a Windows-drive symlink/directory, so Wine reports `init_redirects ... c:/windows: Not a directory` and `could not load kernel32.dll, status c0000135`. This scope must not delete, recreate, chown, or otherwise alter `./metatrader`; prefix recovery requires separate authorization and a backup-first plan.

## Progress
- Created 2026-09-20 after confirmed read-only diagnostics.
- MT5-1 evidence: the LinuxServer startup log reports `User UID: 1000` and `User GID: 1000`; the MT5 launcher runs as `abc` UID 1000. `dpkg-query` reports `wine32:i386=ii`, `wine64:amd64=ii`, `libwine:i386=ii`, and `libwine:amd64=ii`. The prefix owner remains `1000:1000`; it was not modified.
- MT5-2 evidence: container status `running`, health `none`, exit `0`, OOM `false`, restarts `0`; port mapping is `0.0.0.0:3000->3000/tcp` and HTTP status is 200. `terminal64.exe` has no process because of the broken persistent-prefix drive mapping above.
- Next: decide whether to authorize a backup-first repair or restoration of `metatrader/.wine`; MT5 login and GUI setup can only be assessed once `terminal64.exe` starts.
