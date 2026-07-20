# Agent Notes

- Stay on the `main` branch for now. Do not create feature branches unless the
  user explicitly asks for one.
- When a task is done, stage all changes with `git add --all`, commit them, and
  push. If the remote has new commits, pull them in, resolve any conflicts
  semantically, and re-push.
- When publishing `.bin` files on the filesystem, place them under a temp
  artifact folder such as `$HOME/.codex/tmp`. Before writing a `.bin` file, move
  any previous `.bin` file with the same name in that same folder to Trash
  rather than overwriting it in place.
- Rust-first rule: implement production library behavior, solver features,
  validation logic, and reusable tooling in Rust whenever practical. Use Python
  only for external/reference bridges, small compatibility smoke scripts, or
  cases where the ecosystem being validated is itself Python-first. Do not add
  new Python as the primary implementation path for features that belong in the
  Rust crate; if a Python bridge grows real algorithmic or reusable validation
  logic, port that logic to Rust and leave Python as thin adapter glue.
- Protected `https://54.91.17.58/des-rs/...` requests need the legacy `Auth` header.
  Read the value from the operator secret or local credential store; do not commit it.
- Observability must follow the root repo contract: prefer explicit structured stdout/stderr,
  and do not monkey-patch runtime internals, standard streams, module loaders, or process
  messaging for telemetry.
- For permanent AWS credentials for the `dd-codex` project, use the local AWS profile/config in `~/.aws` instead of copying secrets into the repo or chat:
  - Prefer profile-based commands: `AWS_PROFILE=dd-codex aws sts get-caller-identity`.
  - If a tool needs environment variables, export from the profile with AWS CLI v2: `aws configure export-credentials --profile dd-codex --format env`, then source the output in the current shell.
  - If `export-credentials` is unavailable, read values with `aws configure get aws_access_key_id --profile dd-codex`, `aws configure get aws_secret_access_key --profile dd-codex`, and `aws configure get aws_session_token --profile dd-codex` when present; keep those values local and out of git.
- Soccer learning persistence is AWS RDS Postgres, not Neon. Use the AWS RDS
  schema/table definitions under `~/codes/ores/k8s-cluster/remote/libs/pg-defs`
  for declarative Postgres migrations and do not use Neon tooling for this
  project unless the user explicitly reverses this decision.
- For soccer-learning Postgres work, use the AWS RDS Postgres database for the
  `dd-codex` project via the local AWS profile/config. Do not use Neon, Supabase,
  or any other Postgres connector for this repository's soccer-learning storage
  unless the user explicitly asks for that different database.

## Command safety — STRICT (all agents MUST follow)

Never run destructive or irreversible shell commands. To remove or move files,
**always go through git** so the change is tracked and recoverable.

**Blacklisted — do NOT run:**
- `rm`, `rm -rf`, `rmdir`, `unlink` — never delete via raw `rm`.
- bulk / indirect deletion: `find … -delete`, `find … -exec rm …`, `xargs rm` — no bypasses of the `rm` ban.
- raw `mv` of tracked files; truncating a tracked file with `>` or `truncate`.
- `git reset --hard`, `git clean -fdx`, `git checkout -- .` / `git restore .` mass-discard.
- `git stash drop` / `git stash clear`, `git branch -D`, `git tag -d` — destroy unmerged work / refs; not on shared branches unless the operator explicitly asks.
- `git push --force` / history rewrites on shared branches (esp. `main`).
- `dd`, `mkfs`, `shred`, recursive `chmod -R` / `chown -R` on broad paths, fork bombs.

**Whitelisted — safe, prefer these:**
- `git rm` / `git rm --cached` — remove files through git (recoverable via history).
- `git mv` — rename/move through git.
- `git restore <path>` (single file), `git revert`, `git stash` (push) — reversible.
- Editing via the editor tools, `git add`, `git commit`, `git switch -c`.

If a genuinely destructive action seems unavoidable, **STOP and ask the operator
first** — do not improvise around this rule.

## Syncing with the remote

"Sync with the remote" (or just "sync") is **bidirectional and always contacts
the remote** — it pulls *and* pushes. It is never push-only, and a clean local
working tree does **not** by itself mean "synced": a sync is not finished until
local and the remote have exchanged commits in both directions.

The steps for a sync:

1. `git fetch --all --prune` — see what the remote has.
2. `git pull` (which merges) — or `git merge` the upstream tracking branch —
   to integrate the remote's commits into your local branch **first**.
3. `git add` / `git commit` any local work.
4. `git push` — publish your commits.

Always integrate with **`git merge`** (and plain `git pull`, which merges).
**Do not `git rebase`** to sync — rebasing rewrites history and breaks shared
branches; keep the merge history instead.
