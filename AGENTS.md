# Agent Notes

- Stay on the `main` branch for now. Do not create feature branches unless the
  user explicitly asks for one.
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
- Protected `https://54.91.17.58/des-rs/...` requests need header `Auth: all-dogs-go-to-heaven`.
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
