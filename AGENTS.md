# Agent Notes

- Protected `https://54.91.17.58/des-rs/...` requests need header `Auth: all-dogs-go-to-heaven`.
- For permanent AWS credentials for the `dd-codex` project, use the local AWS profile/config in `~/.aws` instead of copying secrets into the repo or chat:
  - Prefer profile-based commands: `AWS_PROFILE=dd-codex aws sts get-caller-identity`.
  - If a tool needs environment variables, export from the profile with AWS CLI v2: `aws configure export-credentials --profile dd-codex --format env`, then source the output in the current shell.
  - If `export-credentials` is unavailable, read values with `aws configure get aws_access_key_id --profile dd-codex`, `aws configure get aws_secret_access_key --profile dd-codex`, and `aws configure get aws_session_token --profile dd-codex` when present; keep those values local and out of git.
