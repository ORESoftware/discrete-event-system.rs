//! Port of src/des/test/external-module-test.ts
//
// `runners/external_program` is now ported. The out-of-process invocation path
// needs python/node + scripts on disk, so here we test the pure, deterministic
// piece: the path-safety guard in `resolve_external_script`. (Subprocess
// execution is covered by the validate-* external runners when interpreters are
// available; the repo-root env override is left untested to avoid mutating
// global process env in parallel tests.)

#[cfg(test)]
mod tests {
    use crate::des::runners::external_program::resolve_external_script;
    use std::path::Path;

    #[test]
    fn resolve_rejects_scripts_outside_external_references() {
        let root = Path::new("/tmp/some-repo");
        // A path that escapes the external-references sandbox must be rejected.
        let err = resolve_external_script(root, "src/secret.py").unwrap_err();
        assert!(err.contains("external-references"), "unexpected error: {err}");
    }

    #[test]
    fn resolve_rejects_missing_script_under_sandbox() {
        let root = Path::new("/tmp/some-repo");
        // Lives under external-references/ but does not exist on disk.
        let err = resolve_external_script(root, "external-references/does-not-exist.py")
            .unwrap_err();
        assert!(err.contains("not found"), "unexpected error: {err}");
    }
}
