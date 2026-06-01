//! Small shared helpers for generated Rust source.

/// Return a Rust raw string literal containing `s`.
///
/// The generator prefers raw strings because specs are JSON-heavy. It increases
/// the hash count until the terminator cannot appear inside the payload, then
/// falls back to an escaped debug string if a pathological payload defeats the
/// small search window.
pub fn rust_raw_string_literal(s: &str) -> String {
    for hashes in 0..16 {
        let marks = "#".repeat(hashes);
        let terminator = format!("\"{marks}");
        if !s.contains(&terminator) {
            return format!("r{marks}\"{s}\"{marks}");
        }
    }
    format!("{s:?}")
}

/// Conservative Rust identifier sanitizer for generated function/module names.
pub fn rust_ident(input: &str, fallback: &str) -> String {
    let mut out = String::new();
    for (i, ch) in input.chars().enumerate() {
        let ok = ch == '_' || ch.is_ascii_alphanumeric();
        if !ok {
            out.push('_');
            continue;
        }
        if i == 0 && ch.is_ascii_digit() {
            out.push('_');
        }
        out.push(ch);
    }
    if out.chars().all(|ch| ch == '_') || is_keyword(&out) {
        fallback.to_string()
    } else {
        out
    }
}

fn is_keyword(s: &str) -> bool {
    matches!(
        s,
        "as" | "break"
            | "const"
            | "continue"
            | "crate"
            | "else"
            | "enum"
            | "extern"
            | "false"
            | "fn"
            | "for"
            | "if"
            | "impl"
            | "in"
            | "let"
            | "loop"
            | "match"
            | "mod"
            | "move"
            | "mut"
            | "pub"
            | "ref"
            | "return"
            | "self"
            | "Self"
            | "static"
            | "struct"
            | "super"
            | "trait"
            | "true"
            | "type"
            | "unsafe"
            | "use"
            | "where"
            | "while"
            | "async"
            | "await"
            | "dyn"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_literal_raises_hash_count_when_needed() {
        let s = "contains \"# once";
        let lit = rust_raw_string_literal(s);
        assert!(lit.starts_with("r##\"") || lit.starts_with("r#\""));
        assert!(lit.contains(s));
    }

    #[test]
    fn identifiers_are_sanitized() {
        assert_eq!(rust_ident("1 bad-name", "fallback"), "_1_bad_name");
        assert_eq!(rust_ident("fn", "fallback"), "fallback");
    }
}
