//! Port of `src/des/runners/run-external-module.ts`.
//!
//! CLI front-end for invoking sanctioned external modules. The TS top-level
//! `main()` becomes [`run`], which returns the process exit code so a thin
//! `fn main()` wrapper (added when the runners module is wired) can
//! `std::process::exit` with it.
//!
//! ## PORT NOTE
//!
//!   * `import './external-modules'` (import-time registration side effect) →
//!     an explicit [`register_built_in_external_modules`] call at the top of
//!     [`run`].
//!   * `process.argv.slice(2)` → `std::env::args().skip(1)`.
//!   * `throw new Error(..)` on bad args → printed to stderr + non-zero exit
//!     (user error, not a panic).
//!   * `JSON.stringify(arg)` for echoing argv → `format!("{arg:?}")` (Rust debug
//!     quoting is JSON-compatible for these path/flag strings).
//!   * `console.log`/`console.error` → `println!`/`eprintln!`;
//!     `process.exit(code)` → returned exit code.

#![allow(dead_code)]

use super::external_modules::register_built_in_external_modules;
use super::external_program::{
    list_external_modules, run_external_module, ExternalModuleParams, ParamValue,
};

fn print_help() {
    println!("Usage:");
    println!("  ts-node src/des/runners/run-external-module.ts --list");
    println!("  ts-node src/des/runners/run-external-module.ts <module-id> [--key=value ...]");
    println!();
    println!("External module invocations are shell-free and source paths must live under external-references/.");
}

fn valid_param_key(key: &str) -> bool {
    let mut chars = key.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// `parseValue(raw)`.
fn parse_value(raw: &str) -> ParamValue {
    if raw == "true" {
        return ParamValue::Bool(true);
    }
    if raw == "false" {
        return ParamValue::Bool(false);
    }
    if is_js_number(raw) {
        if let Ok(num) = raw.parse::<f64>() {
            return ParamValue::Num(num);
        }
    }
    ParamValue::Str(raw.to_string())
}

/// `/^-?\d+(\.\d+)?([eE][+-]?\d+)?$/`.
fn is_js_number(raw: &str) -> bool {
    let bytes = raw.as_bytes();
    let mut i = 0;
    if i < bytes.len() && bytes[i] == b'-' {
        i += 1;
    }
    let start_digits = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == start_digits {
        return false; // need at least one integer digit
    }
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        let frac_start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == frac_start {
            return false; // dot with no fractional digits
        }
    }
    if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
        i += 1;
        if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
            i += 1;
        }
        let exp_start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == exp_start {
            return false;
        }
    }
    i == bytes.len()
}

/// `parseParams(args)`.
fn parse_params(args: &[String]) -> Result<ExternalModuleParams, String> {
    let mut out = ExternalModuleParams::new();
    for arg in args {
        if !arg.starts_with("--") {
            return Err(format!("unexpected argument \"{arg}\""));
        }
        let eq = arg.find('=');
        let eq = match eq {
            Some(idx) => idx,
            None => return Err(format!("expected --key=value, got \"{arg}\"")),
        };
        let key = &arg[2..eq];
        let raw = &arg[eq + 1..];
        if !valid_param_key(key) {
            return Err(format!("invalid param key \"{key}\""));
        }
        out.insert(key.to_string(), parse_value(raw));
    }
    Ok(out)
}

fn pad_end(s: &str, width: usize) -> String {
    if s.chars().count() >= width {
        s.to_string()
    } else {
        format!("{s}{}", " ".repeat(width - s.chars().count()))
    }
}

/// `main()` — returns the process exit code.
pub fn run() -> i32 {
    if let Err(e) = register_built_in_external_modules() {
        eprintln!("Error: {e}");
        return 1;
    }

    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv[0] == "-h" || argv[0] == "--help" {
        print_help();
        return if argv.is_empty() { 1 } else { 0 };
    }

    if argv[0] == "--list" {
        let modules = list_external_modules();
        println!("External modules ({}):", modules.len());
        for m in &modules {
            let env = &m.interpreter.env_var;
            let cmd = std::env::var(env).unwrap_or_else(|_| m.interpreter.default_command.clone());
            println!(
                "  {} {} {} via {env}={cmd}",
                pad_end(&m.id, 30),
                pad_end(m.kind.as_str(), 10),
                m.interpreter.label
            );
            println!("    {}", m.description);
        }
        return 0;
    }

    let id = &argv[0];
    let params = match parse_params(&argv[1..]) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error: {e}");
            return 1;
        }
    };
    let r = match run_external_module(id, &params) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error: {e}");
            return 1;
        }
    };
    println!("external module: {id}");
    let echoed: Vec<String> = r.args.iter().map(|a| format!("{a:?}")).collect();
    println!("command: {} {}", r.command, echoed.join(" "));
    if !r.stdout.trim().is_empty() {
        println!("{}", r.stdout.trim());
    }
    if !r.stderr.trim().is_empty() {
        eprintln!("{}", r.stderr.trim());
    }
    r.status.unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_value_kinds() {
        assert_eq!(parse_value("true"), ParamValue::Bool(true));
        assert_eq!(parse_value("false"), ParamValue::Bool(false));
        assert_eq!(parse_value("42"), ParamValue::Num(42.0));
        assert_eq!(parse_value("-3.5e2"), ParamValue::Num(-350.0));
        assert_eq!(parse_value("auto"), ParamValue::Str("auto".to_string()));
        assert_eq!(parse_value("3abc"), ParamValue::Str("3abc".to_string()));
    }

    #[test]
    fn rejects_bad_args() {
        assert!(parse_params(&["nope".to_string()]).is_err());
        assert!(parse_params(&["--noeq".to_string()]).is_err());
        assert!(parse_params(&["--1bad=1".to_string()]).is_err());
        assert!(parse_params(&["--ok=1".to_string()]).is_ok());
    }
}
