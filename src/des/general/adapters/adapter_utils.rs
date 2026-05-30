//! Port of `src/des/general/adapters/adapter-utils.ts`
//! (module `des::general::adapters::adapter_utils`).
//!
//! Shared helpers for the JSON model-spec adapters (CSV emission, validation
//! summaries, logging). These are genuinely stateless utilities, so they stay
//! free functions rather than transform classes (matching the TS migration
//! header).
//!
//! ## Conversion notes (per the TS "RUST MIGRATION" header)
//!
//!   * `fs.writeFileSync` -> [`std::fs::write`]; the CSV quoting regex
//!     `/[",\n]/` becomes a manual scan in [`needs_quoting`].
//!   * `csvCell`/`jsonCsvCell` took JS `unknown` and called `String(v)` /
//!     `JSON.stringify(v)`. Rust has no `unknown`; callers stringify each cell
//!     first (e.g. `x.to_string()`, `format!("{x:.6}")`, or a JSON string for a
//!     complex value), and the cell functions only apply CSV quoting. The
//!     `String(v)` vs `JSON.stringify(v)` distinction therefore moves to the
//!     call site. `json_csv_*` is kept distinct for parity with the TS imports
//!     that alias it.
//!   * `numberPair`/`optionalNumberPair` `throw` on a length != 2 -> `panic!`
//!     (an invariant violation) and return a fixed-size `[f64; 2]`.
//!   * `withLogger` was generic over `T | Promise<T>`; the Rust port is a sync
//!     generic taking a closure, with the logger closed via an explicit
//!     `close()` in all paths (the `try/finally`).
//!   * `JsonlLogger` -> [`crate::des::observability::logger::JsonlLogger`].

#![allow(dead_code)]

use crate::des::general::des_spec::DESRuntimeConfig;
use crate::des::observability::logger::{JsonlLogger, LogLevel};

/// Structural stand-in for the TS `{passed: boolean}` element type accepted by
/// [`validation_line`]. Implemented for the engine's [`ValidationCheck`].
///
/// [`ValidationCheck`]: crate::des::general::des_base::validation::ValidationCheck
pub trait HasPassed {
    fn passed(&self) -> bool;
}

impl HasPassed for crate::des::general::des_base::validation::ValidationCheck {
    fn passed(&self) -> bool {
        self.passed
    }
}

impl HasPassed for crate::des::general::statistical_optimization::ValidationCheck {
    fn passed(&self) -> bool {
        self.passed
    }
}

impl HasPassed for bool {
    fn passed(&self) -> bool {
        *self
    }
}

/// `validationLine` — `"<pass>/<total> checks passed"`.
pub fn validation_line<T: HasPassed>(checks: &[T]) -> String {
    let pass = checks.iter().filter(|c| c.passed()).count();
    format!("{}/{} checks passed", pass, checks.len())
}

/// JS regex `/[",\n]/` — does the cell contain a quote, comma, or newline?
fn needs_quoting(s: &str) -> bool {
    s.contains('"') || s.contains(',') || s.contains('\n')
}

/// `csvCell(v)` — quote-and-escape a single already-stringified cell.
pub fn csv_cell(s: &str) -> String {
    if needs_quoting(s) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// `csvRow(values)` — join already-stringified cells with `,` after quoting.
pub fn csv_row<I, S>(values: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    values
        .into_iter()
        .map(|v| csv_cell(v.as_ref()))
        .collect::<Vec<_>>()
        .join(",")
}

/// `jsonCsvCell(v)` — identical CSV quoting to [`csv_cell`]; kept distinct for
/// the TS imports that alias `jsonCsvRow as csvRow`. (In TS the difference was
/// `String(v)` vs `JSON.stringify(v)`, which is now the caller's choice — see
/// the module docs.)
pub fn json_csv_cell(s: &str) -> String {
    csv_cell(s)
}

/// `jsonCsvRow(values)`.
pub fn json_csv_row<I, S>(values: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    csv_row(values)
}

/// `writeCsvLines(path, lines)` — write the lines joined by `\n`.
pub fn write_csv_lines(csv_path: &str, lines: &[String]) {
    std::fs::write(csv_path, lines.join("\n")).unwrap_or_else(|e| {
        panic!("writeCsvLines: failed to write {csv_path}: {e}");
    });
}

/// `numberPair(values, fallback, name)` — coerce an optional slice into a fixed
/// `[f64; 2]`, falling back when absent. Panics if the length is not 2 (TS
/// `throw`, an invariant violation).
pub fn number_pair(values: Option<&[f64]>, fallback: [f64; 2], name: &str) -> [f64; 2] {
    match values {
        Some(pair) => {
            if pair.len() != 2 {
                panic!("{name} must have length 2");
            }
            [pair[0], pair[1]]
        }
        None => fallback,
    }
}

/// `optionalNumberPair(values, name)` — `None` when absent, else a `[f64; 2]`
/// (panicking on a length != 2).
pub fn optional_number_pair(values: Option<&[f64]>, name: &str) -> Option<[f64; 2]> {
    let values = values?;
    if values.len() != 2 {
        panic!("{name} must have length 2");
    }
    Some([values[0], values[1]])
}

/// `withLogger(runtime, fn)` — build a [`JsonlLogger`] if `runtime.outputs.log`
/// is set, hand it (or `None`) to `f`, then close it before returning (the TS
/// `try/finally`).
pub fn with_logger<T, F>(runtime: &DESRuntimeConfig, f: F) -> T
where
    F: FnOnce(Option<&mut JsonlLogger>) -> T,
{
    let log_path = runtime.outputs.as_ref().and_then(|o| o.log.clone());
    match log_path {
        Some(path) => {
            let mut logger = JsonlLogger::new(&path, LogLevel::Debug);
            let result = f(Some(&mut logger));
            logger.close();
            result
        }
        None => f(None),
    }
}

/// `defaultFramesPath(htmlPath)` — swap a trailing `.html` for `.frames.jsonl`.
pub fn default_frames_path(html_path: &str) -> String {
    if let Some(stripped) = html_path.strip_suffix(".html") {
        format!("{stripped}.frames.jsonl")
    } else {
        format!("{html_path}.frames.jsonl")
    }
}

/// The `{htmlPath?, frames}` shape returned by [`frames_path`].
#[derive(Clone, Debug)]
pub struct FramesPath {
    pub html_path: Option<String>,
    pub frames: String,
}

/// `framesPath(runtime, model)` — resolve the HTML and frames output paths.
pub fn frames_path(runtime: &DESRuntimeConfig, model: &str) -> FramesPath {
    let out = runtime.outputs.clone().unwrap_or_default();
    let html_path = out.html.clone();
    let frames = out.frames.clone().unwrap_or_else(|| match &html_path {
        Some(html) => default_frames_path(html),
        None => format!("out/{model}.frames.jsonl"),
    });
    FramesPath { html_path, frames }
}
