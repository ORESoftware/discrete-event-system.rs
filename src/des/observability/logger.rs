//! Port of `src/des/observability/logger.ts`.
//!
//! Append-only line-delimited JSON (JSONL) event logger plus a convenience
//! reader for offline validators / comparators. The file is opened truncating
//! ('w') so run-to-run logs do not stack on top of each other; writes are
//! synchronous and cheap when filtered out (the level check happens before
//! serialization).
//!
//! The TypeScript engine relied on the host's `JSON` plus node `fs`/`path`.
//! This crate has no serde dependency, so JSON is handled by a small
//! self-contained value type ([`JsonValue`]) with a serializer ([`Display`] /
//! [`JsonValue::to_string_pretty`]) and a recursive-descent parser
//! ([`parse_json`]). `fs.createWriteStream`/`mkdirSync` become
//! `std::fs::File`/`create_dir_all` behind a `BufWriter`, and `JSON.parse`
//! errors surface as a `Result` carrying the offending line number.

#![allow(dead_code)]

use std::collections::HashMap;
use std::fmt;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

// =============================================================================
// Minimal JSON value type + (de)serialization (serde is not a dependency here).
// =============================================================================

/// An arbitrary JSON value — the Rust analog of `Record<string, any>` event
/// payloads. Objects keep their entries in insertion order (like a JS object)
/// so serialization is stable and the validators can rely on field order.
#[derive(Clone, Debug, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    /// Object field access. Returns `None` for non-objects or missing keys.
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Object(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Follow a chain of object keys (e.g. `["config", "probabilities", "x"]`).
    pub fn pointer(&self, path: &[&str]) -> Option<&JsonValue> {
        let mut cur = self;
        for k in path {
            cur = cur.get(k)?;
        }
        Some(cur)
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Vec<JsonValue>> {
        match self {
            JsonValue::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&Vec<(String, JsonValue)>> {
        match self {
            JsonValue::Object(o) => Some(o),
            _ => None,
        }
    }

    /// Pretty-print with `indent`-space indentation, mirroring
    /// `JSON.stringify(value, null, indent)`.
    pub fn to_string_pretty(&self, indent: usize) -> String {
        let mut out = String::new();
        self.write_pretty(&mut out, indent, 0);
        out
    }

    fn write_compact(&self, out: &mut String) {
        match self {
            JsonValue::Null => out.push_str("null"),
            JsonValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            JsonValue::Number(n) => out.push_str(&number_to_json(*n)),
            JsonValue::String(s) => write_escaped(s, out),
            JsonValue::Array(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    v.write_compact(out);
                }
                out.push(']');
            }
            JsonValue::Object(o) => {
                out.push('{');
                for (i, (k, v)) in o.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_escaped(k, out);
                    out.push(':');
                    v.write_compact(out);
                }
                out.push('}');
            }
        }
    }

    fn write_pretty(&self, out: &mut String, indent: usize, level: usize) {
        match self {
            JsonValue::Object(o) => {
                if o.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push_str("{\n");
                let pad = " ".repeat(indent * (level + 1));
                for (i, (k, v)) in o.iter().enumerate() {
                    out.push_str(&pad);
                    write_escaped(k, out);
                    out.push_str(": ");
                    v.write_pretty(out, indent, level + 1);
                    if i + 1 < o.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                out.push_str(&" ".repeat(indent * level));
                out.push('}');
            }
            JsonValue::Array(a) => {
                if a.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push_str("[\n");
                let pad = " ".repeat(indent * (level + 1));
                for (i, v) in a.iter().enumerate() {
                    out.push_str(&pad);
                    v.write_pretty(out, indent, level + 1);
                    if i + 1 < a.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                out.push_str(&" ".repeat(indent * level));
                out.push(']');
            }
            other => other.write_compact(out),
        }
    }
}

impl fmt::Display for JsonValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = String::new();
        self.write_compact(&mut s);
        f.write_str(&s)
    }
}

/// Format a number for JSON, matching JS `JSON.stringify`: finite numbers use
/// the shortest round-tripping decimal (Rust `{}` uses the same algorithm as
/// JS `String`), and non-finite values serialize as `null`.
fn number_to_json(n: f64) -> String {
    if n.is_finite() {
        format!("{n}")
    } else {
        "null".to_string()
    }
}

fn write_escaped(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Parse a single JSON value from `input`. Returns an error message on
/// malformed input or trailing characters (matching `JSON.parse`'s strictness).
pub fn parse_json(input: &str) -> Result<JsonValue, String> {
    let mut p = JsonParser {
        chars: input.chars().collect(),
        pos: 0,
    };
    p.skip_ws();
    let v = p.parse_value()?;
    p.skip_ws();
    if p.pos != p.chars.len() {
        return Err(format!("unexpected trailing characters at position {}", p.pos));
    }
    Ok(v)
}

struct JsonParser {
    chars: Vec<char>,
    pos: usize,
}

impl JsonParser {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn next(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn parse_value(&mut self) -> Result<JsonValue, String> {
        self.skip_ws();
        match self.peek() {
            Some('{') => self.parse_object(),
            Some('[') => self.parse_array(),
            Some('"') => Ok(JsonValue::String(self.parse_string()?)),
            Some('t') | Some('f') => self.parse_bool(),
            Some('n') => self.parse_null(),
            Some(c) if c == '-' || c.is_ascii_digit() => self.parse_number(),
            Some(c) => Err(format!("unexpected character '{c}' at position {}", self.pos)),
            None => Err("unexpected end of input".to_string()),
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, String> {
        self.expect('{')?;
        let mut entries: Vec<(String, JsonValue)> = Vec::new();
        self.skip_ws();
        if self.peek() == Some('}') {
            self.pos += 1;
            return Ok(JsonValue::Object(entries));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some('"') {
                return Err(format!("expected string key at position {}", self.pos));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect(':')?;
            let val = self.parse_value()?;
            entries.push((key, val));
            self.skip_ws();
            match self.next() {
                Some(',') => continue,
                Some('}') => break,
                other => {
                    return Err(format!(
                        "expected ',' or '}}' but found {other:?} at position {}",
                        self.pos
                    ))
                }
            }
        }
        Ok(JsonValue::Object(entries))
    }

    fn parse_array(&mut self) -> Result<JsonValue, String> {
        self.expect('[')?;
        let mut items: Vec<JsonValue> = Vec::new();
        self.skip_ws();
        if self.peek() == Some(']') {
            self.pos += 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            let val = self.parse_value()?;
            items.push(val);
            self.skip_ws();
            match self.next() {
                Some(',') => continue,
                Some(']') => break,
                other => {
                    return Err(format!(
                        "expected ',' or ']' but found {other:?} at position {}",
                        self.pos
                    ))
                }
            }
        }
        Ok(JsonValue::Array(items))
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.expect('"')?;
        let mut s = String::new();
        loop {
            match self.next() {
                None => return Err("unterminated string".to_string()),
                Some('"') => break,
                Some('\\') => match self.next() {
                    Some('"') => s.push('"'),
                    Some('\\') => s.push('\\'),
                    Some('/') => s.push('/'),
                    Some('n') => s.push('\n'),
                    Some('r') => s.push('\r'),
                    Some('t') => s.push('\t'),
                    Some('b') => s.push('\u{08}'),
                    Some('f') => s.push('\u{0C}'),
                    Some('u') => {
                        let mut code = 0u32;
                        for _ in 0..4 {
                            let c = self.next().ok_or("bad unicode escape")?;
                            let d = c.to_digit(16).ok_or("bad unicode escape")?;
                            code = code * 16 + d;
                        }
                        s.push(char::from_u32(code).unwrap_or('\u{FFFD}'));
                    }
                    other => return Err(format!("bad escape {other:?}")),
                },
                Some(c) => s.push(c),
            }
        }
        Ok(s)
    }

    fn parse_number(&mut self) -> Result<JsonValue, String> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E' || c.is_ascii_digit() {
                self.pos += 1;
            } else {
                break;
            }
        }
        let token: String = self.chars[start..self.pos].iter().collect();
        token
            .parse::<f64>()
            .map(JsonValue::Number)
            .map_err(|e| format!("bad number '{token}': {e}"))
    }

    fn parse_bool(&mut self) -> Result<JsonValue, String> {
        if self.match_literal("true") {
            Ok(JsonValue::Bool(true))
        } else if self.match_literal("false") {
            Ok(JsonValue::Bool(false))
        } else {
            Err(format!("invalid literal at position {}", self.pos))
        }
    }

    fn parse_null(&mut self) -> Result<JsonValue, String> {
        if self.match_literal("null") {
            Ok(JsonValue::Null)
        } else {
            Err(format!("invalid literal at position {}", self.pos))
        }
    }

    fn match_literal(&mut self, lit: &str) -> bool {
        let litchars: Vec<char> = lit.chars().collect();
        if self.pos + litchars.len() <= self.chars.len()
            && self.chars[self.pos..self.pos + litchars.len()] == litchars[..]
        {
            self.pos += litchars.len();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, c: char) -> Result<(), String> {
        if self.peek() == Some(c) {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!("expected '{c}' at position {}", self.pos))
        }
    }
}

// =============================================================================
// Log levels.
// =============================================================================

/// `type LogLevel = 'trace' | 'debug' | 'info' | 'warn' | 'error'`. Ordering
/// follows the `LEVEL_ORDER` rank below (trace < debug < info < warn < error).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    /// `LEVEL_ORDER[level]`.
    pub fn rank(self) -> u8 {
        match self {
            LogLevel::Trace => 0,
            LogLevel::Debug => 1,
            LogLevel::Info => 2,
            LogLevel::Warn => 3,
            LogLevel::Error => 4,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            LogLevel::Trace => "trace",
            LogLevel::Debug => "debug",
            LogLevel::Info => "info",
            LogLevel::Warn => "warn",
            LogLevel::Error => "error",
        }
    }
}

/// Rank of a level name, or `None` for an unknown level. An unknown level is
/// never filtered out (mirroring JS `undefined < minLevel === false`).
fn level_order(level: &str) -> Option<u8> {
    match level {
        "trace" => Some(0),
        "debug" => Some(1),
        "info" => Some(2),
        "warn" => Some(3),
        "error" => Some(4),
        _ => None,
    }
}

/// The base shape every event shares. Open event payloads are represented as a
/// [`JsonValue::Object`]; this struct documents the common fields.
#[derive(Clone, Debug)]
pub struct BaseEvent {
    pub kind: String,
    pub level: Option<LogLevel>,
    pub t: Option<f64>,
}

// =============================================================================
// The JSONL logger.
// =============================================================================

pub struct JsonlLogger {
    stream: Option<BufWriter<File>>,
    min_level: u8,
    file_path: String,
    event_count: u64,
    by_kind: HashMap<String, u64>,
}

impl JsonlLogger {
    /// Open `file_path` for truncating writes, creating parent directories as
    /// needed. Panics if the file cannot be created (an environment invariant).
    pub fn new(file_path: &str, min_level: LogLevel) -> Self {
        if let Some(parent) = Path::new(file_path).parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fs::create_dir_all(parent);
            }
        }
        let file = File::create(file_path)
            .unwrap_or_else(|e| panic!("JsonlLogger: cannot create '{file_path}': {e}"));
        JsonlLogger {
            stream: Some(BufWriter::new(file)),
            min_level: min_level.rank(),
            file_path: file_path.to_string(),
            event_count: 0,
            by_kind: HashMap::new(),
        }
    }

    /// Write one event (a `{level, ...event}` object). Filtered out cheaply if
    /// the event's level is below `min_level`.
    pub fn log(&mut self, event: JsonValue) {
        let entries = match event {
            JsonValue::Object(e) => e,
            _ => Vec::new(),
        };
        let level = entries
            .iter()
            .find(|(k, _)| k == "level")
            .and_then(|(_, v)| v.as_str())
            .unwrap_or("info")
            .to_string();
        if let Some(rank) = level_order(&level) {
            if rank < self.min_level {
                return;
            }
        }
        let kind = entries
            .iter()
            .find(|(k, _)| k == "kind")
            .and_then(|(_, v)| v.as_str())
            .unwrap_or("")
            .to_string();

        // Build `{level, ...event}` (level first, deduplicated).
        let mut out: Vec<(String, JsonValue)> = Vec::with_capacity(entries.len() + 1);
        out.push(("level".to_string(), JsonValue::String(level)));
        for (k, v) in entries {
            if k == "level" {
                continue;
            }
            out.push((k, v));
        }
        let line = JsonValue::Object(out).to_string();
        if let Some(w) = self.stream.as_mut() {
            let _ = writeln!(w, "{line}");
        }
        self.event_count += 1;
        *self.by_kind.entry(kind).or_insert(0) += 1;
    }

    pub fn get_event_count(&self) -> u64 {
        self.event_count
    }

    pub fn get_kind_counts(&self) -> HashMap<String, u64> {
        self.by_kind.clone()
    }

    pub fn get_file_path(&self) -> &str {
        &self.file_path
    }

    /// Flush and close the underlying file. Idempotent.
    pub fn close(&mut self) {
        if let Some(mut w) = self.stream.take() {
            let _ = w.flush();
        }
    }
}

impl Drop for JsonlLogger {
    fn drop(&mut self) {
        if let Some(w) = self.stream.as_mut() {
            let _ = w.flush();
        }
    }
}

/// Convenience reader for offline validators / comparators. Returns an error
/// (carrying the 1-based line number) on a malformed JSONL line.
pub fn read_events(file_path: &str) -> Result<Vec<JsonValue>, String> {
    let raw = fs::read_to_string(file_path).map_err(|e| format!("cannot read '{file_path}': {e}"))?;
    let mut events: Vec<JsonValue> = Vec::new();
    for (i, line) in raw.split('\n').enumerate() {
        if line.is_empty() {
            continue;
        }
        match parse_json(line) {
            Ok(v) => events.push(v),
            Err(err) => {
                return Err(format!(
                    "malformed JSONL at line {} of {file_path}: {err}",
                    i + 1
                ))
            }
        }
    }
    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("des_logger_{}_{}", std::process::id(), name));
        p
    }

    fn obj(entries: Vec<(&str, JsonValue)>) -> JsonValue {
        JsonValue::Object(entries.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    #[test]
    fn write_read_roundtrip_and_filtering() {
        let path = temp_path("roundtrip.jsonl");
        let p = path.to_str().unwrap();
        {
            let mut logger = JsonlLogger::new(p, LogLevel::Info);
            logger.log(obj(vec![
                ("kind", JsonValue::String("a".into())),
                ("t", JsonValue::Number(1.0)),
            ]));
            // Below the min level -> filtered out.
            logger.log(obj(vec![
                ("kind", JsonValue::String("b".into())),
                ("level", JsonValue::String("debug".into())),
            ]));
            logger.log(obj(vec![
                ("kind", JsonValue::String("a".into())),
                ("level", JsonValue::String("warn".into())),
            ]));
            logger.close();
            assert_eq!(logger.get_event_count(), 2);
            let counts = logger.get_kind_counts();
            assert_eq!(counts.get("a"), Some(&2));
            assert_eq!(counts.get("b"), None);
            assert_eq!(logger.get_file_path(), p);
        }
        let events = read_events(p).expect("read events");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].get("kind").and_then(|v| v.as_str()), Some("a"));
        assert_eq!(events[0].get("level").and_then(|v| v.as_str()), Some("info"));
        assert_eq!(events[1].get("level").and_then(|v| v.as_str()), Some("warn"));
        let _ = fs::remove_file(p);
    }

    #[test]
    fn json_parse_serialize_roundtrip() {
        let src = r#"{"kind":"tick","t":3,"populations":{"S":2,"E":1},"flag":true,"none":null,"edges":[["a","b"],["c","d"]]}"#;
        let v = parse_json(src).expect("parse");
        assert_eq!(v.get("kind").and_then(|x| x.as_str()), Some("tick"));
        assert_eq!(v.pointer(&["populations", "S"]).and_then(|x| x.as_f64()), Some(2.0));
        assert_eq!(v.get("flag").and_then(|x| x.as_bool()), Some(true));
        assert_eq!(v.get("none"), Some(&JsonValue::Null));

        let serialized = v.to_string();
        let v2 = parse_json(&serialized).expect("reparse");
        assert_eq!(v2.pointer(&["populations", "E"]).and_then(|x| x.as_f64()), Some(1.0));

        let edges = v.get("edges").and_then(|x| x.as_array()).unwrap();
        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0].as_array().unwrap()[0].as_str(), Some("a"));
    }

    #[test]
    fn number_formatting_matches_js() {
        assert_eq!(number_to_json(5.0), "5");
        assert_eq!(number_to_json(2.5), "2.5");
        assert_eq!(number_to_json(0.1), "0.1");
        assert_eq!(number_to_json(f64::INFINITY), "null");
        assert_eq!(number_to_json(f64::NAN), "null");
    }

    #[test]
    fn pretty_printing() {
        let v = obj(vec![
            ("a", JsonValue::Number(1.0)),
            ("b", JsonValue::Array(vec![JsonValue::Number(2.0)])),
        ]);
        let pretty = v.to_string_pretty(2);
        assert!(pretty.contains("\n  \"a\": 1"));
        assert!(parse_json(&pretty).is_ok());
    }

    #[test]
    fn read_events_reports_line_number() {
        let path = temp_path("malformed.jsonl");
        let p = path.to_str().unwrap();
        fs::write(p, "{\"ok\":1}\nnot json\n").unwrap();
        let err = read_events(p).unwrap_err();
        assert!(err.contains("line 2"), "err was: {err}");
        let _ = fs::remove_file(p);
    }
}
