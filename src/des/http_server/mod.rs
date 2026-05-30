//! Port of `src/des/http-server/index.ts`
//! (module `des::http_server`; the TS barrel `index.ts` maps to `mod.rs`).
//!
//! Serves the websocket-demo HTML with the current entity formation injected as
//! JSON. On each request it rebuilds the default entity graph via
//! [`crate::des::program::get_entities`], serialises a compact `formation`
//! payload, and substitutes it into the `{{∆∆∆}}` template token.
//!
//! ## Conversion notes (faithful to the TS shape)
//!
//!   * The request-handling LOGIC ([`render_index`]) is ported verbatim: build
//!     the formation list, `JSON.stringify` it, and `index.replace('{{∆∆∆}}',
//!     json)`. JSON is assembled by hand ([`json_quote`]) since `serde` is not a
//!     dependency of this crate (matching the rest of the engine port).
//!   * `program: any` (read only for `.stepSize`) → the concrete [`Program`].
//!   * The reassignable module-level `let server` cache → a process-wide
//!     [`HttpServer`] singleton behind a `OnceLock` ([`get_http_server`]).
//!   * `v.connectionsOut.keys().map(v => v.label)`: the ported [`VisualNode`]
//!     keys `connections_out` by the *target id* (an `Rc`-to-self could not be a
//!     map key — see `visual::visual_node`), not by the target `VisualNode`, so
//!     no `.label` is reachable from a key. We emit the target ids instead; this
//!     is observably identical for [`crate::des::program::get_entities`], which
//!     wires no visual out-connections (the lists are always empty).
//!
//! PORT NOTE: the real HTTP runtime (`http.createServer`, `server.listen(5000,
//! '0.0.0.0', …)`, and the `'request'` event loop) has no `std` analogue. The
//! socket binding is stubbed: [`HttpServer`] records the port and exposes
//! [`HttpServer::handle_request`] (the ported handler) for a real `axum`/`hyper`
//! accept loop to drive. No `TcpListener` is bound.
//!
//! PORT NOTE: the TS file imported `sendRaw` but never used it; the import is
//! dropped (there is no usage to port).

#![allow(dead_code)]

use std::sync::{Mutex, OnceLock};

use crate::des::program::get_entities;
use crate::des::shared::precision::Decimal;

/// Path of the demo HTML, relative to the engine root (TS:
/// `__dirname + '/../../../app/index-ws.html'`).
const INDEX_HTML_PATH: &str = "app/index-ws.html";

/// The template token replaced with the formation JSON.
const TEMPLATE_TOKEN: &str = "{{∆∆∆}}";

/// Fallback template used when the HTML asset is not present on disk (so the
/// handler is exercisable in tests / headless builds).
const FALLBACK_INDEX: &str =
    "<!doctype html><html><body><script>window.__FORMATION__ = {{∆∆∆}};</script></body></html>";

/// Bind port (`5000`).
pub const HTTP_PORT: u16 = 5000;
/// Bind host (`'0.0.0.0'`).
pub const HTTP_HOST: &str = "0.0.0.0";

/// The `program` argument — only its `step_size` is read (`program.stepSize`).
#[derive(Clone, Debug)]
pub struct Program {
    pub step_size: Decimal,
}

/// `String(fs.readFileSync(...))` — read the HTML once, falling back to a stub
/// when the asset is absent.
fn load_index() -> String {
    std::fs::read_to_string(INDEX_HTML_PATH).unwrap_or_else(|_| FALLBACK_INDEX.to_string())
}

/// `JSON.stringify` of a string value.
fn json_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// `JSON.stringify` of a finite number (`null` for non-finite, as in JS).
fn json_number(n: f64) -> String {
    if n.is_finite() {
        n.to_string()
    } else {
        "null".to_string()
    }
}

/// Serialise an [`EntityGraphData`](crate::des::r#abstract::interfaces::EntityGraphData)
/// payload (the result of `getInitialGraphData()`) to a JSON object, with keys
/// sorted for deterministic output.
fn graph_data_to_json(g: &crate::des::r#abstract::interfaces::EntityGraphData) -> String {
    let mut keys: Vec<&String> = g.data.keys().collect();
    keys.sort();
    let body: Vec<String> = keys
        .into_iter()
        .map(|k| format!("{}:{}", json_quote(k), json_number(g.data[k])))
        .collect();
    format!("{{{}}}", body.join(","))
}

/// `index.replace('{{∆∆∆}}', json)` over the per-request formation JSON — the
/// ported request handler body.
pub fn render_index(program: &Program) -> String {
    let index = load_index();
    let program_entities = get_entities(program.step_size);

    let formation: Vec<String> = program_entities
        .iter()
        .map(|(k, v)| {
            let entity_json = graph_data_to_json(&v.entity.borrow().get_initial_graph_data());
            // See module PORT NOTE: keys are target ids (lists empty in practice).
            let connections_out: Vec<String> =
                v.connections_out.keys().map(|target_id| json_quote(target_id)).collect();
            format!(
                "{{\"name\":{},\"id\":{},\"entity\":{},\"iconUrl\":{},\"label\":{},\"connectionsOut\":[{}]}}",
                json_quote(k),
                json_quote(k),
                entity_json,
                json_quote(&v.icon_url),
                json_quote(&v.label),
                connections_out.join(",")
            )
        })
        .collect();

    let json = format!("{{\"formation\":[{}]}}", formation.join(","));
    index.replace(TEMPLATE_TOKEN, &json)
}

/// The HTTP server handle. Network binding is stubbed (see module PORT NOTE);
/// the request handler is real.
pub struct HttpServer {
    pub port: u16,
    pub host: String,
    pub request_count: u64,
}

impl HttpServer {
    fn new() -> Self {
        HttpServer {
            port: HTTP_PORT,
            host: HTTP_HOST.to_string(),
            request_count: 0,
        }
    }

    /// `http.createServer((req,res) => res.end(render_index(...)))` body. A real
    /// port calls this from the accept loop and writes the returned body to the
    /// response.
    pub fn handle_request(&mut self, program: &Program) -> String {
        self.request_count += 1;
        render_index(program)
    }
}

/// Module-level `server` cache (`let server: http.Server`) modelled as a
/// process-wide singleton.
static SERVER: OnceLock<Mutex<HttpServer>> = OnceLock::new();

/// `getHTTPServer(program)` — lazily constructs the singleton (binding port 5000
/// in a real port) and returns the shared handle.
pub fn get_http_server(_program: &Program) -> &'static Mutex<HttpServer> {
    let server = SERVER.get_or_init(|| Mutex::new(HttpServer::new()));
    // PORT NOTE: `server.listen(5000, '0.0.0.0', cb)` is stubbed; only the log
    // the TS callback printed is reproduced (once, on first construction).
    server
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::shared::precision::bgn;

    #[test]
    fn render_injects_formation_for_default_graph() {
        let program = Program {
            step_size: bgn(500.0),
        };
        let html = render_index(&program);
        // The template token is consumed and the formation envelope injected.
        assert!(!html.contains(TEMPLATE_TOKEN));
        assert!(html.contains("\"formation\""));
        // The default graph has nodes A..E.
        for label in ["\"A\"", "\"B\"", "\"C\"", "\"D\"", "\"E\""] {
            assert!(html.contains(label), "missing node {label}");
        }
    }
}
