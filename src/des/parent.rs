//! Port of `src/des/parent.ts` (module `des::parent`).
//!
//! The host process: stands up the HTTP + websocket servers and forks the child
//! worker once a `{start:true}` websocket message arrives, forwarding the
//! child's messages back to that socket.
//!
//! ## Conversion notes (faithful to the TS shape)
//!
//!   * The top-level server setup + `run` closure → [`run`] (a `pub fn`, NOT
//!     `fn main`; this is a library crate).
//!   * `const program = { stepSize: bgn(500) }` → the local [`Program`] stub
//!     (the `http-server` module is not part of this port — see PORT NOTE).
//!   * `httpServer.on('request', …)` request counter → [`Parent::on_request`].
//!   * `wsServer.on('connection', c => c.on('message', deJSON(…)))` gate → the
//!     ported [`de_json`] + [`Parent::on_message`]: only the first `{start:true}`
//!     message triggers [`Parent::run_child`].
//!   * `child_process.fork(childPath)` → [`std::process::Command`] spawning the
//!     `child` worker; `__dirname + '/child.js'` → [`child_path`] derived from
//!     [`std::env::current_exe`].
//!
//! PORT NOTE: the async runtime that the TS file implicitly relies on (the Node
//! event loop driving `server.on(...)` / `c.on('message', ...)` / the forked
//! child's IPC `k.on('message', ...)`) has no `std` analogue. The connection and
//! request event loops are stubbed: the handler LOGIC is ported as plain
//! methods that a real `tokio` accept loop would invoke, and the child's
//! stdout/stderr are inherited rather than piped through an IPC channel.
//!
//! PORT NOTE: `parent.ts` imported `fisherYatesShuffle`, `sendRaw`,
//! `HasManyOutputConnections`, the `wss` registry and `safe-stringify` but never
//! used them in the body; those imports are dropped (nothing to port).

#![allow(dead_code)]

use std::process::{Command, Stdio};

use crate::des::shared::precision::{bgn, Decimal};
use crate::des::ws_server::ws_server::get_websocket_server;

// PORT NOTE: `./ws-server/ws-server` (`getWebsocketServer`) IS ported — `run`
// wires the real singleton from `crate::des::ws_server::ws_server`. That module
// intentionally abstracts the socket bind (no `TcpListener`: `std` has no async
// socket runtime, and the repo avoids a heavy `tokio` / `tungstenite`
// dependency); the websocket handshake + connection-registry LOGIC is ported and
// unit-tested there. Only `./http-server` (`getHTTPServer`) remains unported
// (not in this port's file set), so it keeps a local stub; `program` is the bag
// `getHTTPServer` received.

/// `const program = { stepSize: bgn(500) }` — the object the TS passed to
/// `getHTTPServer`.
pub struct Program {
    /// `stepSize`.
    pub step_size: Decimal,
}

/// Stub HTTP server handle (the value `getHTTPServer(program)` returned).
pub struct HttpServerStub;

/// `getHTTPServer(program)` — PORT NOTE: no HTTP stack is wired; returns a stub.
fn get_http_server(_program: &Program) -> HttpServerStub {
    HttpServerStub
}

/// `deJSON(cb)` — parse a raw websocket frame into a [`StartMessage`] before
/// handing it to the callback. The TS helper `JSON.parse`d the frame; here we
/// parse with `serde_json` and read just the one field the gate inspects.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StartMessage {
    pub start: bool,
}

/// `deJSON` — parse the frame with `serde_json` and read the `start` flag. A
/// malformed frame or a missing / non-boolean `start` field yields
/// `start: false` (the gate simply never fires), matching the TS behaviour
/// where only a truthy `start` triggers the child.
pub fn de_json(raw: &str) -> StartMessage {
    let start = serde_json::from_str::<serde_json::Value>(raw)
        .ok()
        .and_then(|v| v.get("start").and_then(|s| s.as_bool()))
        .unwrap_or(false);
    StartMessage { start }
}

/// Resolve the child worker path (`path.resolve(__dirname + '/child.js')`).
/// PORT NOTE: maps to a sibling `child` executable next to the current binary;
/// falls back to the bare name when the current exe path is unavailable.
pub fn child_path() -> std::path::PathBuf {
    match std::env::current_exe() {
        Ok(exe) => exe
            .parent()
            .map(|dir| dir.join("child"))
            .unwrap_or_else(|| "child".into()),
        Err(_) => "child".into(),
    }
}

/// Host-process state (the module-level `requestCount` / `started` locals).
pub struct Parent {
    pub program: Program,
    pub request_count: u64,
    pub started: bool,
}

impl Parent {
    pub fn new() -> Self {
        // `const program = { stepSize: bgn(500) }`.
        Parent {
            program: Program {
                step_size: bgn(500.0),
            },
            request_count: 0,
            started: false,
        }
    }

    /// `httpServer.on('request', (a,b) => console.info(...))`.
    pub fn on_request(&mut self, method: &str, url: &str) {
        self.request_count += 1;
        println!(
            "server received request: {} {} {}",
            self.request_count, method, url
        );
    }

    /// The websocket `message` handler body: gate on the first `{start:true}`.
    /// Returns `true` if this message started the child.
    pub fn on_message(&mut self, raw: &str) -> bool {
        let v = de_json(raw);
        println!("got a message: {:?}", v);
        if !self.started && v.start {
            self.started = true;
            self.run_child();
            return true;
        }
        false
    }

    /// `run(c)` — fork the child worker.
    ///
    /// PORT NOTE: the Node IPC channel (`k.on('message', m => c.send(m))`) has no
    /// `std` analogue; the child's stdout/stderr are inherited (the closest
    /// faithful stand-in for `k.stdout.pipe(process.stdout)`), and forwarding the
    /// child's structured messages onto the socket is left to a real port.
    pub fn run_child(&self) {
        let path = child_path();
        let _ = Command::new(path)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn();
    }
}

impl Default for Parent {
    fn default() -> Self {
        Parent::new()
    }
}

/// Entry point (the TS top-level setup; `run` here wires the ported servers).
pub fn run() {
    let mut parent = Parent::new();

    // `const httpServer = getHTTPServer(program)`.
    let _http_server = get_http_server(&parent.program);

    // `const wsServer = getWebsocketServer()` — the real ported singleton
    // (`&'static Mutex<WebsocketServer>`); no socket is bound (see PORT NOTE).
    let _ws_server = get_websocket_server();

    // PORT NOTE: the `wsServer.on('connection', …)` / `httpServer.on('request',
    // …)` event loops are driven by the (stubbed) async runtime; the handlers
    // (`Parent::on_request` / `Parent::on_message`) carry the real logic and are
    // invoked by a real accept loop. `parent` is retained so its `started` /
    // `request_count` state outlives the handler registration.
    let _ = &mut parent;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn de_json_recognises_start() {
        assert!(de_json("{\"start\": true}").start);
        assert!(de_json("{ \"start\":true }").start);
        assert!(!de_json("{\"start\": false}").start);
        assert!(!de_json("{}").start);
    }

    #[test]
    fn message_gate_fires_once() {
        let mut p = Parent::new();
        // Non-start messages do not flip the gate.
        assert!(!p.on_message("{\"ping\":1}"));
        assert!(!p.started);
        // First start message flips it (child spawn is best-effort / ignored).
        let fired = p.on_message("{\"start\":true}");
        assert!(p.started);
        // Whether `run_child` could spawn is environment-dependent; the gate
        // state is what matters.
        let _ = fired;
        // Subsequent start messages are ignored.
        assert!(!p.on_message("{\"start\":true}"));
    }

    #[test]
    fn request_counter_increments() {
        let mut p = Parent::new();
        p.on_request("GET", "/");
        p.on_request("POST", "/x");
        assert_eq!(p.request_count, 2);
    }
}
