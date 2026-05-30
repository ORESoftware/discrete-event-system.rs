//! Port of `src/des/ws-server/ws-server.ts`
//! (module `des::ws_server::ws_server`).
//!
//! A WebSocket server that tracks live connections so they can be broadcast to.
//! The TS module kept a reassignable module-level `server` plus a shared
//! `wss.connections` set; on every new connection it greeted the peer with
//! `{received:true}` and registered/unregistered the socket on open/close.
//!
//! ## Conversion notes (faithful to the TS shape)
//!
//!   * The message-handling LOGIC (greet on connect, add to registry, drop on
//!     close) is ported verbatim into [`WebsocketServer::on_connection`] /
//!     [`WebsocketServer::on_close`] operating over a [`WsConnectionRegistry`].
//!   * `wss.connections: Set<WebSocket>` → [`WsConnectionRegistry`], a set keyed
//!     by a connection id (raw sockets are not `Hash`, exactly as the migration
//!     header noted), holding boxed [`WsConnection`] sinks.
//!   * `safe.stringify({received:true})` → [`RECEIVED_MESSAGE`].
//!   * The reassignable module-level `server` cache → a process-wide singleton
//!     behind a `OnceLock` ([`get_websocket_server`]).
//!
//! PORT NOTE: the real socket runtime (`new WebSocket.Server({host, port})`, the
//! accept loop, and the per-socket `on('message')` / `on('close')` event wiring)
//! has no `std` analogue. The network binding is stubbed: [`WebsocketServer`]
//! records `host`/`port` and exposes the handlers as plain methods that a real
//! `tokio-tungstenite` accept loop would call. No actual `TcpListener` is bound.

#![allow(dead_code)]

use std::sync::{Mutex, OnceLock};

/// The greeting payload pushed to every freshly-connected peer
/// (`safe.stringify({received:true})`).
pub const RECEIVED_MESSAGE: &str = "{\"received\":true}";

/// Bind host (`'0.0.0.0'`).
pub const WS_HOST: &str = "0.0.0.0";
/// Bind port (`6969`).
pub const WS_PORT: u16 = 6969;

/// One live peer connection. `send` is the only capability the server uses on a
/// socket; a real port backs this with a `tokio-tungstenite` sink.
pub trait WsConnection: Send {
    /// Stable id for registry membership (raw sockets are not `Hash`).
    fn id(&self) -> &str;
    /// `c.send(...)`.
    fn send(&self, message: &str);
}

/// `wss = { connections: new Set<WebSocket.WebSocket>() }` — the shared registry
/// of live connections, keyed by connection id.
#[derive(Default)]
pub struct WsConnectionRegistry {
    connections: Vec<Box<dyn WsConnection>>,
}

impl WsConnectionRegistry {
    pub fn new() -> Self {
        WsConnectionRegistry {
            connections: Vec::new(),
        }
    }

    /// `connections.add(c)` (set semantics — no duplicate ids).
    pub fn add(&mut self, c: Box<dyn WsConnection>) {
        if !self
            .connections
            .iter()
            .any(|existing| existing.id() == c.id())
        {
            self.connections.push(c);
        }
    }

    /// `connections.delete(c)`.
    pub fn delete(&mut self, id: &str) -> bool {
        let before = self.connections.len();
        self.connections.retain(|existing| existing.id() != id);
        self.connections.len() != before
    }

    /// `connections.size`.
    pub fn size(&self) -> usize {
        self.connections.len()
    }

    pub fn ids(&self) -> Vec<String> {
        self.connections
            .iter()
            .map(|c| c.id().to_string())
            .collect()
    }
}

/// The websocket server. Holds the bind address and the live-connection
/// registry; the connection lifecycle handlers carry the ported TS logic.
pub struct WebsocketServer {
    pub host: String,
    pub port: u16,
    pub connections: WsConnectionRegistry,
}

impl WebsocketServer {
    /// `new WebSocket.Server({host: '0.0.0.0', port: 6969})`.
    ///
    /// PORT NOTE: no `TcpListener` is bound here (no `std` socket runtime); a
    /// real port binds and spawns the accept loop at this point.
    pub fn new() -> Self {
        WebsocketServer {
            host: WS_HOST.to_string(),
            port: WS_PORT,
            connections: WsConnectionRegistry::new(),
        }
    }

    /// `server.on('connection', c => { ... })` body: greet the peer with
    /// `{received:true}` and register it.
    pub fn on_connection(&mut self, c: Box<dyn WsConnection>) {
        println!("new connection.");
        c.send(RECEIVED_MESSAGE);
        self.connections.add(c);
    }

    /// `c.once('close', () => wss.connections.delete(c))`.
    pub fn on_close(&mut self, connection_id: &str) {
        self.connections.delete(connection_id);
    }

    /// `server.on('close', () => console.log('websocket server closed.'))`.
    pub fn on_server_close(&self) {
        println!("websocket server closed.");
    }
}

impl Default for WebsocketServer {
    fn default() -> Self {
        WebsocketServer::new()
    }
}

/// Module-level `server` cache (`let server: WebSocket.Server | null = null`)
/// modelled as a process-wide singleton, not a reassignable global.
static SERVER: OnceLock<Mutex<WebsocketServer>> = OnceLock::new();

/// `getWebsocketServer()` — lazily constructs the singleton server (binding the
/// socket on first call) and returns the shared handle.
pub fn get_websocket_server() -> &'static Mutex<WebsocketServer> {
    SERVER.get_or_init(|| Mutex::new(WebsocketServer::new()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// A test connection that records what it was sent.
    struct FakeConn {
        id: String,
        sent: Rc<RefCell<Vec<String>>>,
    }

    // The trait is `Send`; the test stays single-threaded, but we satisfy the
    // bound by storing only `Send` data.
    unsafe impl Send for FakeConn {}

    impl WsConnection for FakeConn {
        fn id(&self) -> &str {
            &self.id
        }
        fn send(&self, message: &str) {
            self.sent.borrow_mut().push(message.to_string());
        }
    }

    #[test]
    fn greets_and_registers_then_drops() {
        let mut server = WebsocketServer::new();
        assert_eq!(server.port, 6969);

        let sent = Rc::new(RefCell::new(Vec::new()));
        server.on_connection(Box::new(FakeConn {
            id: "c1".to_string(),
            sent: sent.clone(),
        }));

        assert_eq!(server.connections.size(), 1);
        assert_eq!(sent.borrow().as_slice(), [RECEIVED_MESSAGE.to_string()]);

        // Duplicate id is ignored (set semantics).
        server.on_connection(Box::new(FakeConn {
            id: "c1".to_string(),
            sent: sent.clone(),
        }));
        assert_eq!(server.connections.size(), 1);

        server.on_close("c1");
        assert_eq!(server.connections.size(), 0);
    }
}
