//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/ws-server/ws-server.ts`
//! Rust target: `src/des/ws_server/ws_server.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/ws-server/ws-server.ts",
    "src/des/ws_server/ws_server.rs",
    &["RUST MIGRATION:", "- Target: src/des/ws_server/ws_server.rs", "- Replace ws.WebSocketServer with axum websocket routes or tokio-tungstenite on tokio.", "- wss.connections should become shared connection state such as Arc<Mutex<HashSet<ConnectionId>>> plus sender handles, not raw socket objects.", "- safe-stringify maps to serde_json::to_string with explicit serializable structs; send/close handlers should return Result.", "- getWebsocketServer should become an async startup function that binds once and exposes typed broadcast/connection APIs."],
    &["getWebsocketServer", "wss"],
);
