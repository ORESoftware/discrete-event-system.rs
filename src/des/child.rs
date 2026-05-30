//! Port of `src/des/child.ts`.
//!
//! Forked worker process: builds the entity graph, drives it, and (in the TS)
//! streams batched updates back to the parent over IPC.
//!
//! PORT NOTE: this file is fundamentally a Node child-process worker. Several of
//! its dependencies have no analog in the Rust tree and are runtime concerns:
//!   * `getWebsocketServer` / `wss` (`./ws-server/ws-server`) — websocket server
//!     (`ws` → would be `tokio-tungstenite`). Not ported; omitted.
//!   * `process.send` IPC + `@oresoftware/safe-stringify` — the parent channel.
//!     Modeled here by [`Program::send_batch_message_to_parent`] writing to
//!     stdout (a stand-in for the IPC pipe). `safe.stringify` → plain strings.
//!   * `setTimeout` batching (`makeTimeout`) — there is no ambient event loop in
//!     a library crate, so the 2 s flush timer is documented but not scheduled.
//!   * `process.on('SIGCONT' | 'message', …)` — IPC/signal handlers are dropped;
//!     the TS bottom-of-file `run()` invocation maps to [`run`].
//!   * `./program` (`getEntities`) and `./visual/visual-node` wiring — the
//!     `program` module is NOT in the Rust tree, so `get_entities` is stubbed
//!     (empty ordered set). The doubled loop / Fisher–Yates drive structure is
//!     preserved against that (currently empty) list.

#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::des::general::general::fisher_yates_shuffle;
use crate::des::r#abstract::r#abstract::Entity;
use crate::des::shared::capabilities::SeededRandom;
use crate::des::shared::precision::{bgn, Decimal};

/// Mirrors the TS `program` object: step size, pending batches, and flags.
struct Program {
    step_size: Decimal,
    batches: Vec<String>,
    stop: bool,
    turn_off_sources: bool,
    running: bool,
}

impl Program {
    fn new() -> Self {
        // `bgn(parseInt(process.env.step_size || '500'))`.
        let step = std::env::var("step_size")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(500.0);
        Program {
            step_size: bgn(step),
            batches: Vec::new(),
            stop: false,
            turn_off_sources: false,
            running: false,
        }
    }

    /// `sendMessageToParent(m)` — buffer, flushing once more than 10 accumulate.
    fn send_message_to_parent(&mut self, m: String) {
        self.batches.push(m);
        if self.batches.len() > 10 {
            let v = std::mem::take(&mut self.batches);
            self.send_batch_message_to_parent(v);
        }
    }

    /// `sendBatchMessageToParent(m)` — TS resets the flush timer and calls
    /// `process.send`. Here the IPC pipe is stdout (see PORT NOTE).
    fn send_batch_message_to_parent(&mut self, m: Vec<String>) {
        // PORT NOTE: `clearTimeout(program.to); program.to = makeTimeout();` — no
        // event loop to (re)schedule the 2 s flush against.
        for line in m {
            println!("{line}");
        }
    }
}

// PORT NOTE: local stub for `./program::getEntities` (module not ported). Returns
// the ordered entity list the TS run() would build (A..E) — empty here.
fn get_entities(_step_size: Decimal) -> Vec<(String, Rc<RefCell<dyn Entity>>)> {
    Vec::new()
}

/// `const run = () => {...}` — build the graph, validate, and drive it.
fn run_program(program: &mut Program) {
    let program_entities = get_entities(program.step_size);

    // TS: subscribe each node and wire A->B->C->D->E (entity + visual edges).
    // PORT NOTE: subscription + `addOutConnection`/`addVisualConnectionOut`
    // wiring lived in the un-ported `program`/`visual-node` modules.

    let mut program_list = program_entities;
    for (_, v) in &program_list {
        v.borrow_mut().do_validation_before_run();
    }

    let mut rng = SeededRandom::new(1);

    // First phase: 10 ticks over a Fisher–Yates-shuffled order.
    'outer1: for _ in 0..10 {
        if program.stop {
            eprintln!("stop flag was flipped, breaking out of loop.");
            break;
        }
        fisher_yates_shuffle(&mut program_list, &mut rng);
        for (_, v) in &program_list {
            if program.stop {
                eprintln!("stop flag was flipped, breaking out of loop.");
                break 'outer1;
            }
            v.borrow_mut().do_time_step(program.step_size);
        }
    }

    // `(global as any).turnOffSources = true;`
    program.turn_off_sources = true;

    // Second phase: 100 ticks.
    'outer2: for _ in 0..100 {
        if program.stop {
            eprintln!("stop flag was flipped, breaking out of loop.");
            break;
        }
        fisher_yates_shuffle(&mut program_list, &mut rng);
        for (_, v) in &program_list {
            if program.stop {
                eprintln!("stop flag was flipped, breaking out of loop.");
                break 'outer2;
            }
            v.borrow_mut().do_time_step(program.step_size);
        }
    }

    let mut i = 0;
    for (_, e) in &program_list {
        i += 1;
        println!("{i} {i} {i} {i} {i} {i} {i} {i} {i} **************************************");
        let _ = e.borrow().get_with_computed_properties();
    }
}

/// Entry point (TS bottom-of-file `run()` invocation).
pub fn run() {
    let mut program = Program::new();
    if !program.running {
        program.running = true;
        println!("running loop.");
        run_program(&mut program);
    }
}
