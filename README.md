# discrete-event-system.rs (`des_engine`)

A Rust **library / SDK for modeling, simulating, solving, and rendering** discrete,
continuous, and mixed (hybrid) systems. It is a faithful Rust port of the
TypeScript `des-engine`, with the source mapping 1:1 (`src/des/<path>.ts` →
`src/des/<path>.rs`).

One engine spans many modeling paradigms as **peers** — discrete-event (FEL)
networks, MDP / POMDP decision processes, hybrid continuous+discrete block
diagrams, visual-block dataflow, and a family of optimization solvers (LP, MILP,
shortest-path, network-flow, simulated annealing, …). Every paradigm produces
the **same uniform output**: a `RunArtifact` that carries an animated frame
stream *and* a results document, and that renders to a self-contained
interactive HTML player with no extra dependencies.

```text
describe a model  ──►  run it  ──►  RunArtifact { frames, results }  ──►  HTML player  +  JSON results
   (JSON spec)         (in-proc)     (uniform, paradigm-neutral)          (report)        (data)
```

- **Crate:** `des_engine` (edition 2021, `MIT OR Apache-2.0`)
- **Dependencies:** intentionally tiny — `serde`/`serde_json` only at the I/O
  boundary, `rust_decimal`/`num-rational` for exact arithmetic, `uuid` for entity
  ids. No async runtime, no web framework: **the engine is transport-agnostic**,
  so you embed it behind whatever server/CLI/desktop shell you like.

---

## Table of contents

- [Install](#install)
- [The SDK surface](#the-sdk-surface)
- [Quickstart](#quickstart)
- [Embedding in a server (run & serve reports + results)](#embedding-in-a-server-run--serve-reports--results)
- [The simulation catalogue](#the-simulation-catalogue)
- [Output artifacts](#output-artifacts)
- [Module map](#module-map)
- [Build & test](#build--test)

---

## Install

Add it as a path or git dependency (not yet published to crates.io):

```toml
# Cargo.toml
[dependencies]
des_engine = { git = "https://github.com/ORESoftware/discrete-event-system.rs" }
# or, as a submodule / monorepo path:
# des_engine = { path = "../discrete-event-system.rs" }
```

Everything an embedder needs is re-exported from one shallow import:

```rust
use des_engine::prelude::*;
```

---

## The SDK surface

The full engine lives under `des_engine::des::*`, but consumers should build
against the **first-class seams** gathered in `des_engine::prelude` (and mirrored
under `des_engine::sdk`). The seams are deliberately **JSON-first**, so the same
contracts work in-process, across an HTTP boundary, or over IPC.

| Seam | What it gives you | Key items |
| --- | --- | --- |
| **Model contract** | Describe a model as JSON → validate → run → uniform artifact | `with_builtins`, `CitizenRegistry`, `ModelCitizen`, `ModelDescriptor`, `RunArtifact` |
| **Simulation catalogue** | ~60 ready-made simulations that render HTML/JSON reports | `simulation_catalogue`, `run_simulations_matching`, `run_all_simulations` |
| **Streaming solvers** | Drive an iterative solver (LP/MILP/MDP/POMDP) over JSONL | `run_named_jsonl`, `run_jsonl`, `streaming_contracts`, `StreamContract` |
| **Visual-block dataflow** | Build a flat block graph and run it as signal dataflow | `StudioGraph`, `RuntimeCell`, `RuntimeOp`, `Composite`, `CompiledStudio` |
| **Executive selection** | Route a graph to the simplest engine that can run it | `select`, `requirements_for_studio`, `Executive`, `StudioExecutive`, `HybridExecutive` |
| **Plugins** | Run an external program and render its output as a player | `PluginManifest`, `run_and_render`, `PluginTransport`, `ProcessTransport` |
| **Service discovery** | Self-describe a server's routes/capabilities as JSON | `ServiceBuilder`, `ServiceDescriptor`, `DesExtension` |

`des_engine::sdk::surface()` returns the crate name, version, and the list of
SDK modules — handy to expose in your own server's `/info` endpoint.

---

## Quickstart

### 1. Run a first-class model from a JSON spec

```rust
use des_engine::prelude::*;
use serde_json::json;

let registry = with_builtins(); // pre-loaded: mdp, pomdp, hybrid, studio

// Discover what's available and the example spec to start from.
for d in registry.descriptors() {
    println!("{} — schema {} — methods {:?}", d.kind, d.spec_schema, d.methods);
}

// Run a spec (use a descriptor's `example_spec` as a template).
let spec = json!({ "$schema": "des/mdp/v1", /* states, actions, transitions, ... */ });
let artifact: RunArtifact = registry.run("mdp", &spec)?;

// Two uniform outputs from the same artifact:
let report_html: String = artifact.to_player_html(); // a self-contained interactive page
let results = &artifact.results;                      // the solved policy/value/solution (JSON)
# Ok::<(), CitizenError>(())
```

A bad spec comes back as `Err(CitizenError::InvalidSpec(msg))` with a message
phrased so a user (or an LLM) can self-correct — the engine never panics out of
`run`.

### 2. Stream an iterative solver over JSONL

```rust
use des_engine::prelude::*;

// Reads JSONL commands from `reader`, writes JSONL frames (one per iteration)
// to `writer`. Perfect for piping a solver's progress to a client.
run_named_jsonl("lp", std::io::stdin().lock(), &mut std::io::stdout())?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

### 3. Build and run a visual-block dataflow graph

```rust
use des_engine::prelude::*;
use des_engine::des::studio::{signal_chain};

// A ready-made demo graph (ramp → gain▸saturation → sink), or build your own
// with StudioGraph/VisualNode/RuntimeCell.
let mut demo = signal_chain()?;
let run = des_engine::des::studio::run(&mut demo.compiled, demo.steps, demo.dt);
println!("final output = {:?}", run.final_value("output"));
# Ok::<(), des_engine::des::studio::StudioError>(())
```

---

## Embedding in a server (run & serve reports + results)

The engine is built to be wrapped by a thin HTTP server. There are **three ways
to serve work**, and a typical service exposes all three:

1. **Model runs** — accept a JSON spec, run it in-process, and respond with
   either the rendered HTML **report** (`artifact.to_player_html()`) or the raw
   **results** (`artifact.results` / `artifact.to_jsonl()`).
2. **Catalogue simulations** — run one or more catalogue sims that render
   `out/*.html` report pages, then serve the `out/` directory as static files.
3. **Streaming solves** — pipe a request body of JSONL commands into a solver
   and stream JSONL frames back.

### The pattern

```rust
use std::sync::Arc;
use axum::{
    extract::{Path, State},
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use tokio::sync::Mutex;
use des_engine::prelude::*;
use des_engine::des::simulations::{run_simulations_matching, simulation_catalogue};

struct AppState {
    registry: CitizenRegistry,
    // Catalogue sims drive a process-global clock/RNG and `println!` a report,
    // so they MUST run one at a time. Serialize them behind a lock.
    sim_lock: Mutex<()>,
}

#[tokio::main]
async fn main() {
    // Catalogue sims write `out/*.html` RELATIVE to the working dir — point the
    // process at a writable dir you can also serve as static files.
    let work_dir = std::env::var("DES_WORK_DIR").unwrap_or_else(|_| "/tmp/des-work".into());
    std::fs::create_dir_all(&work_dir).unwrap();
    std::env::set_current_dir(&work_dir).unwrap(); // sims now render into ./out/

    let state = Arc::new(AppState {
        registry: with_builtins(),
        sim_lock: Mutex::new(()),
    });

    let app = Router::new()
        .route("/healthz", get(|| async { Json(json!({ "ok": true })) }))
        // ── discovery ─────────────────────────────────────────────
        .route("/models", get(list_models))
        .route("/simulations", get(list_sims))
        // ── 1. model runs: report (HTML) or results (JSON) ─────────
        .route("/models/:kind/run", post(run_model))
        // ── 2. catalogue sims → render out/*.html, then serve them ─
        .route("/simulate/:needle", post(run_sims))
        .nest_service("/out", tower_http::services::ServeDir::new("out"))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8112").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// Discovery: advertise the model kinds + their example specs.
async fn list_models(State(s): State<Arc<AppState>>) -> impl IntoResponse {
    Json(json!({ "models": s.registry.descriptors() }))
}

async fn list_sims() -> impl IntoResponse {
    let names: Vec<&str> = simulation_catalogue().into_iter().map(|(n, _)| n).collect();
    Json(json!({ "count": names.len(), "simulations": names }))
}

// 1. Run a user-supplied spec and return a REPORT (HTML) or RESULTS (JSON).
async fn run_model(
    State(s): State<Arc<AppState>>,
    Path(kind): Path<String>,
    Json(spec): Json<Value>,
) -> Response {
    match s.registry.run(&kind, &spec) {
        Ok(artifact) => Html(artifact.to_player_html()).into_response(), // interactive report
        // …or `Json(artifact.results).into_response()` for the raw solution,
        // or `artifact.to_jsonl()` for the frame stream.
        Err(e) => (axum::http::StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    }
}

// 2. Run catalogue sims (serialized); they render out/*.html as a side effect.
async fn run_sims(
    State(s): State<Arc<AppState>>,
    Path(needle): Path<String>,
) -> impl IntoResponse {
    let _guard = s.sim_lock.lock().await; // never run two at once
    let outcomes = tokio::task::spawn_blocking(move || run_simulations_matching(&needle))
        .await
        .unwrap();
    // Each ok outcome left an `out/<name>.html` report behind, now served at /out/.
    Json(json!({
        "ran": outcomes.iter().map(|o| json!({ "name": o.name, "ok": o.ok, "ms": o.millis }))
                       .collect::<Vec<_>>()
    }))
}
```

> Add `axum`, `tokio` (`features = ["full"]`), and `tower-http` (`features =
> ["fs"]`) to *your server's* `Cargo.toml`; `des_engine` itself pulls in none of
> them.

### Key embedding facts

- **Run sims serially.** The engine drives a process-global simulation clock and
  RNG and prints its report to stdout; running two simulations concurrently
  interleaves output and races shared state. Guard catalogue runs with a lock
  (and offload the blocking work with `spawn_blocking` under async runtimes).
  `run_all_simulations()` / `run_simulations_matching()` are already strictly
  serial and **panic-isolated** (each entry is `catch_unwind`-wrapped and reported
  as `SimOutcome { ok: false }` rather than taking down the process).
- **Working directory = where reports land.** Catalogue sims write `out/*.html`,
  `out/*-framework.json`, and JSONL frame files **relative to the current
  directory**. `chdir` into a writable dir at startup and serve that `out/` dir.
- **`RunArtifact` is the universal currency.** `to_player_html()` → a
  self-contained interactive report; `results` → the machine-readable solution;
  `to_jsonl()` → the raw frame stream. Pick per request (e.g. a `?format=json`
  query param).
- **Self-describe via `service`.** Use `ServiceBuilder` / `ServiceDescriptor` to
  emit a JSON inventory of your routes and capabilities, and the RFC 8288 `Link`
  header for crawlers — so the JSON is the one source of truth and your docs page
  is just a view over it.

### Reference implementation

The production **`dd-des-rs`** axum service (in the `k8s-cluster` repo under
`remote/deployments/des-rs/`) is a complete, deployed example of this pattern. It
exposes `/models`, `/models/:kind/run`, `/simulate`, `/streaming/:name`,
`/out/*`, `/healthz`, generated API docs, and a curated landing page — all over
the seams above. It fetches the engine's `main` at pod start and rebuilds, so the
deployment tracks this library automatically.

---

## The simulation catalogue

~60 entry points (and ~97 runnable demo binaries) cover a broad span of domains —
elevators (queue-based and next-event/FEL), epidemics (SEIR), traffic networks,
electric circuits, DC-motor control (with back-EMF), MPC/LQR controllers,
controllability/observability analysis (including shadow/dual Gramian
evaluation), MDP/POMDP problems, neural networks, genetic algorithms, network
flow, and more.

Run any one as a standalone binary:

```bash
cargo run --release --bin main_elevator_highrise
cargo run --release --bin main_dc_motor_anim
cargo run --release --bin main_build_site   # assembles a curated out/index.html
```

Or enumerate/drive them from code with `simulation_catalogue()` and
`run_simulations_matching("elevator")`.

---

## Output artifacts

Everything a simulation produces is written under `out/`:

- `out/<name>.html` — a self-contained interactive player (the **report**); open
  it directly in a browser, no server required.
- `out/<name>-framework.json`, `out/<name>.summary.json` — structured **results**.
- `out/<name>.frames.jsonl` — the raw animation frame stream (one JSON object per
  step), the same data `RunArtifact::to_jsonl()` returns.

These are exactly the files an embedding server serves as static content (see the
`/out` route above).

---

## Module map

```text
des_engine
├── prelude            # the curated SDK import (use des_engine::prelude::*)
├── sdk                # crate name/version + SDK module list for diagnostics
└── des
    ├── model          # first-class-citizen contract: ModelCitizen, RunArtifact, CitizenRegistry
    ├── exec            # executive-selection seam (studio | hybrid | des-run-loop)
    ├── studio          # two-layer visual-block + runtime-cell dataflow engine
    ├── hybrid          # continuous RK4 + multirate discrete + zero-crossing events
    ├── fel             # next-event (Future Event List) discrete-event simulation
    ├── decision        # MDP / POMDP specs, solvers, and visualization
    ├── streaming       # JSONL streaming-solver contracts (lp, milp, mdp, pomdp)
    ├── plugin          # external-program plugin system + self-contained HTML player
    ├── service         # service self-description + discovery (JSON-first)
    ├── shared          # dependency-free foundation (Transform trait, linalg, RNG, …)
    ├── general         # the broad model/solver library (control, optimization, RL, …)
    └── main_*          # runnable simulation demos (entry points, not SDK surface)
```

---

## Build & test

```bash
cargo build --release      # build the library + all demo binaries
cargo test                 # run the unit/integration test suite
cargo clippy --all-targets # lints
cargo doc --no-deps --open # API docs (library only; the main_* bins are excluded)
```

The library is `#![forbid]`-clean of panics on its public `run`/validate seams:
all recoverable failures surface as typed errors (`CitizenError`, `StudioError`,
`PluginError`, …), and catalogue runs are panic-isolated.
