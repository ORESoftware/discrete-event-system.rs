# discrete-event-system.rs (`des_engine`)

`des_engine` is a Rust library/SDK for modeling, simulating, solving, and
rendering discrete, continuous, and mixed systems. It is the Rust port of the
ORESoftware TypeScript discrete-event-system engine, with a deliberately small,
JSON-first SDK surface for embedders: web servers, desktop apps, CLIs, workers,
notebooks, and plugin hosts.

The crate is not just a collection of demo binaries. The important part is the
library surface:

- Run models from JSON specs through one uniform API.
- Solve MDPs, POMDPs, LPs, MILPs, hybrid block diagrams, future-event-list
  simulations, and time-stepped simulations.
- Return machine-readable results as JSON.
- Return frame streams as JSONL for playback, storage, or live streaming.
- Render self-contained HTML players and report pages that can be served by any
  HTTP server.
- Advertise an embedding service through a machine-readable discovery
  descriptor.
- Run external simulation plugins that emit JSON or JSONL and render them with
  the same built-in player.

```text
JSON spec / command stream
        |
        v
des_engine runner / solver
        |
        v
RunArtifact { frames, results, summary }
        |
        +--> report.html  (interactive HTML player)
        +--> results.json (machine-readable result document)
        +--> frames.jsonl (one frame/event per line)
```

## Contents

- [Install](#install)
- [What This SDK Is For](#what-this-sdk-is-for)
- [The SDK Surface](#the-sdk-surface)
- [Quickstart](#quickstart)
- [Embedding In A Server](#embedding-in-a-server)
- [Reports And Artifacts](#reports-and-artifacts)
- [Running The Included Demos](#running-the-included-demos)
- [Development](#development)
- [Module And Build Performance Plan](#module-and-build-performance-plan)
- [Module Map](#module-map)

## Install

Add it as a git or path dependency:

```toml
[dependencies]
des_engine = { git = "https://github.com/ORESoftware/discrete-event-system.rs" }

# or, in a monorepo:
# des_engine = { path = "../discrete-event-system.rs" }
```

Most embedders start with the curated prelude:

```rust
use des_engine::prelude::*;
```

The full engine is still available under `des_engine::des::*` when you need a
specific model family or lower-level API.

## What This SDK Is For

Use `des_engine` when you want a Rust service to accept a model/specification,
run a simulation or solver, and serve the outcome back to another system.

Typical use cases:

- A backend API where clients submit simulation specs and receive
  `results.json`, `frames.jsonl`, and `report.html`.
- A dashboard that compares traffic, elevator, epidemic, queueing, control,
  optimization, or decision-process runs.
- A worker that runs a catalogue of simulations on a schedule and stores the
  artifacts.
- An AI-assisted modeling service where an LLM emits a JSON spec, the engine
  validates/runs it, and the UI renders the returned artifact.
- A plugin host where third-party binaries produce JSON/JSONL and the SDK turns
  that output into a standard result or player.

The engine is JSON-first on purpose. JSON is the boundary for HTTP, IPC,
plugins, and persistence. Internally the crate uses strongly typed Rust APIs;
externally, embedders can treat runs as portable artifacts.

## The SDK Surface

The stable embedding surface is gathered in `des_engine::prelude` and mirrored
under `des_engine::sdk`.

| Seam | What it gives you | Key items |
| --- | --- | --- |
| Model contract | Describe a model as JSON, validate it, run it, and get one uniform artifact | `with_builtins`, `CitizenRegistry`, `ModelCitizen`, `ModelDescriptor`, `RunArtifact` |
| Acausal equations | Compile ModelingToolkit-style explicit ODE/algebraic specs with alias elimination, diagnostics, UI metadata, and RK4/Euler simulation | `AcausalModelSpec`, `compile_acausal_model`, `simulate_acausal_model`, `acausal_workbench_descriptor` |
| Schema-backed authoring | Validate, simulate, inspect, and generate Rust from typed model-graph specs | `AuthoringSpec`, `authoring_json_schema`, `compile_hybrid_graph`, `generate_rust` |
| Graph specs and codegen | Save/load Studio and Hybrid diagrams as schema-derived JSON, validate them through typed Rust specs, and emit Rust runner source | `ModelGraphSpec`, `StudioModelSpec`, `HybridModelSpec`, `model_graph_json_schema`, `studio_model_json_schema`, `hybrid_model_json_schema`, `studio::generate_rust_code`, `hybrid::generate_rust_code` |
| Streaming solvers | Drive iterative solvers over JSONL | `run_named_jsonl`, `run_jsonl`, `streaming_contracts`, `StreamContract` |
| Plugins | Run an external program and render its JSON/JSONL output | `PluginManifest`, `run_and_render`, `PluginTransport`, `ProcessTransport` |
| Service discovery | Self-describe a server's routes and capabilities as JSON | `ServiceBuilder`, `ServiceDescriptor`, `DesExtension` |
| Equation-based modeling | Run JSON/LaTeX/XML equation specs as first-class Modelica-style citizens | `EquationCitizen`, `EQUATION_SCHEMA` |
| Visual block dataflow | Build or load a flat block graph and run it as signal dataflow | `StudioGraph`, `RuntimeCell`, `RuntimeOp`, `Composite`, `CompiledStudio`, `demo_from_spec`, `STUDIO_SPEC_SCHEMA` |
| Studio analysis and UI | Inspect Studio specs with N2 dependency data, design variables/objectives/constraints, sweep drivers, self-contained workbench pages, and generated run/N2/sweep players | `analyze_model_spec`, `run_design_sweep`, `render_workbench_html`, `write_workbench_html`, `write_studio_player_html` |
| Design studies | Tune studio-spec parameters against final signal objectives | `run_design_study`, `StudioDesignStudy`, `StudioDesignVariable` |
| Executive selection | Route a graph to the simplest engine that can run it | `select`, `requirements_for_studio`, `Executive`, `StudioExecutive`, `HybridExecutive` |

`des_engine::sdk::surface()` returns the crate name, version, and the list of
SDK modules. It is useful for `/info` or diagnostics endpoints in an embedding
server.

The acausal equation surface is the first open-source JuliaSim/ModelingToolkit
directional slice in this crate: model documents carry variables, parameters,
units, explicit derivative equations, algebraic assignments, and alias/connect
equations. The compiler eliminates aliases, orders algebraics, rejects algebraic
loops, emits structural diagnostics, and returns the same `RunArtifact` +
HTML-player output as the rest of the SDK.

## Quickstart

### Run a first-class model from a JSON spec

The `des::model` module defines the paradigm-neutral model contract:

- `ModelDescriptor`: discovery metadata for a model kind.
- `ModelCitizen`: validates and runs a JSON spec.
- `CitizenRegistry`: registry of runnable model kinds.
- `RunArtifact`: the uniform output of a run.

The built-in registry currently includes acausal, MDP, POMDP, authoring,
hybrid, equation, and studio model citizens. Studio and Hybrid specs are graph
documents whose JSON Schema is generated from Rust types with `schemars`, so
editor payloads, saved files, and generated Rust runners share the same
contract.

```rust
use des_engine::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let registry = with_builtins();

    for descriptor in registry.descriptors() {
        println!(
            "{} - schema {} - methods {:?}",
            descriptor.kind, descriptor.spec_schema, descriptor.methods
        );
    }

    let descriptor = registry
        .get("mdp")
        .expect("built-in MDP citizen")
        .descriptor();
    let artifact = registry.run("mdp", &descriptor.example_spec)?;

    let report_html = artifact.to_player_html();
    let frames_jsonl = artifact.to_jsonl();
    let results_json = artifact.results;

    println!("summary: {}", artifact.summary);
    println!("report bytes: {}", report_html.len());
    println!("frame bytes: {}", frames_jsonl.len());
    println!("result keys: {:?}", results_json.as_object().map(|o| o.len()));

    Ok(())
}
```

Bad specs return `Err(CitizenError::InvalidSpec(msg))` with a recoverable error
message instead of panicking out of the embedding application.

### Stream an iterative solver over JSONL

The `des::streaming` module exposes long-lived JSONL contracts for iterative
solvers:

- `lp`
- `milp` / `mip` / `ip`
- `mdp`
- `pomdp`

Each model consumes one JSON command per line and emits one JSON frame per line.
Malformed commands become error frames where possible, so a stream can continue
after bad input.

```rust
use des_engine::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = br#"{"op":"init","numStates":2,"gamma":0.9}
"#;

    let mut output = Vec::new();
    let handled = run_named_jsonl("mdp", &input[..], &mut output)?;

    assert!(handled);
    println!("{}", String::from_utf8(output)?);
    Ok(())
}
```

### Render an external plugin

The `des::plugin` module lets a host run any program that writes JSON or JSONL
to stdout. The SDK parses the output and renders it through the same HTML
player used by in-process artifacts.

```rust
use des_engine::des::plugin::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest = PluginManifest {
        id: "mm1".into(),
        name: "M/M/1 queue".into(),
        version: "1.0.0".into(),
        description: "Queue length over time".into(),
        runtime: PluginRuntimeKind::Rust,
        transport: PluginTransportKind::Stdio,
        language: None,
        run: RunSpec::new("./target/release/mm1-plugin", &[]),
        output: OutputKind::Jsonl,
        player: PlayerKind::Sim,
        controls: vec![UiControl::range("speed", "Speed (fps)", 1.0, 30.0, 1.0, 8.0)],
        title: None,
    };

    let html = run_and_render(&manifest)?;
    std::fs::write("out/mm1.html", html)?;
    Ok(())
}
```

## Embedding In A Server

`des_engine` does not require a specific web framework. A server integration
usually has four thin layers:

1. Parse the request body into `serde_json::Value`.
2. Route the request to a model citizen, streaming solver, catalogue simulation,
   or plugin.
3. Store the returned artifact under a run id.
4. Serve the artifact as JSON, JSONL, and/or HTML.

The core run path looks like this:

```rust
use des_engine::prelude::*;
use serde_json::Value;

pub struct StoredRun {
    pub kind: String,
    pub results: Value,
    pub frames_jsonl: String,
    pub report_html: String,
    pub summary: String,
}

pub fn run_model(kind: &str, spec: Value) -> Result<StoredRun, CitizenError> {
    let registry = with_builtins();
    let artifact = registry.run(kind, &spec)?;

    Ok(StoredRun {
        kind: artifact.kind.clone(),
        results: artifact.results.clone(),
        frames_jsonl: artifact.to_jsonl(),
        report_html: artifact.to_player_html(),
        summary: artifact.summary.clone(),
    })
}
```

An HTTP server can expose routes like:

- `POST /runs/{kind}`: call `run_model(kind, body_json)`, assign a run id, store
  the artifact, and return `{ "id": "...", "summary": "..." }`.
- `GET /runs/{id}/results.json`: return `StoredRun.results` as
  `application/json`.
- `GET /runs/{id}/frames.jsonl`: return `StoredRun.frames_jsonl` as
  `application/x-ndjson`.
- `GET /runs/{id}/report.html`: return `StoredRun.report_html` as
  `text/html; charset=utf-8`.
- `GET /api/docs.json`: return `ServiceDescriptor::to_json_string()`.

With Axum or another async Rust framework, handlers stay small because the SDK
owns validation, solving, and rendering:

```rust,ignore
async fn create_run(
    Path(kind): Path<String>,
    State(store): State<RunStore>,
    Json(spec): Json<serde_json::Value>,
) -> Result<Json<CreateRunResponse>, ApiError> {
    let stored = tokio::task::spawn_blocking(move || run_model(&kind, spec))
        .await
        .map_err(ApiError::Join)?
        .map_err(ApiError::Citizen)?;

    let id = store.insert(stored).await;
    Ok(Json(CreateRunResponse { id }))
}

async fn get_results(
    Path(id): Path<String>,
    State(store): State<RunStore>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let run = store.get(&id).await.ok_or(ApiError::NotFound)?;
    Ok(Json(run.results))
}

async fn get_report(
    Path(id): Path<String>,
    State(store): State<RunStore>,
) -> Result<Html<String>, ApiError> {
    let run = store.get(&id).await.ok_or(ApiError::NotFound)?;
    Ok(Html(run.report_html))
}
```

For streaming solvers, route the request body into `run_named_jsonl` and return
the emitted bytes as `application/x-ndjson`:

```rust
use des_engine::prelude::*;

pub fn solve_stream(model: &str, request_jsonl: &[u8]) -> std::io::Result<Option<Vec<u8>>> {
    let mut response_jsonl = Vec::new();
    let handled = run_named_jsonl(model, request_jsonl, &mut response_jsonl)?;

    if handled {
        Ok(Some(response_jsonl))
    } else {
        Ok(None)
    }
}
```

### Service discovery

The `des::service` module builds a JSON service descriptor that an embedding
server can expose at `/api/docs.json`. The descriptor advertises routes,
capabilities, registered extensions, and canonical docs locations.

```rust
use des_engine::des::service::{
    EndpointKind, EngineCatalogExtension, ServiceBuilder, ServiceInfo,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = ServiceBuilder::new(ServiceInfo {
        name: "simulation-api".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        description: "HTTP API backed by des_engine".to_string(),
    });

    builder
        .endpoint("GET", "/health", "Health check", EndpointKind::Service)
        .endpoint("POST", "/runs/{kind}", "Run a model spec", EndpointKind::Action)
        .endpoint("GET", "/runs/{id}/results.json", "Run results", EndpointKind::Action)
        .endpoint("GET", "/runs/{id}/frames.jsonl", "Run frame stream", EndpointKind::Action)
        .endpoint("GET", "/runs/{id}/report.html", "HTML player/report", EndpointKind::Action);

    builder.register(Box::new(EngineCatalogExtension))?;

    let descriptor = builder.build();
    println!("{}", descriptor.to_json_string());
    println!("Link: {}", descriptor.link_header_relative());
    Ok(())
}
```

### Catalogue simulations

The crate ships many runnable simulation binaries under `src/bin`, and the
library exposes the same catalogue through `des::simulations`.

Catalogue/report demos generally write HTML and JSON artifacts under `out/`
relative to the current working directory. In a server, run them from a writable
work directory and serve that `out/` directory as static content. Because
catalogue demos may print to stdout and use process-local simulation state, run
them serially from a server worker instead of launching multiple catalogue demos
inside the same process at once.

### Supply-chain optimization coverage

The supply-chain stack is represented across the `des::general` modules rather
than as one monolithic planner:

| Technique | Crate coverage |
| --- | --- |
| Inventory control and dynamic programming | `inventory_dp`, `multistage_stochastic`, `main_inventory_mdp`, `main_newsvendor` |
| Forecasting and state estimation | `nonlinear_forecasting_model`, `kalman_filter`, stochastic SDE/control reports |
| LP and interior-point methods | `lp`, `lp_des`, `des_lp_bridge`; use `LP_SOLVER=internal-ipm` for the native primal-dual IPM or `scipy:highs-ipm` for SciPy HiGHS IPM |
| Mixed-integer optimization | `milp_bnb`, `ip_mip_des` |
| Network flow and transportation | `max_flow`, `network_flow`, `traffic_flow`, `stochastic_flow_mdp` |
| Vehicle routing and routing heuristics | `classical_optimization_models` VRP savings and nearest-neighbor runs |
| Stochastic optimization and simulation | `stochastic_lp`, `statistical_optimization`, `fel`, `hybrid`, catalogue simulations |
| Model predictive control and reinforcement learning | `mpc_double_integrator`, `temp_control` MPC, `qlearning_des`, `ppo_des`, `actor_critic_gridworld` |

### PyDy / OpenMDAO-style coverage

The crate is moving toward an open, Rust-native modeling workbench that spans
PyDy-style dynamics workflows and OpenMDAO-style system optimization:

| Capability | Current coverage |
| --- | --- |
| Model specification | JSON-first `ModelCitizen` specs, studio block diagrams, hybrid demos, plugin manifests |
| Simulation | FEL, hybrid continuous/discrete/event runs, studio dataflow, MDP/POMDP rollout, catalogue simulations |
| Visualization | Uniform `RunArtifact` HTML player, report pages, frame JSONL, studio workbench |
| Parameter reruns | Studio workbench edits the JSON spec, reruns in-browser, and exports the updated model |
| Components and connections | Studio `VisualNode` blocks with typed scalar ports and an N2-style connection matrix |
| Design variables/objectives/drivers | Studio `design` specs with variables, objective targets, finite-difference sensitivities, gradient-descent traces, and an Optimize UI |
| Recording | Run artifacts preserve results JSON, frame streams, summaries, and design traces |

The next high-impact gaps are symbolic multibody equation generation and 3D
body/camera/light visualization on the PyDy side, plus richer OpenMDAO-like
groups, analytic derivatives, constraints, nonlinear solvers, and case readers.

## Reports And Artifacts

The SDK has two related output styles:

- `RunArtifact` player output: lightweight, uniform output for model citizens
  and plugin runs. Use `to_player_html()`, `to_jsonl()`, and `results`.
- Studio workbench output: `write_workbench_html("out/studio/workbench.html",
  &starter_model_spec())` emits a browser UI with palette, canvas, inspector,
  N2 matrix, JSON editor, local run, and design-variable sweep views.
- Studio player output: `write_studio_player_html("out", &starter_model_spec())`
  emits `run-player.html`, `n2-player.html`, and `sweep-player.html` under
  `out/studio/` for catalog-ready animation playback.
- Report pages: richer narrative pages used by demo/report binaries. These are
  built with `des::animation::run_report` and written by binaries such as
  `main_stochastic_sde_report` and `main_empirical_control_report`.

For a server, prefer `RunArtifact` as the stable API return type. Use report
pages when you want a curated human-readable page for a specific simulation
family.

Recommended content types:

- Results JSON: `application/json`
- Frame streams: `application/x-ndjson`
- HTML players/reports: `text/html; charset=utf-8`
- Service descriptor: `application/json`

Recommended persistence layout:

```text
runs/
  <run-id>/
    spec.json
    results.json
    frames.jsonl
    report.html
    metadata.json
```

`metadata.json` usually records the model kind, engine version, timestamp,
duration, summary, and any user/project identifiers from the host application.

## Running The Included Demos

The crate ships runnable binaries under `src/bin`. They are useful as examples
and regression checks for the SDK, but they are not the only way to use the
library.

```sh
cargo run --bin main_traffic
cargo run --bin main_temp_control
cargo run --bin main_stochastic_sde_report
cargo run --bin main_empirical_control_report
cargo run --bin main_build_site
cargo run --bin main_studio_workbench
```

Report binaries write HTML artifacts under `out/`.
`main_studio_workbench` writes `out/studio/workbench.html`, a static browser
tool for inspecting, running, sweeping, and exporting `des/studio-graph/v1`
model specs. The full site build also writes `out/studio/spec-workbench.html`
for editing and dragging `des/studio/v1` JSON block diagrams.

## Development

Build and test:

```sh
cargo build --release
cargo test --all-targets --all-features
cargo test --all-targets --all-features -- --ignored
cargo test --doc --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt -- --check
```

Useful SDK imports:

```rust
use des_engine::prelude::*;
use des_engine::des::service::*;
use des_engine::des::plugin::*;
use des_engine::des::streaming::*;
```

## Module And Build Performance Plan

The current package is intentionally organized as one public SDK crate,
`des_engine`, with a deep `des::*` module tree and many demo/validation
binaries under `src/bin`. Rust modules keep the source tree understandable, but
they are not strong compilation boundaries. Cargo's main rebuild unit is the
crate/package, while `rustc` incremental compilation reuses work inside that
crate where it can.

After `git pull`, `cargo build` evaluates the Cargo dependency graph:

- Unchanged dependency crates are reused from the build cache.
- Changed crates are rebuilt.
- Crates that depend on a changed crate are rebuilt as needed.
- Changes inside this single `des_engine` crate can still cause a broad rebuild
  of the crate, even when the edit was made in one source module.

For better compile-time isolation, the long-term direction is to keep the
existing `des_engine::prelude` and `des_engine::sdk` surface stable while
gradually moving internally coherent areas into a Cargo workspace.

Recommended workspace shape:

```text
discrete-event-system.rs/
  Cargo.toml                  # workspace root
  crates/
    des-core/                 # shared types, errors, time, ids, JSON value helpers
    des-model/                # ModelCitizen, RunArtifact, registry, SDK contracts
    des-fel/                  # future-event-list engine and queueing primitives
    des-decision/             # MDP/POMDP specs, solvers, rollout, visualization data
    des-hybrid/               # hybrid block runtime and executive
    des-studio/               # visual block graph, authoring/runtime cells
    des-animation/            # frame types, HTML players, reports
    des-plugin/               # external process/plugin contract
    des-service/              # service discovery descriptors
    des-engine/               # facade crate preserving today's public imports
    des-demos/                # optional demo/validation binaries
```

The facade crate should re-export the stable SDK modules so embedders can keep
using:

```rust
use des_engine::prelude::*;
use des_engine::des::plugin::*;
use des_engine::des::streaming::*;
```

Good crate boundaries for this project:

- Put small, stable, dependency-light primitives in `des-core`.
- Keep JSON-facing SDK contracts in `des-model` so embedders and model families
  share one artifact shape.
- Let each major model family (`fel`, `decision`, `hybrid`, `studio`,
  `acausal`, `equation`) depend inward on contracts instead of sideways on one
  another.
- Keep HTML rendering, reports, and animation output outside the numerical core
  where practical.
- Move demo and validation binaries into a separate package once they no longer
  need private internals.
- Use feature flags for optional heavy surfaces, especially plugins, HTML
  rendering, schemas, service integration, or future GPU/back-end adapters.

Things to avoid:

- Splitting every source directory into a crate. Crate boundaries should
  represent stable APIs and real dependency cuts, not just file organization.
- Letting the facade crate pull every optional dependency by default.
- Exposing large generic or macro-heavy internals across crate boundaries unless
  that is part of the intended public API.
- Using FFI only for compile speed. FFI is valuable for C/Python/JS/external
  plugin boundaries, but it adds ABI, linking, `unsafe`, and testing overhead.
  Rust-to-Rust modularity should use workspace crates first.

Suggested migration order:

1. Measure current build behavior with `cargo build --timings`.
2. Move dependency-light shared primitives into `des-core`.
3. Move the first-class model contract into `des-model`.
4. Move one mostly self-contained family, such as `des-fel` or `des-decision`,
   and keep compatibility re-exports in `des_engine`.
5. Move animation/reporting and demo binaries later, after the core crates have
   settled.
6. Add CI checks with package-scoped commands such as `cargo test -p des-core`
   and `cargo test -p des-engine --all-features`.

Day-to-day build guidance:

```sh
cargo check
cargo test -p des_engine
cargo build --timings
CARGO_INCREMENTAL=1 cargo build
```

Once the workspace split exists, prefer package-scoped commands while
developing a narrow area:

```sh
cargo check -p des-fel
cargo test -p des-decision
cargo run -p des-demos --bin main_traffic
```

If rebuilds are still slow after crate boundaries are in place, consider adding
`sccache` for shared compiler artifact caching and reviewing dependency
features to keep default builds lean.

## Module Map

```text
des_engine
|-- prelude       # curated SDK import: use des_engine::prelude::*
|-- sdk           # crate name/version + SDK module list for diagnostics
`-- des
    |-- acausal  # equation-based models, structural diagnostics, UI metadata
    |-- model     # ModelCitizen, RunArtifact, CitizenRegistry
    |-- authoring # JSON Schema-backed model graph authoring and codegen
    |-- streaming # JSONL solver contracts for LP, MILP, MDP, POMDP
    |-- plugin    # external process plugin contract and HTML player renderer
    |-- service   # HTTP service descriptor and extension seam
    |-- equation  # JSON/LaTeX/XML equation specs as first-class model citizens
    |-- studio    # visual block graph and runtime cell layer
    |-- hybrid    # mixed continuous/discrete/event simulation
    |-- fel       # future-event-list simulation primitives and examples
    |-- decision  # MDP/POMDP specs, solvers, rollouts, visualization
    |-- animation # frame types, HTML players, report-page utilities
    |-- general   # broad model/solver library
    `-- main_*    # runnable simulation demos, not the primary SDK surface
```

In short: use the deep `des::*` tree when you need a specific model family, and
use `des_engine::prelude::*` when you are embedding the SDK in a server or app.
