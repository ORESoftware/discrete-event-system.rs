# des_engine

`des_engine` is a Rust library/SDK for modeling, simulating, solving, and
rendering discrete, continuous, and mixed systems. It is the Rust port of the
ORESoftware TypeScript discrete-event-system engine, with the SDK surface shaped
for embedders: web servers, desktop apps, CLIs, workers, notebooks, and plugin
hosts.

The crate is not only a collection of demo binaries. The important part is the
library surface:

- Run models from JSON specs through one uniform API.
- Solve MDPs, POMDPs, LPs, MILPs, hybrid block diagrams, future-event-list
  simulations, and time-stepped simulations.
- Return machine-readable results as JSON.
- Return frame streams as JSONL for playback, storage, or live streaming.
- Render self-contained HTML players and report pages that can be served by any
  HTTP server.
- Advertise the service surface through a machine-readable discovery descriptor.
- Run external simulation plugins that emit JSON or JSONL and render them with
  the same built-in player.

## What This SDK Is For

Use `des_engine` when you want a Rust service to accept a model/specification,
run a simulation or solver, and serve the outcome back to another system.

Typical use cases:

- A backend API where clients submit simulation specs and receive `results.json`,
  `frames.jsonl`, and `report.html`.
- A dashboard that lets users compare traffic, elevator, epidemic, queueing,
  control, optimization, or decision-process runs.
- A worker that runs a catalogue of simulations on a schedule and stores the
  artifacts.
- An AI-assisted modeling service where an LLM emits a JSON spec, the engine
  validates/runs it, and the UI renders the returned artifact.
- A plugin host where third-party binaries produce JSON/JSONL and the SDK turns
  that output into a standard result or player.

The engine is JSON-first on purpose. JSON is the boundary for HTTP, IPC,
plugins, and persistence. The core Rust APIs stay strongly typed internally,
but embedders can treat runs as portable artifacts.

## Main Concepts

### First-class model citizens

The `des::model` module defines the paradigm-neutral contract:

- `ModelDescriptor`: discovery metadata for a model kind.
- `ModelCitizen`: validates and runs a JSON spec.
- `CitizenRegistry`: registry of runnable model kinds.
- `RunArtifact`: the uniform output of a run.

`RunArtifact` is the key server-facing type. It contains:

- `results`: a JSON document for programmatic consumers.
- `frames`: JSON values that can be serialized as JSONL.
- `to_jsonl()`: frame stream text.
- `to_player_html()`: a self-contained HTML player/report for browsers.

The built-in registry currently includes MDP, POMDP, hybrid, and studio model
citizens:

```rust
use des_engine::prelude::*;

let registry = with_builtins();

for descriptor in registry.descriptors() {
    println!("{}: {}", descriptor.kind, descriptor.description);
}
```

### Streaming solvers

The `des::streaming` module exposes long-lived JSONL contracts for iterative
solvers:

- `lp`
- `milp` / `mip` / `ip`
- `mdp`
- `pomdp`

These models consume one JSON command per line and emit one JSON frame per line.
Malformed commands are converted into error frames where possible, so a stream
can continue after bad input.

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
    let docs_json = descriptor.to_json_string();
    let link_header = descriptor.link_header_relative();
    println!("{docs_json}");
    println!("{link_header}");
    Ok(())
}
```

### External plugins

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
2. Route the request to a model citizen, streaming solver, built-in simulation,
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

An HTTP server can then expose routes like:

- `POST /runs/{kind}`: call `run_model(kind, body_json)`, assign a run id, store
  the artifact, and return `{ "id": "...", "summary": "..." }`.
- `GET /runs/{id}/results.json`: return `StoredRun.results` as
  `application/json`.
- `GET /runs/{id}/frames.jsonl`: return `StoredRun.frames_jsonl` as
  `application/x-ndjson`.
- `GET /runs/{id}/report.html`: return `StoredRun.report_html` as `text/html`.
- `GET /api/docs.json`: return `ServiceDescriptor::to_json_string()`.

With Axum or another async Rust framework, the handlers stay small because the
SDK already owns validation, solving, and rendering:

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

For streaming solvers, the route shape is similar, but the response body is
JSONL:

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

For external simulation programs, store the manifest in your server database or
configuration, run it on demand, and serve the returned HTML:

```rust
use des_engine::des::plugin::{run_and_render, PluginError, PluginManifest};

pub fn run_plugin_report(manifest: &PluginManifest) -> Result<String, PluginError> {
    run_and_render(manifest)
}
```

## Reports And Artifacts

The SDK has two related output styles:

- `RunArtifact` player output: lightweight, uniform output for model citizens
  and plugin runs. Use `to_player_html()`, `to_jsonl()`, and `results`.
- Report pages: richer narrative pages used by demo/report binaries. These are
  built with `des::animation::run_report` and written by binaries such as
  `main_stochastic_sde_report` and `main_empirical_control_report`.

For a server, prefer `RunArtifact` as the stable API return type. Use report
pages when you want a curated, human-readable page for a specific simulation
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

The crate ships many runnable binaries under `src/bin`. They are useful as
examples and regression checks for the SDK, but they are not the only way to use
the library.

```sh
cargo run --bin main_traffic
cargo run --bin main_temp_control
cargo run --bin main_stochastic_sde_report
cargo run --bin main_empirical_control_report
```

Report binaries write HTML artifacts under `out/`.

## Development

Build and test:

```sh
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

## Module Map

- `des::model`: first-class model contract and `RunArtifact`.
- `des::streaming`: JSONL solver contracts for LP, MILP, MDP, and POMDP.
- `des::plugin`: external process plugin contract and HTML player renderer.
- `des::service`: HTTP service discovery descriptor and extension seam.
- `des::studio`: visual block graph and runtime cell layer.
- `des::hybrid`: mixed continuous/discrete/event simulation.
- `des::fel`: future-event-list simulation primitives and examples.
- `des::decision`: MDP/POMDP specs, solvers, rollouts, and visualization.
- `des::animation`: frame types, HTML players, and report-page utilities.

In short: use the deep `des::*` tree when you need a specific model family, and
use `des_engine::prelude::*` when you are embedding the SDK in a server or app.
