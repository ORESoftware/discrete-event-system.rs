# Soccer Engine Extraction — Implementation Spec

Status: **proposal for review** (no code moved yet). Work happens on branch
`rl-actor-critic-league-worldmodel`, isolated from the `main` auto-syncer; one
merge at the end.

Goal: pull all soccer domain code out of the `des_engine`
(`discrete-event-system.rs`) crate into a new **agnostic** library crate
`soccer_engine` (`soccer-sim-game-engine.rs`) that depends on `des_engine` for
optimization/learning, and stand up a new axum **`dd-soccer-rs`** root server with
per-UUID games — without changing the existing `dd-des-rs` server's behavior.

---

## 1. Target architecture

```
discrete-event-system.rs  (des_engine)         generic: DES base, NN primitives,
   ▲                                            LP/IP-MIP/Clarabel, MDP/POMDP, evolution,
   │ path dep                                   prng, pg_util, animation framework
soccer-sim-game-engine.rs (soccer_engine)       AGNOSTIC soccer library — NO HTTP server
   ▲            ▲            ▲                    by default: sim/physics/rules/agents,
   │            │            │                    Q-learning + actor-critic + league +
   │            │            │                    world model, planner, rotation
   │            │            └─────────────► desktop game        (default features only)
   │            └──────────────────────────► dd-soccer-rs  (NEW) axum, uuid games, SSE
   └───────────────────────────────────────► dd-des-rs (existing) axum, keeps /soccer/* routes
```

No cycles. `des_engine` never references soccer. Both servers + the desktop game
depend on `soccer_engine`; `soccer_engine` depends on `des_engine`.

### 1a. The agnostic-engine constraint (NEW — desktop game will also import it)

`soccer_engine` must build as a pure library with **no HTTP server and no web
transport by default**. The boundary inside today's `soccer.rs`:

| Concern | Symbols (current `soccer.rs`) | Destination |
|---|---|---|
| **Engine core** (always in lib) | `SoccerMatch`, `SoccerRealtimeSession`, `SoccerStepRequest/Response`, `step_for_live_http`*, `reset_match`, `state_response`, policy mgmt, learning (actor-critic/league/world-model), `SimulationTrace` | `soccer_engine` default |
| **Web bridge** (feature `web-bridge`, default OFF) | `SoccerLiveHttpBridge`, `SoccerLiveHttpReply`, `SoccerLiveServerConfig`, `LiveGameRegistry`, `soccer_live_page_html`/`soccer_simulation_page_html` + the `*.html` assets, `try_write_soccer_playback_artifacts` | `soccer_engine` (feature-gated) |
| **Embedded socket server** (feature `embedded-http-server`, default OFF) | `run_live_soccer_server`, `handle_live_soccer_*`, `LiveHttpResponse`, `parse_live_http_request`, `normalize_live_http_path`, `TcpListener`/chunked writers | `soccer_engine` (feature-gated; for the `main_soccer_live` local-dev bin only) |

\* `step_for_live_http` / `compact_for_live_http` / `to_live_http_frame` are named
"http" but are transport-agnostic typed I/O (no sockets) — they stay in core.
(Optional later: rename to drop the `_http` suffix.)

Why feature-gate rather than physically split the 115k-line `soccer.rs`: the
socket-server region is roughly contiguous (`soccer.rs:52970–53722` + HTML
includes at `57313–57343` + worker helpers `951–962`); moving just those into
`soccer_engine::live::{bridge,server}` submodules is a bounded, low-risk
sub-step. The 100k+ lines of sim/learning never get touched.

- **Desktop game**: `soccer_engine = { default-features = false }` (or default; the
  server features are off by default) → pure sim/learning, zero `TcpListener`.
- **dd-des-rs / dd-soccer-rs**: `soccer_engine = { features = ["web-bridge"] }` →
  they own the axum socket layer and call `SoccerLiveHttpBridge`.
- **`main_soccer_live` bin**: `features = ["embedded-http-server"]` (local dev).

---

## 2. File move list (29 files → `soccer_engine`)

Moved verbatim (git mv), then import-rewritten. Destination layout flattens the
`des::` path since soccer is now the crate root.

| Current (`des_engine`) | Lines | Destination (`soccer_engine`) |
|---|---:|---|
| `src/des/general/soccer.rs` | 115808 | `src/soccer.rs` (split out `live/{bridge,server}.rs` per §1a) |
| `src/des/soccer_learning.rs` | 5730 | `src/learning.rs` |
| `src/des/soccer_learning_pg.rs` | 3924 | `src/learning_pg.rs` (feature `postgres`) |
| `src/des/general/soccer_rotation.rs` | 3453 | `src/rotation.rs` |
| `src/des/soccer_planner/{mod,model,solve,ui}.rs` | 2505 | `src/planner/` |
| `src/des/animation/scenes/soccer_scene.rs` | 792 | `src/scenes/soccer_scene.rs` |
| `src/des/animation/scenes/soccer_ipmip_solver_scene.rs` | 971 | `src/scenes/ipmip_solver_scene.rs` |
| `src/des/streaming/soccer.rs` | 239 | `src/streaming.rs` |
| `src/des/runners/validate_soccer.rs` | 400 | `src/runners/validate.rs` |
| `src/des/test/soccer_test.rs` | 303 | `tests/soccer_test.rs` or `src/tests.rs` |
| `src/des/main_soccer*.rs` (5) | ~1383 | `src/bin/` thin entry points |
| `src/bin/main_soccer*.rs`, `validate_soccer.rs`, `render_*` (13) | ~12k | `src/bin/` |
| `src/des/general/soccer_live_ui.html`, `soccer_ui.html` | — | `assets/` (feature `web-bridge`) |

Total ≈ **150k LOC** relocated. The bins are thin (most 5–11 lines wrapping a
`run()`); the learning/CLI bins (`main_soccer_learning_run/queue/set_play`) are
the large ones and move whole.

---

## 3. Registry inversion (the only real design work)

Today 8 shared files *enumerate* soccer (~37 reference lines). Invert so
`des_engine` exposes extensible shapes and `soccer_engine` *provides* its entries;
a composed binary (or the server) merges them.

| Shared file | Soccer refs | Action |
|---|---:|---|
| `des/mod.rs` | 7 | remove `pub mod soccer_*` + `main_soccer*` decls |
| `des/general/mod.rs` | 2 | remove `pub mod soccer; pub mod soccer_rotation;` |
| `des/simulations.rs` | 6 | drop soccer catalogue entries → `soccer_engine::soccer_sims()` |
| `des/runners/mod.rs` | 1 | drop `validate_soccer` → `soccer_engine::soccer_runners()` |
| `des/streaming/mod.rs` | 8 | drop `StreamingSoccerPlanner` registration → `soccer_engine::soccer_streaming_contracts()` |
| `des/animation/scenes/mod.rs` | 2 | drop soccer scenes → `soccer_engine::soccer_scenes()` |
| `des/html_index.rs` | 10 | drop soccer landing entries → `soccer_engine::soccer_html_index_entries()` |
| `des/test/mod.rs` | 1 | drop `soccer_test` (moves with soccer) |

**API shapes added to `des_engine`** (so domains compose instead of being
enumerated):

```rust
// des_engine::des::registry
pub struct SimCatalogue { /* generic entries */ }
impl SimCatalogue { pub fn extend(&mut self, entries: impl IntoIterator<Item = SimEntry>); }
pub struct SceneRegistry  { /* ... extend(...) ... */ }
pub struct StreamingContracts { /* ... register(Box<dyn StreamingModel>) ... */ }
pub struct RunnerRegistry { /* ... */ }
pub struct HtmlIndex { /* ... */ }
```

`soccer_engine` provides:

```rust
// soccer_engine
pub fn soccer_sims() -> Vec<SimEntry>;
pub fn soccer_scenes() -> Vec<SceneEntry>;
pub fn soccer_streaming_contracts() -> Vec<Box<dyn StreamingModel>>;
pub fn soccer_runners() -> Vec<RunnerEntry>;
pub fn soccer_html_index_entries() -> Vec<HtmlIndexEntry>;
```

Composition point (the server, or a `des-with-soccer` demo bin):

```rust
let mut cat = des_engine::default_sim_catalogue();
cat.extend(soccer_engine::soccer_sims());
```

This is the work that requires judgment; everything else is mechanical.

### Non-seams (confirmed, no action)
- `fel/learning_pg.rs` → its only soccer mention is a **comment**; it has its own
  `fel_elevator_pg_sslmode` TLS logic. **No coupling.** (Optional later DRY:
  hoist the shared TLS/sslmode/retry into `des_engine::des::pg_util` so soccer's
  and elevator's pg adapters share it — nice-to-have, not required for the cut.)
- `external_validation_tools.rs` → its soccer mentions are **example string
  literals** (a yaml fixture, a `soccer_matches` SQL example, an OpenAPI title).
  **No coupling.** Leave as-is.

---

## 4. Import rewrites

In the 29 moved files: `crate::des::general::<generic>` → `des_engine::des::general::<generic>`
(lp, neural_network, des_base, ode, prng, hungarian, ip_mip_des, evolution, …),
`crate::des::shared::*` → `des_engine::des::shared::*`, `crate::des::animation::types`
→ `des_engine::des::animation::types`. Intra-soccer paths (`crate::des::soccer_*`,
`crate::des::general::soccer*`) become crate-local `crate::*`. All ~45 generic
items are already `pub` (verified) — **no visibility widening needed**. Scripted
`sed` + compile-driven fixups.

---

## 5. The new `dd-soccer-rs` server

New crate `k8s-cluster/remote/deployments/soccer-rs` (axum, like `dd-des-rs`),
`soccer_engine = { path = "../../submodules/soccer-sim-game-engine.rs", features = ["web-bridge"] }`,
`des_engine = { path = "../../submodules/discrete-event-system.rs" }`.

**Per-UUID game registry** (the new capability):
```rust
struct GameSession { match_: SoccerMatch, frames: RingBuffer<MatchFrame>,
                     tx: broadcast::Sender<MatchFrame>, last_seen: Instant }
struct Games(DashMap<Uuid, Arc<Mutex<GameSession>>>);   // TTL-evicted
```
Each game runs a background tick-loop task; frames fan out over a broadcast
channel → SSE/WS. (Upgrade over today's poll-per-`/api/step`.) Reuses the engine's
`SoccerRealtimeSession`/`step_for_live_http` per game, keyed by uuid instead of one
global session.

**Route table** (root, uuid in `?id=`):

| Method | Path | Behavior |
|---|---|---|
| POST | `/soccer/game` | mint uuid, start a `GameSession`, return `{id}` |
| GET | `/soccer/game?id=<uuid>` | game state/metadata JSON |
| GET | `/soccer/games` | list active game ids |
| GET | `/soccer/live?id=<uuid>` | live 2D UI bound to the game; frames via SSE `/soccer/live/stream?id=` |
| GET | `/soccer/sim?id=<uuid>` | playback/replay page + per-game artifacts |
| POST | `/api/step?id=<uuid>` | advance (fallback to poll model) |
| POST | `/api/{reset,assign,input,learning}?id=<uuid>` | session controls scoped to the game |
| GET | `/healthz` `/readyz` | probes |

HTML/UI assets (`soccer_live_ui.html`, `soccer_ui.html`) ship in the **server
crate** (or via the engine `web-bridge` feature) — not the agnostic default lib.

---

## 6. `dd-des-rs` change (back-compat, zero behavior change)

`dd-des-rs` currently imports soccer from `des_engine`
(`SoccerLiveHttpBridge`, `SoccerLiveServerConfig`, `try_write_soccer_playback_artifacts`,
`soccer_planner`). After the cut those symbols live in `soccer_engine`, so:
- add `soccer_engine = { path = "../../submodules/soccer-sim-game-engine.rs", features = ["web-bridge"] }`
- rewrite the ~2 `des_engine::des::general::soccer::` / `des_engine::des::soccer_planner::`
  imports → `soccer_engine::…`
- keep `des_engine` for the generic bits it still uses.
All existing routes (`/soccer/live`, `/soccer/planner`, `/out/soccer-sim.html`,
`/api/*`) keep working unchanged.

---

## 7. Deploy (mirror `dd-des-rs`; in-pod clone-adjacent)

New `remote/argocd/dd-next-runtime/dd-soccer-rs.{deployment,service,hpa,networkpolicy}.yaml`,
mirroring `dd-des-rs`:
- `rust:1.90-bookworm`; startup script clones **both** submodules adjacent
  (`discrete-event-system.rs@main` + `soccer-sim-game-engine.rs@main`) so the
  `path` deps resolve, then `cargo run --release`. (`dd-des-rs` already builds the
  equivalent total source in-pod successfully, so this size is proven; prebuilt
  multi-stage Docker image stays a fallback if pod-build latency bites.)
- Resource limits + memory cap (avoid node OOM); writable `/tmp` for `CARGO_TARGET_DIR`.
- Gateway nginx (`dd-remote-gateway.configmap.yaml`): add `location /soccer/ { proxy_pass http://dd-soccer-rs.default.svc.cluster.local:PORT/; }`
  (+ SSE headers / long read timeout on `/soccer/live`). **`/des-rs/*` unchanged.**

---

## 8. Phase plan (each phase ends green)

- **P0 — De-risk.** Done: work isolated on a branch; RL work committed
  (`57d7fa9`). One merge to `main` at the very end.
- **P1 — Workspace.** Convert `discrete-event-system.rs` into a temporary Cargo
  workspace with members `des_engine` + a new `soccer_engine` (path dep). Keeps
  the refactor atomic and green in one `cargo build`.
- **P2 — Registry inversion + agnostic boundary.** Add the `des_engine` registry
  shapes; carve the socket-server / web-bridge code in `soccer.rs` into
  feature-gated `live::{server,bridge}` modules. des_engine still builds with soccer
  present (moved to the workspace member), green.
- **P3 — Move + rewrite.** Relocate the 29 files into `soccer_engine`; rewrite
  imports; soccer bins → `soccer_engine/src/bin`. Full soccer test suite green;
  `des_engine` builds with zero soccer.
- **P4 — Repo split.** Promote `soccer_engine` to `github.com/ORESoftware/soccer-sim-game-engine.rs`;
  `des_engine = { path = "../discrete-event-system.rs" }`; add to the submodule +
  auto-syncer sets.
- **P5 — Servers.** Point `dd-des-rs` at `soccer_engine` (§6); build/deploy the new
  `dd-soccer-rs` (§5, §7).

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| Auto-syncer churn on `main` (#1) | All work on the branch; single merge at end |
| 115k-line `soccer.rs` + many rewrites | Scripted path rewrite; compile-driven; `soccer.rs` moves whole (only the server region is carved) |
| Agnostic-engine: missing a `cfg` gate | `cargo build` the engine with default features in CI → must have no `TcpListener`/HTTP symbols |
| Registry inversion misses an entry | P2 before the move; full test suite + `/out/*` smoke per phase |
| Cross-repo path dep at deploy | Reuse `dd-des-rs`'s proven clone-adjacent mechanism |
| Build-from-source clones 2 repos | Extend `dd-des-rs` clone script; cache `CARGO_TARGET_DIR` |

---

## 10. Open items for the desktop-game consumer (out of scope, noted)

- The engine must expose a clean, transport-free **drive API**: construct a
  `SoccerMatch`/`SoccerRealtimeSession`, `run_time_step()`, read frames — already
  present (`SoccerStepRequest/Response`, `state_response`). The desktop game renders
  frames itself; it never touches `web-bridge`/`embedded-http-server`.
- Confirm the engine's default feature set pulls in `postgres`/`clarabel` only
  when needed (gate `learning_pg` behind `feature = "postgres"`; Clarabel comes
  via `des_engine` already).
