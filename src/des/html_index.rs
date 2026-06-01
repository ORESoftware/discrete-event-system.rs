//! Central registry of HTML artifacts for the landing index and site builder.
//!
//! Every simulation that produces a viewable HTML page registers here with its
//! title, description, and optional generator. [`main_build_site`] reads
//! [`html_index_groups`] for featured cards and calls [`generate_html_artifacts`]
//! during a full rebuild.

#![allow(dead_code)]

use crate::des::animation::run_report::{IndexEntry, IndexGroup};
use serde_json::{json, Value};

/// Metadata for one HTML page on the landing index.
#[derive(Clone, Copy, Debug)]
pub struct HtmlIndexSpec {
    pub kind: &'static str,
    pub title: &'static str,
    pub href: &'static str,
    pub description: &'static str,
}

/// A featured section on the landing index.
#[derive(Clone, Copy, Debug)]
pub struct HtmlIndexGroupSpec {
    pub heading: &'static str,
    pub blurb: &'static str,
    pub entries: &'static [HtmlIndexSpec],
}

const MODELING_TOOLS: &[HtmlIndexSpec] = &[
    HtmlIndexSpec {
        kind: "tool",
        title: "Modeling Studio",
        href: "studio/modeling-studio.html",
        description:
            "Block-diagram editor with palette metadata, wiring, inspector fields, JSON save/load, and a local scalar run plot.",
    },
    HtmlIndexSpec {
        kind: "workbench",
        title: "Modeling Workbench",
        href: "studio/workbench.html",
        description:
            "OpenMDAO-style N2 dependency view, objective/constraint metadata, local run controls, and a parameter-sweep driver.",
    },
    HtmlIndexSpec {
        kind: "workbench",
        title: "Studio JSON Workbench",
        href: "studio/spec-workbench.html",
        description:
            "Author, inspect, run, drag, and export JSON block diagrams with nested runtime cells.",
    },
];

const CONTROL_ANIMATIONS: &[HtmlIndexSpec] = &[
    HtmlIndexSpec {
        kind: "animation",
        title: "Wind MPPT — optimal torque",
        href: "wind-mppt/animation-optimal-torque.html",
        description:
            "Variable-speed PMSG turbine tracking optimal tip-speed ratio via T = K_opt·ω².",
    },
    HtmlIndexSpec {
        kind: "animation",
        title: "Wind MPPT — PI speed loop",
        href: "wind-mppt/animation-pi.html",
        description: "Same turbine driven by a PI controller tracking ω* = λ*·V/R.",
    },
    HtmlIndexSpec {
        kind: "animation",
        title: "DC motor — closed-loop PI",
        href: "dc-motor/animation-closed.html",
        description: "Back-EMF ODE motor; PI speed control tracking 60→100 rad/s with a load step.",
    },
    HtmlIndexSpec {
        kind: "animation",
        title: "DC motor — open loop",
        href: "dc-motor/animation-open.html",
        description: "Step-voltage response showing back-EMF rise throttling armature current.",
    },
    HtmlIndexSpec {
        kind: "animation",
        title: "Controllability & Observability",
        href: "obs-ctrl/animation.html",
        description:
            "Kalman rank tests, MDP reachability, and POMDP distinguishability storyboard.",
    },
    HtmlIndexSpec {
        kind: "animation",
        title: "Empirical control — structured run",
        href: "empirical-control/player.html",
        description:
            "Playable frame stream for LTI Gramian, MDP reachability, and POMDP belief checks.",
    },
    HtmlIndexSpec {
        kind: "animation",
        title: "Temperature control — winter heat",
        href: "temp-control/animation.html",
        description: "Heating-only indoor temperature control over a cold 24-hour winter day.",
    },
    HtmlIndexSpec {
        kind: "animation",
        title: "Temperature control — heat/cool",
        href: "temp-control/animation-heat-cool.html",
        description: "Bidirectional heat-pump control with night heating and afternoon cooling.",
    },
];

const STUDIO_TOOLS: &[HtmlIndexSpec] = &[HtmlIndexSpec {
    kind: "workbench",
    title: "DES Studio Workbench",
    href: "studio/workbench.html",
    description:
        "Author, inspect, run, drag, and export JSON block diagrams with the visual-block studio.",
}];

const NUMERICAL_SOLVERS: &[HtmlIndexSpec] = &[
    HtmlIndexSpec {
        kind: "animation",
        title: "L-BFGS — gradient descent",
        href: "numerical-solvers/lbfgs.html",
        description: "Limited-memory BFGS minimizing a smooth objective, one curvature-corrected step per DES tick.",
    },
    HtmlIndexSpec {
        kind: "animation",
        title: "Needleman–Wunsch alignment",
        href: "numerical-solvers/sequence-alignment.html",
        description: "Dynamic-programming global sequence alignment filling one score-table row per tick.",
    },
    HtmlIndexSpec {
        kind: "animation",
        title: "Metropolis–Hastings MCMC",
        href: "numerical-solvers/metropolis-hastings.html",
        description: "Random-walk Markov-chain Monte Carlo sampling a target density, accept/reject per tick.",
    },
    HtmlIndexSpec {
        kind: "animation",
        title: "Differential Evolution",
        href: "numerical-solvers/differential-evolution.html",
        description: "Population-based DE/rand/1/bin evolutionary search; best fitness improves per generation.",
    },
    HtmlIndexSpec {
        kind: "animation",
        title: "Prim's minimum spanning tree",
        href: "numerical-solvers/prim-mst.html",
        description: "Greedy graph optimization adding the cheapest frontier edge to the tree each tick.",
    },
    HtmlIndexSpec {
        kind: "animation",
        title: "Backpropagation MLP",
        href: "numerical-solvers/backprop-mlp.html",
        description: "A single-hidden-layer neural net trained by full-batch backprop, one gradient epoch per tick.",
    },
    HtmlIndexSpec {
        kind: "animation",
        title: "EM — Gaussian mixture",
        href: "numerical-solvers/gaussian-mixture-em.html",
        description: "Expectation–Maximization for a 1-D Gaussian mixture; log-likelihood rises each iteration.",
    },
    HtmlIndexSpec {
        kind: "animation",
        title: "Mean-field variational inference",
        href: "numerical-solvers/mean-field-vi.html",
        description: "Coordinate-ascent VI (CAVI) fitting a Normal–Gamma posterior, refining E[τ] per tick.",
    },
];

const RUN_REPORTS: &[HtmlIndexSpec] = &[
    HtmlIndexSpec {
        kind: "simulation",
        title: "Traffic flow — five intersection",
        href: "traffic-flow-five-intersection.html",
        description: "Signalized five-intersection road network with moving car snapshots and lane-phase highlights.",
    },
    HtmlIndexSpec {
        kind: "simulation",
        title: "Smart traffic flow",
        href: "smart-traffic-flow.html",
        description: "Smart movable cars with shuffled actor updates, accident instrumentation, and live traffic metrics.",
    },
    HtmlIndexSpec {
        kind: "run report",
        title: "DC motor — shadow controllability & observability",
        href: "dc-motor/shadow-observability-controllability.html",
        description: "Dual evaluator for the back-EMF plant: Kalman rank tests plus Gramian degree metrics for weak/strong directions.",
    },
    HtmlIndexSpec {
        kind: "run report",
        title: "Empirical controllability & observability",
        href: "empirical-control/report.html",
        description: "Gramian degree (min/max directions) and Monte-Carlo trial estimates vs analytic Kalman tests.",
    },
    HtmlIndexSpec {
        kind: "run report",
        title: "Stochastic SDEs + 3 ML algorithms",
        href: "stochastic-sde/report.html",
        description: "Euler–Maruyama engine with MLE system-id, Ensemble Kalman filtering, and a diffusion model.",
    },
];

const EXTENDED_SIMULATIONS: &[HtmlIndexSpec] = &[
    HtmlIndexSpec {
        kind: "simulation",
        title: "Elevator high-rise dispatch",
        href: "elevator-highrise.html",
        description: "Multi-car elevator bank over a full high-rise day with passenger flows and car trajectories.",
    },
    HtmlIndexSpec {
        kind: "simulation",
        title: "FactMachine markets",
        href: "factmachine-markets.html",
        description: "Prediction-market dynamics with order flow, liquidity, and participant behavior.",
    },
    HtmlIndexSpec {
        kind: "simulation",
        title: "Two-disease epidemic",
        href: "two-disease.html",
        description: "Competing epidemic strains with contact dynamics and compartment transitions.",
    },
    HtmlIndexSpec {
        kind: "run report",
        title: "Shadow evaluation report",
        href: "shadow-eval/report.html",
        description: "Side-by-side shadow-mode evaluation of controllability and observability metrics.",
    },
    HtmlIndexSpec {
        kind: "animation",
        title: "Soccer IP-MIP feasible solver",
        href: "soccer-IP-MIP-feasible-solver.html",
        description: "Mixed-integer solver animation for the soccer rotation feasibility model.",
    },
    HtmlIndexSpec {
        kind: "animation",
        title: "Soccer IP-MIP feasible run",
        href: "soccer-IP-MIP-feasible.html",
        description: "Full soccer rotation feasibility simulation with player movement snapshots.",
    },
];

/// All featured index groups, in display order.
pub fn html_index_groups() -> &'static [HtmlIndexGroupSpec] {
    static GROUPS: &[HtmlIndexGroupSpec] = &[
        HtmlIndexGroupSpec {
            heading: "Modeling tools",
            blurb: "Browser-based block diagram authoring and analysis for the emerging PyDy/OpenMDAO-style studio.",
            entries: MODELING_TOOLS,
        },
        HtmlIndexGroupSpec {
            heading: "Control-system animations",
            blurb: "Interactive HTML players (play / pause / scrub / speed) built on the DES animation engine.",
            entries: CONTROL_ANIMATIONS,
        },
        HtmlIndexGroupSpec {
            heading: "Optimization, learning & inference solvers",
            blurb: "Classic numerical algorithms expressed as source → solver → sink DES pipelines, animated one iteration per tick: gradient descent, dynamic programming, MCMC, evolutionary search, graph optimization, backprop, and probabilistic inference.",
            entries: NUMERICAL_SOLVERS,
        },
        HtmlIndexGroupSpec {
            heading: "Numerical & machine-learning runs",
            blurb: "Reproducible run reports with the full console output of each simulation.",
            entries: RUN_REPORTS,
        },
        HtmlIndexGroupSpec {
            heading: "Network, epidemic & domain simulations",
            blurb: "Large-scale DES models — elevators, markets, epidemics, sports optimization, and shadow evaluation — rendered as interactive HTML.",
            entries: EXTENDED_SIMULATIONS,
        },
    ];
    GROUPS
}

/// Convert registry specs into landing-page entries.
pub fn to_index_entries(specs: &[HtmlIndexSpec]) -> Vec<IndexEntry> {
    specs
        .iter()
        .map(|s| IndexEntry {
            kind: s.kind.to_string(),
            title: s.title.to_string(),
            href: s.href.to_string(),
            description: s.description.to_string(),
        })
        .collect()
}

/// Build [`IndexGroup`] values from the registry (caller filters by file existence).
pub fn index_groups_from_registry() -> Vec<IndexGroup> {
    html_index_groups()
        .iter()
        .map(|g| IndexGroup {
            heading: g.heading.to_string(),
            blurb: g.blurb.to_string(),
            entries: to_index_entries(g.entries),
        })
        .collect()
}

/// Regenerate every Rust-native HTML artifact the site builder knows about.
pub fn generate_html_artifacts() {
    eprintln!("Generating Modeling Studio...");
    match crate::des::studio::write_studio_editor_html("out") {
        Ok(path) => eprintln!("  • {}", path.display()),
        Err(e) => eprintln!("  ! Modeling Studio generation failed: {e}"),
    }
    if let Err(e) = crate::des::studio::write_workbench_html(
        "out/studio/workbench.html",
        &crate::des::studio::starter_model_spec(),
    ) {
        eprintln!("  ! Modeling Workbench generation failed: {e}");
    } else {
        eprintln!("  • out/studio/workbench.html");
    }
    match crate::des::studio::write_workbench("out/studio/spec-workbench.html") {
        Ok(path) => eprintln!("  • {}", path.display()),
        Err(e) => eprintln!("  ! Studio JSON Workbench generation failed: {e}"),
    }

    eprintln!("Generating decision, hybrid, and plugin players...");
    match generate_decision_pages() {
        Ok(()) => eprintln!("  • out/decision/*.html"),
        Err(e) => eprintln!("  ! decision player generation failed: {e}"),
    }
    match generate_hybrid_pages() {
        Ok(()) => eprintln!("  • out/hybrid/*.html"),
        Err(e) => eprintln!("  ! hybrid player generation failed: {e}"),
    }
    match generate_plugin_pages() {
        Ok(()) => eprintln!("  • out/plugin/*.html"),
        Err(e) => eprintln!("  ! plugin player generation failed: {e}"),
    }

    eprintln!("Regenerating control-system animations...");
    generate_wind_mppt_pages();
    generate_dc_motor_pages();
    crate::des::main_observability_controllability_anim::run();

    eprintln!("Regenerating temperature-control animations...");
    crate::des::main_temp_control_anim::run_preset(
        crate::des::main_temp_control_anim::Scenario::Winter,
        "out/temp-control/animation.html",
    );
    crate::des::main_temp_control_anim::run_preset(
        crate::des::main_temp_control_anim::Scenario::HeatCool,
        "out/temp-control/animation-heat-cool.html",
    );

    eprintln!("Regenerating numerical / ML solver animations...");
    crate::des::main_numerical_solver_anim::run();

    eprintln!("Generating run reports...");
    crate::des::main_empirical_control_report::run();
    crate::des::main_stochastic_sde_report::run();

    eprintln!("Generating traffic simulations...");
    match crate::des::main_traffic::write_traffic_html_pages() {
        Ok((traffic, smart)) => {
            eprintln!("  • {traffic}");
            eprintln!("  • {smart}");
        }
        Err(e) => eprintln!("  ! traffic HTML generation failed: {e}"),
    }

    eprintln!("Generating extended simulation players...");
    crate::des::main_elevator_highrise::run();
    crate::des::main_factmachine_markets::run();
    generate_two_disease_page();
    crate::des::main_shadow_eval::run();
    crate::des::main_soccer_rotation::run_anim();
}

fn with_opts(mut spec: Value, opts: Value) -> Value {
    if let (Value::Object(map), Value::Object(extra)) = (&mut spec, opts) {
        for (key, value) in extra {
            map.insert(key, value);
        }
    }
    spec
}

fn generate_decision_pages() -> Result<(), Box<dyn std::error::Error>> {
    use crate::des::decision::{machine_maintenance_mdp, tiger_pomdp};
    use crate::des::exec::{
        requirements_for_studio, select, ExecCapabilities, Executive, HybridExecutive,
        StudioExecutive,
    };
    use crate::des::hybrid::demos as hybrid_demos;
    use crate::des::model::with_builtins;
    use crate::des::studio::queue_line;

    std::fs::create_dir_all("out/decision")?;
    let registry = with_builtins();

    let mdp_spec = with_opts(
        serde_json::to_value(machine_maintenance_mdp())?,
        json!({ "start": 0, "steps": 24, "seed": 7 }),
    );
    let mdp = registry.run("mdp", &mdp_spec)?;
    std::fs::write("out/decision/mdp.html", mdp.to_player_html())?;
    std::fs::write("out/decision/mdp.frames.jsonl", mdp.to_jsonl())?;

    let pomdp_spec = with_opts(
        serde_json::to_value(tiger_pomdp())?,
        json!({ "method": "lookahead", "horizon": 3, "steps": 18, "seed": 5 }),
    );
    let pomdp = registry.run("pomdp", &pomdp_spec)?;
    std::fs::write("out/decision/pomdp.html", pomdp.to_player_html())?;
    std::fs::write("out/decision/pomdp.frames.jsonl", pomdp.to_jsonl())?;

    let hybrid = registry.run("hybrid", &json!({ "demo": "bouncing-ball" }))?;
    std::fs::write("out/decision/hybrid.html", hybrid.to_player_html())?;

    for demo in ["signal-chain", "mixer", "queue-line"] {
        let studio = registry.run("studio", &json!({ "demo": demo }))?;
        std::fs::write(
            format!("out/decision/studio-{demo}.html"),
            studio.to_player_html(),
        )?;
    }

    let demo = queue_line()?;
    let req = requirements_for_studio(&demo.compiled);
    let _chosen = select(req).expect("an executive for a dataflow graph");
    let mut studio_exec = StudioExecutive::from_demo(demo);
    let routed = studio_exec.run();
    std::fs::write("out/decision/exec-routed.html", routed.to_player_html())?;

    let (compiled, opts) = hybrid_demos::closed_loop()?;
    let mut hybrid_exec = HybridExecutive::new(
        compiled,
        opts,
        "closed-loop",
        "Hybrid Block Diagram",
        "Closed-loop control.",
    );
    let _ = select(ExecCapabilities {
        continuous: true,
        events: true,
        ..Default::default()
    });
    let _ = hybrid_exec.run();

    let descriptors: Vec<Value> = registry
        .descriptors()
        .iter()
        .map(|d| serde_json::to_value(d).unwrap())
        .collect();
    std::fs::write(
        "out/decision/citizens.json",
        serde_json::to_string_pretty(&json!({ "citizens": descriptors }))?,
    )?;
    Ok(())
}

fn generate_hybrid_pages() -> Result<(), Box<dyn std::error::Error>> {
    use crate::des::hybrid::{demos, executive::simulate, Trace};
    use crate::des::plugin::{
        render_player_html, OutputKind, PlayerKind, PluginManifest, PluginOutput, PluginRun,
        PluginRuntimeKind, PluginTransportKind, RunSpec, UiControl,
    };

    fn manifest(id: &str, name: &str, desc: &str, controls: Vec<UiControl>) -> PluginManifest {
        PluginManifest {
            id: id.to_string(),
            name: name.to_string(),
            version: "1.0.0".to_string(),
            description: desc.to_string(),
            runtime: PluginRuntimeKind::Rust,
            transport: PluginTransportKind::Stdio,
            language: None,
            run: RunSpec::new("hybrid-internal", &[]),
            output: OutputKind::Jsonl,
            player: PlayerKind::Sim,
            controls,
            title: Some(name.to_string()),
        }
    }

    fn render(manifest: &PluginManifest, frames: Vec<Value>) -> String {
        let run = PluginRun {
            plugin_id: manifest.id.clone(),
            output: PluginOutput::Jsonl(frames),
            exit_code: Some(0),
            stderr: String::new(),
        };
        render_player_html(manifest, &run)
    }

    fn ball_frames(trace: &Trace) -> Vec<Value> {
        let (ts, hs) = trace.series("ball.p0[0]").expect("height channel");
        let (_, vs) = trace.series("ball.p0[1]").expect("velocity channel");
        ts.iter()
            .enumerate()
            .map(|(k, &t)| {
                let h = hs[k];
                let cy = 222.0 - h.max(0.0) * 180.0;
                json!({
                    "t": t,
                    "height": h,
                    "velocity": vs[k],
                    "shapes": [
                        { "kind": "line", "x1": 20.0, "y1": 224.0, "x2": 180.0, "y2": 224.0,
                          "stroke": "#475569", "strokeWidth": 2.0 },
                        { "kind": "circle", "x": 100.0, "y": cy, "r": 12.0, "fill": "#2563eb" },
                        { "kind": "text", "x": 100.0, "y": 18.0, "text": format!("h = {h:.3}"),
                          "anchor": "middle", "fontSize": 12.0, "fill": "#0f172a" }
                    ]
                })
            })
            .collect()
    }

    std::fs::create_dir_all("out/hybrid")?;

    let (compiled, opts) = demos::closed_loop()?;
    let trace = simulate(&compiled, &opts);
    let m = manifest(
        "hybrid-closed-loop",
        "Hybrid: multirate closed loop",
        "Continuous first-order plant regulated to a setpoint by a discrete-time PI controller.",
        vec![
            UiControl::range("speed", "Speed (fps)", 1.0, 60.0, 1.0, 20.0),
            UiControl::select(
                "metric",
                "Feature signal",
                &["all", "plant.p0", "pi.p0", "error.p0", "reference.p0"],
                "all",
                Some("metric"),
            ),
        ],
    );
    std::fs::write(
        "out/hybrid/closed-loop.html",
        render(&m, trace.to_jsonl_frames()),
    )?;

    let (compiled, opts) = demos::bouncing_ball()?;
    let trace = simulate(&compiled, &opts);
    let m = manifest(
        "hybrid-bouncing-ball",
        "Hybrid: bouncing ball",
        "A continuous plant with a floor zero-crossing and an energy-losing reflection event.",
        vec![UiControl::range(
            "speed",
            "Speed (fps)",
            1.0,
            60.0,
            1.0,
            30.0,
        )],
    );
    std::fs::write(
        "out/hybrid/bouncing-ball.html",
        render(&m, ball_frames(&trace)),
    )?;
    Ok(())
}

fn generate_plugin_pages() -> Result<(), Box<dyn std::error::Error>> {
    use crate::des::plugin::{
        render_player_html, OutputKind, PlayerKind, PluginManifest, PluginOutput, PluginRun,
        PluginRuntimeKind, PluginTransportKind, RunSpec, UiControl,
    };

    fn run_with(plugin_id: &str, output: PluginOutput) -> PluginRun {
        PluginRun {
            plugin_id: plugin_id.to_string(),
            output,
            exit_code: Some(0),
            stderr: String::new(),
        }
    }

    std::fs::create_dir_all("out/plugin")?;

    let queue = PluginManifest {
        id: "queue".to_string(),
        name: "M/M/1 Queue (embedded Rust plugin demo)".to_string(),
        version: "1.0.0".to_string(),
        description:
            "A deterministic M/M/1-style queue emits JSONL frames; the core renders a sim player."
                .to_string(),
        runtime: PluginRuntimeKind::Rust,
        transport: PluginTransportKind::Stdio,
        language: None,
        run: RunSpec::new("embedded-plugin-queue", &[]),
        output: OutputKind::Jsonl,
        player: PlayerKind::Sim,
        controls: vec![
            UiControl::range("speed", "Speed (fps)", 1.0, 30.0, 1.0, 8.0),
            UiControl::toggle("show_n", "Show n(t)", true, Some("n")),
            UiControl::toggle("show_busy", "Show server busy", true, Some("serverBusy")),
        ],
        title: None,
    };
    let queue_frames = queue_demo_frames();
    std::fs::write(
        "out/plugin/queue.html",
        render_player_html(
            &queue,
            &run_with("queue", PluginOutput::Jsonl(queue_frames)),
        ),
    )?;

    let lp = PluginManifest {
        id: "lp".to_string(),
        name: "LP Solver (embedded Rust plugin demo)".to_string(),
        version: "1.0.0".to_string(),
        description: "A single JSON LP result rendered with the plugin results player.".to_string(),
        runtime: PluginRuntimeKind::Rust,
        transport: PluginTransportKind::Stdio,
        language: None,
        run: RunSpec::new("embedded-plugin-lp", &[]),
        output: OutputKind::Json,
        player: PlayerKind::Results,
        controls: vec![
            UiControl::toggle("show_vars", "Variables", true, Some("variables")),
            UiControl::toggle("show_cons", "Constraints", true, Some("constraints")),
            UiControl::toggle("raw", "Show raw JSON", false, Some("rawJson")),
        ],
        title: None,
    };
    std::fs::write(
        "out/plugin/lp.html",
        render_player_html(&lp, &run_with("lp", PluginOutput::Json(lp_demo_result()))),
    )?;
    Ok(())
}

fn queue_demo_frames() -> Vec<Value> {
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 32) as u32) as f64 / u32::MAX as f64
    };

    let lambda = 0.55;
    let mu = 0.70;
    let steps = 160;
    let mut n: i64 = 0;
    let mut frames = Vec::with_capacity(steps);

    for tick in 0..steps {
        if next() < lambda {
            n += 1;
        }
        if n > 0 && next() < mu {
            n -= 1;
        }
        let busy = if n > 0 { 1 } else { 0 };
        let mut shapes = vec![json!({
            "kind": "rect",
            "x": 40,
            "y": 150,
            "w": 80,
            "h": 80,
            "rx": 10,
            "fill": if busy == 1 { "#16a34a" } else { "#cbd5e1" },
            "stroke": "#0f172a",
            "strokeWidth": 2,
            "label": "server"
        })];
        for i in 0..n.min(12) {
            shapes.push(json!({
                "kind": "circle",
                "x": 170 + i * 36,
                "y": 190,
                "r": 14,
                "fill": "#2563eb",
                "stroke": "#1e3a8a",
                "strokeWidth": 1.5
            }));
        }
        if n > 12 {
            shapes.push(json!({
                "kind": "text",
                "x": 620,
                "y": 196,
                "text": format!("+{} more", n - 12),
                "fontSize": 13,
                "fill": "#475569"
            }));
        }
        frames.push(json!({
            "t": tick as f64,
            "tick": tick,
            "n": n,
            "serverBusy": busy,
            "shapes": shapes,
            "caption": format!(
                "tick {tick} - {n} in system, server {}",
                if busy == 1 { "busy" } else { "idle" }
            )
        }));
    }
    frames
}

fn lp_demo_result() -> Value {
    let variables = json!([
        { "name": "pumps", "value": 12.0, "objective": 44.0, "reducedCost": 0.0, "lower": 0.0, "upper": 24.0, "basis": "basic" },
        { "name": "valves", "value": 18.0, "objective": 31.0, "reducedCost": 0.0, "lower": 0.0, "upper": 30.0, "basis": "basic" },
        { "name": "motors", "value": 7.0, "objective": 86.0, "reducedCost": 0.0, "lower": 0.0, "upper": 12.0, "basis": "basic" },
        { "name": "controllers", "value": 5.0, "objective": 73.0, "reducedCost": 0.0, "lower": 0.0, "upper": 10.0, "basis": "basic" },
        { "name": "frames", "value": 10.0, "objective": 22.0, "reducedCost": 0.0, "lower": 0.0, "upper": 18.0, "basis": "basic" },
        { "name": "sensors", "value": 14.0, "objective": 18.0, "reducedCost": 0.0, "lower": 0.0, "upper": 25.0, "basis": "basic" },
        { "name": "premium_kits", "value": 0.0, "objective": 105.0, "reducedCost": -8.25, "lower": 0.0, "upper": 8.0, "basis": "nonbasic" },
        { "name": "rush_service", "value": 9.0, "objective": 16.0, "reducedCost": 0.0, "lower": 0.0, "upper": null, "basis": "basic" }
    ]);
    let constraints = json!([
        { "name": "labor_hours", "sense": "<=", "activity": 620.0, "rhs": 620.0, "residual": 0.0, "dual": 1.50, "binding": true },
        { "name": "cnc_hours", "sense": "<=", "activity": 360.0, "rhs": 360.0, "residual": 0.0, "dual": 2.25, "binding": true },
        { "name": "assembly_slots", "sense": "<=", "activity": 392.0, "rhs": 410.0, "residual": 18.0, "dual": 0.0, "binding": false },
        { "name": "steel_kg", "sense": "<=", "activity": 900.0, "rhs": 900.0, "residual": 0.0, "dual": 0.42, "binding": true },
        { "name": "electronics_units", "sense": "<=", "activity": 480.0, "rhs": 480.0, "residual": 0.0, "dual": 1.10, "binding": true },
        { "name": "packaging_units", "sense": "<=", "activity": 211.0, "rhs": 260.0, "residual": 49.0, "dual": 0.0, "binding": false },
        { "name": "min_pumps_contract", "sense": ">=", "activity": 12.0, "rhs": 10.0, "residual": 2.0, "dual": 0.0, "binding": false },
        { "name": "min_valves_contract", "sense": ">=", "activity": 18.0, "rhs": 15.0, "residual": 3.0, "dual": 0.0, "binding": false },
        { "name": "shipping_pallets", "sense": "<=", "activity": 180.0, "rhs": 180.0, "residual": 0.0, "dual": 0.35, "binding": true },
        { "name": "quality_budget", "sense": "<=", "activity": 87.0, "rhs": 95.0, "residual": 8.0, "dual": 0.0, "binding": false }
    ]);
    json!({
        "status": "optimal",
        "objectiveSense": "max",
        "objective": 2665.0,
        "iterations": 18,
        "solveMs": 3.84,
        "algorithm": "revised-simplex",
        "variableCount": 8,
        "constraintCount": 10,
        "variables": variables,
        "constraints": constraints
    })
}

fn generate_wind_mppt_pages() {
    crate::des::main_wind_mppt_anim::run_controller("optimal-torque");
    crate::des::main_wind_mppt_anim::run_controller("pi");
}

fn generate_dc_motor_pages() {
    let original_mode = std::env::var("MODE").ok();
    std::env::remove_var("MODE");
    crate::des::main_dc_motor_anim::run();
    std::env::set_var("MODE", "open");
    crate::des::main_dc_motor_anim::run();
    match original_mode {
        Some(value) => std::env::set_var("MODE", value),
        None => std::env::remove_var("MODE"),
    }
}

fn generate_two_disease_page() {
    let original_animate = std::env::var("ANIMATE").ok();
    std::env::set_var("ANIMATE", "1");
    crate::des::main_two_disease::run();
    match original_animate {
        Some(value) => std::env::set_var("ANIMATE", value),
        None => std::env::remove_var("ANIMATE"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn registry_covers_all_numerical_solvers() {
        let slugs = [
            "lbfgs",
            "sequence-alignment",
            "metropolis-hastings",
            "differential-evolution",
            "prim-mst",
            "backprop-mlp",
            "gaussian-mixture-em",
            "mean-field-vi",
        ];
        let hrefs: HashSet<&str> = NUMERICAL_SOLVERS.iter().map(|s| s.href).collect();
        for slug in slugs {
            let expected = format!("numerical-solvers/{slug}.html");
            assert!(
                hrefs.contains(expected.as_str()),
                "missing index entry for {slug}"
            );
        }
    }

    #[test]
    fn registry_covers_featured_control_players() {
        let hrefs: HashSet<&str> = CONTROL_ANIMATIONS.iter().map(|s| s.href).collect();
        for expected in [
            "wind-mppt/animation-optimal-torque.html",
            "wind-mppt/animation-pi.html",
            "dc-motor/animation-closed.html",
            "dc-motor/animation-open.html",
            "obs-ctrl/animation.html",
            "empirical-control/player.html",
            "temp-control/animation.html",
            "temp-control/animation-heat-cool.html",
        ] {
            assert!(
                hrefs.contains(expected),
                "missing index entry for {expected}"
            );
        }
    }

    #[test]
    fn embedded_plugin_demo_payloads_are_nonempty() {
        assert_eq!(queue_demo_frames().len(), 160);
        let lp = lp_demo_result();
        assert_eq!(lp["status"].as_str(), Some("optimal"));
        assert_eq!(lp["variableCount"].as_u64(), Some(8));
    }

    #[test]
    fn registry_hrefs_are_unique() {
        let mut seen = HashSet::new();
        for group in html_index_groups() {
            for entry in group.entries {
                assert!(seen.insert(entry.href), "duplicate href: {}", entry.href);
            }
        }
    }
}
