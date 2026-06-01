//! Central registry of HTML artifacts for the landing index and site builder.
//!
//! Every simulation that produces a viewable HTML page registers here with its
//! title, description, and optional generator. [`main_build_site`] reads
//! [`html_index_groups`] for featured cards and calls [`generate_html_artifacts`]
//! during a full rebuild.

#![allow(dead_code)]

use crate::des::animation::run_report::{IndexEntry, IndexGroup};

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
    HtmlIndexSpec {
        kind: "animation",
        title: "Studio Run Player",
        href: "studio/run-player.html",
        description:
            "Playable Studio model execution with live block values, animated wires, and signal timelines.",
    },
    HtmlIndexSpec {
        kind: "animation",
        title: "Studio N2 Player",
        href: "studio/n2-player.html",
        description:
            "OpenMDAO-style N2 dependency matrix player that reveals component coupling and validation status.",
    },
    HtmlIndexSpec {
        kind: "animation",
        title: "Studio Sweep Player",
        href: "studio/sweep-player.html",
        description:
            "Parameter-sweep driver player for design variables, objectives, constraints, and best feasible cases.",
    },
    HtmlIndexSpec {
        kind: "tool",
        title: "Delivery Scheduler",
        href: "delivery-planner.html",
        description:
            "Paste delivery addresses with drop-off windows, solve a time-window route, copy the itinerary, and play the route animation.",
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
        title: "Signal & control analysis methods",
        href: "signal-processing/player.html",
        description:
            "Animated Z, Laplace, Fourier, DFT/FFT, wavelet, Mellin, Radon, and Lagrange state-space analysis.",
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

const CALCULUS_OF_VARIATIONS: &[HtmlIndexSpec] = &[
    HtmlIndexSpec {
        kind: "animation",
        title: "Shortest curve - Euler-Lagrange line",
        href: "calculus-of-variations/shortest-curve.html",
        description:
            "Fixed-endpoint arc-length minimization with the stationary straight-line solution and first-integral diagnostics.",
    },
    HtmlIndexSpec {
        kind: "animation",
        title: "Brachistochrone - cycloid",
        href: "calculus-of-variations/brachistochrone.html",
        description:
            "Classical minimum-time descent problem rendered as a cycloid with endpoint and Beltrami first-integral checks.",
    },
    HtmlIndexSpec {
        kind: "animation",
        title: "Minimal surface - catenoid",
        href: "calculus-of-variations/minimal-surface-catenoid.html",
        description:
            "Surface-area variational problem between equal coaxial rings, animated as the catenoid stationary surface.",
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
        kind: "interactive model",
        title: "Vehicle jump planner",
        href: "vehicle-jump/player.html",
        description: "Non-linear ramp-jump trajectory planner with wind vector, atmospheric density, ramp angle, distance, and landing slope controls.",
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
            heading: "Calculus of variations",
            blurb: "Analytic variational problem models with playable stationary curves, Euler-Lagrange equations, first integrals, and diagnostics.",
            entries: CALCULUS_OF_VARIATIONS,
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
    match crate::des::studio::write_studio_player_html(
        "out",
        &crate::des::studio::starter_model_spec(),
    ) {
        Ok(paths) => {
            for path in paths {
                eprintln!("  • {}", path.display());
            }
        }
        Err(e) => eprintln!("  ! Studio player generation failed: {e}"),
    }
    eprintln!("Generating Delivery Scheduler...");
    crate::des::delivery_planner::write_delivery_planner_artifacts();

    eprintln!("Regenerating control-system animations...");
    generate_wind_mppt_pages();
    generate_dc_motor_pages();
    crate::des::main_observability_controllability_anim::run();
    match crate::des::main_signal_processing::write_signal_processing_player_html("out") {
        Ok(path) => eprintln!("  • {}", path.display()),
        Err(e) => eprintln!("  ! signal-processing player generation failed: {e}"),
    }

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

    eprintln!("Regenerating calculus-of-variations animations...");
    crate::des::main_calculus_of_variations_anim::run();

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

    eprintln!("Generating vehicle jump planner...");
    match crate::des::main_vehicle_jump::write_vehicle_jump_player_html("out") {
        Ok(path) => eprintln!("  • {}", path.display()),
        Err(e) => eprintln!("  ! vehicle jump planner generation failed: {e}"),
    }

    eprintln!("Generating two-disease epidemic animation...");
    generate_two_disease_page();
}

fn generate_wind_mppt_pages() {
    let original_controller = std::env::var("CONTROLLER").ok();
    std::env::remove_var("CONTROLLER");
    crate::des::main_wind_mppt_anim::run();
    std::env::set_var("CONTROLLER", "pi");
    crate::des::main_wind_mppt_anim::run();
    match original_controller {
        Some(value) => std::env::set_var("CONTROLLER", value),
        None => std::env::remove_var("CONTROLLER"),
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
    fn registry_covers_calculus_of_variations_players() {
        let hrefs: HashSet<&str> = CALCULUS_OF_VARIATIONS.iter().map(|s| s.href).collect();
        for slug in [
            "shortest-curve",
            "brachistochrone",
            "minimal-surface-catenoid",
        ] {
            let expected = format!("calculus-of-variations/{slug}.html");
            assert!(
                hrefs.contains(expected.as_str()),
                "missing calculus-of-variations index entry for {slug}"
            );
        }
    }

    #[test]
    fn registry_includes_signal_processing_player() {
        let href = crate::des::main_signal_processing::SIGNAL_PROCESSING_PLAYER_REL_PATH;
        assert!(CONTROL_ANIMATIONS.iter().any(|entry| entry.href == href));
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
