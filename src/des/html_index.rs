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

    eprintln!("Regenerating control-system animations...");
    // Wind MPPT scene not yet ported to FrameRecorder — skip until scene exists.
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
    fn registry_hrefs_are_unique() {
        let mut seen = HashSet::new();
        for group in html_index_groups() {
            for entry in group.entries {
                assert!(seen.insert(entry.href), "duplicate href: {}", entry.href);
            }
        }
    }
}
