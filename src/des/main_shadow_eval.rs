//! Dual / shadow simulation evaluation demo.
//!
//! Treats each simulation as a black box, spins up *shadow copies*, perturbs
//! them, and recovers the controllability & observability Gramians from the
//! responses alone — then quantifies them and (for the linear cases)
//! cross-checks against the analytic Gramian of the known model.
//!
//! Headliner: the back-EMF DC motor (`des::general::control_systems::dc_motor`)
//! is probed exactly as it runs in simulation (RK4 on `L di/dt = V − R i − K_e ω`,
//! `J dω̇ = K_t i − B ω`), and the shadow verdict is shown to agree with the
//! analytic `state_space()` Gramian.
//!
//! It also abstracts the motor's speed regimes into a coarse MDP/POMDP and
//! re-asks controllability (reachability) and observability (sensor
//! distinguishability) through the decision-process lens.
//!
//! Writes `out/shadow-eval/report.json` (machine-readable) and
//! `out/shadow-eval/report.html` (styled report), and prints the summary to
//! stdout. Delegates to
//! [`crate::des::general::control_systems::shadow_eval`].
#![allow(dead_code)]

use std::path::Path;

use serde::Serialize;

use crate::des::animation::run_report::{MetricRow, ReportSection, RunReportPage};
use crate::des::general::control_systems::dc_motor::DcMotorParams;
use crate::des::general::control_systems::observability_controllability::{
    StateSpaceModel, StateSpaceSpec,
};
use crate::des::general::control_systems::shadow_eval::{
    assess_nested, attach_lti_cross_check, build_regime_mdp_sampled, evaluate_shadow,
    DcMotorShadowPlant, LtiPlant, NestedAssessment, ShadowEvalOpts, ShadowReport,
};

/// A balanced demo motor: moderate electrical (τ_e = L/R = 0.1 s) and mechanical
/// (τ_mech = J/B = 0.5 s) time constants so the speed regimes are crossable in a
/// reasonable macro-step, and `K_e = K_t` (a consistent PM motor).
fn demo_motor_params() -> DcMotorParams {
    DcMotorParams {
        resistance: 1.0,
        inductance: 0.1,
        back_emf_constant: 0.05,
        torque_constant: 0.05,
        inertia: 0.01,
        friction: 0.02,
    }
}

fn double_integrator() -> StateSpaceModel {
    StateSpaceModel::new(StateSpaceSpec {
        a: vec![vec![0.0, 1.0], vec![0.0, 0.0]],
        b: vec![vec![0.0], vec![1.0]],
        c: vec![vec![1.0, 0.0]],
        d: None,
    })
}

/// Two decoupled stable modes: B drives only mode 0, C sees only mode 0 — so the
/// shadow probe must flag mode 1 as both uncontrollable and unobservable.
fn decoupled_modes() -> StateSpaceModel {
    StateSpaceModel::new(StateSpaceSpec {
        a: vec![vec![-1.0, 0.0], vec![0.0, -2.0]],
        b: vec![vec![1.0], vec![0.0]],
        c: vec![vec![1.0, 0.0]],
        d: None,
    })
}

#[derive(Serialize)]
struct ShadowArtifact {
    title: String,
    description: String,
    systems: Vec<ShadowReport>,
    nested: Vec<NestedAssessment>,
}

struct ShadowEvalDemo;

impl ShadowEvalDemo {
    fn run(&self) {
        let mut log = String::new();
        let mut systems: Vec<ShadowReport> = Vec::new();

        log.push_str("============ Dual / shadow controllability & observability ============\n");
        log.push_str(
            "Each system is probed as a BLACK BOX: shadow copies are perturbed and the\n\
             controllability / observability Gramians are recovered from the responses.\n\n",
        );

        // ── 1. Back-EMF DC motor (the real RK4 plant) ──
        let motor = DcMotorShadowPlant::new(demo_motor_params());
        let motor_opts = ShadowEvalOpts {
            horizon: 200,
            dt: 0.01,
            epsilon: 1e-4,
            keep_matrices: false,
        };
        let mut motor_report =
            evaluate_shadow("DC motor (back-EMF, RK4 plant)", &motor, &motor_opts);
        attach_lti_cross_check(&mut motor_report, &motor.state_space_model(), false);
        log.push_str(&motor_report.summary_lines());
        log.push('\n');
        systems.push(motor_report);

        // ── 2. Double integrator (controllable + observable) ──
        let di = double_integrator();
        let mut di_report = evaluate_shadow(
            "double integrator",
            &LtiPlant::new(&di),
            &ShadowEvalOpts::default(),
        );
        attach_lti_cross_check(&mut di_report, &di, false);
        log.push_str(&di_report.summary_lines());
        log.push('\n');
        systems.push(di_report);

        // ── 3. Decoupled modes (mode 1 hidden + undrivable) ──
        let dec = decoupled_modes();
        let mut dec_report = evaluate_shadow(
            "decoupled modes",
            &LtiPlant::new(&dec),
            &ShadowEvalOpts::default(),
        );
        attach_lti_cross_check(&mut dec_report, &dec, false);
        log.push_str(&dec_report.summary_lines());
        log.push('\n');
        systems.push(dec_report);

        // ── 4. Nested MDP / POMDP: abstract the motor's speed regimes ──
        log.push_str("\n============ Nested MDP/POMDP abstraction of the motor ============\n");
        log.push_str(
            "Speed ω is bucketed into 3 regimes {slow, mid, fast}; brake/cruise/drive\n\
             voltages are the actions. A SHORT macro-step is shadow-simulated from a\n\
             spread of starting speeds in each regime, so transitions are stochastic\n\
             and the sensor genuinely limits how well the regime can be inferred.\n\n",
        );
        // ω_ss ≈ V / (R·B/K_t + K_e) = V / 0.45.  V = 0 → slow, 2.5 → mid, 6 → fast.
        let edges = [3.0, 9.0];
        // A spread of starting states ([current, ω]) spanning each regime's ω band.
        let regime_samples = vec![
            vec![
                vec![0.0, 0.4],
                vec![0.0, 1.2],
                vec![0.0, 2.0],
                vec![0.0, 2.7],
            ],
            vec![
                vec![0.0, 3.3],
                vec![0.0, 5.0],
                vec![0.0, 6.5],
                vec![0.0, 8.6],
            ],
            vec![
                vec![0.0, 9.4],
                vec![0.0, 11.0],
                vec![0.0, 13.0],
                vec![0.0, 14.6],
            ],
        ];
        let actions = vec![vec![0.0], vec![2.5], vec![8.0]];
        let macro_dt = 0.01;
        let macro_steps = 30; // 0.3 s (< τ_mech) → partial, history-dependent moves
        let mdp_spec = build_regime_mdp_sampled(
            &motor,
            0,
            &edges,
            &regime_samples,
            &actions,
            macro_steps,
            macro_dt,
        );
        log.push_str(&Self::format_mdp(&mdp_spec));

        let sharp = assess_nested("motor speed regimes (calibrated tacho)", &mdp_spec, 0.85);
        let blurry = assess_nested("motor speed regimes (noisy tacho)", &mdp_spec, 0.45);
        log.push_str(&Self::format_nested(&sharp));
        log.push_str(&Self::format_nested(&blurry));

        let nested = vec![sharp, blurry];

        print!("{log}");

        // ── Artifacts ──
        let artifact = ShadowArtifact {
            title: "Dual / shadow controllability & observability".to_string(),
            description:
                "Empirical Gramians recovered from perturbed shadow simulations, with analytic \
                 cross-checks and a nested MDP/POMDP abstraction of the DC-motor speed regimes."
                    .to_string(),
            systems,
            nested,
        };
        let dir = Path::new("out").join("shadow-eval");
        let _ = std::fs::create_dir_all(&dir);
        if let Ok(json) = serde_json::to_string_pretty(&artifact) {
            let _ = std::fs::write(dir.join("report.json"), json);
        }
        let html = Self::render_html(&artifact, &log);
        let out = dir.join("report.html");
        let _ = std::fs::write(&out, html);
        let resolved = std::fs::canonicalize(&out).unwrap_or(out);
        println!("\nShadow-eval report: {}", resolved.display());
        println!("Shadow-eval JSON  : {}", dir.join("report.json").display());
    }

    fn format_mdp(
        spec: &crate::des::general::control_systems::observability_controllability::MdpSpec,
    ) -> String {
        let names = ["slow", "mid", "fast"];
        let act = ["brake", "cruise", "drive"];
        let mut s = String::from("  regime transition pmf  P(next | regime, action):\n");
        for a in 0..spec.num_actions {
            s.push_str(&format!(
                "    action={}:\n",
                act.get(a).copied().unwrap_or("a")
            ));
            for st in 0..spec.num_states {
                let row: Vec<String> = (0..spec.num_states)
                    .map(|t| {
                        format!(
                            "{}:{:.2}",
                            names.get(t).copied().unwrap_or("?"),
                            spec.transition[a][st][t]
                        )
                    })
                    .collect();
                s.push_str(&format!(
                    "      from {:<4} → [{}]\n",
                    names.get(st).copied().unwrap_or("?"),
                    row.join("  ")
                ));
            }
        }
        s.push('\n');
        s
    }

    fn format_nested(n: &NestedAssessment) -> String {
        let mut s = format!("  ── {} (conf {:.2}) ──\n", n.label, n.sensor_confidence);
        s.push_str(&format!(
            "    CONTROLLABILITY: structurally controllable = {}; reachable {}/{} pairs ({:.0}%)\n",
            n.mdp_structurally_controllable,
            n.reachable_pairs,
            n.num_regimes * n.num_regimes,
            n.reachable_fraction * 100.0
        ));
        s.push_str(&format!(
            "      random-policy reach degree / target: [{}]\n",
            Self::vec(&n.per_target_reach_degree)
        ));
        s.push_str(&format!(
            "    OBSERVABILITY:   structurally observable = {}; belief hit-prob: [{}]\n",
            n.pomdp_structurally_observable,
            Self::vec(&n.belief_hit_probability)
        ));
        s.push_str(&format!(
            "      distinguishability degree ∈ [{:.2}, {:.2}]\n\n",
            n.distinguishability_min, n.distinguishability_max
        ));
        s
    }

    fn vec(v: &[f64]) -> String {
        v.iter()
            .map(|x| {
                if x.is_finite() {
                    format!("{x:.2}")
                } else {
                    "∞".to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn sci(x: f64) -> String {
        if x.is_finite() {
            format!("{x:.3e}")
        } else {
            "∞".to_string()
        }
    }

    fn render_html(artifact: &ShadowArtifact, log: &str) -> String {
        let mut page = RunReportPage::new(
            &artifact.title,
            "Black-box obs/ctrl: probe a running simulation with perturbed shadow copies.",
        );
        page.add_section(ReportSection {
            heading: "What this run measures".to_string(),
            description: Some(
                "Rather than reading controllability/observability off known A/B/C matrices, each \
                 system is treated as a black box. Shadow copies are perturbed — one-step input \
                 impulses for the controllability Gramian, ±state nudges for the observability \
                 Gramian — and the Gramians are recovered from the responses alone (so a nonlinear \
                 plant works the same way). Smallest eigenvalue = hardest direction; condition \
                 number = anisotropy; numeric rank = the robust structural verdict. For the linear \
                 cases the shadow estimate is cross-checked against the analytic Gramian."
                    .to_string(),
            ),
            metrics: None,
            log: None,
        });

        for r in &artifact.systems {
            let cross = r
                .cross_check_rel_error
                .map(|e| format!("{e:.2e}"))
                .unwrap_or_else(|| "—".to_string());
            let metrics = vec![
                MetricRow {
                    label: "dimensions (n, m, p)".to_string(),
                    value: format!("{}, {}, {}", r.state_dim, r.input_dim, r.output_dim),
                },
                MetricRow {
                    label: "probe (H, dt, ε)".to_string(),
                    value: format!("{}, {}, {:.0e}", r.horizon, r.dt, r.epsilon),
                },
                MetricRow {
                    label: "CONTROLLABLE".to_string(),
                    value: format!(
                        "{}  (rank {}/{})",
                        if r.controllable { "yes" } else { "NO" },
                        r.empirical_controllability.numeric_rank,
                        r.state_dim
                    ),
                },
                MetricRow {
                    label: "W_c eigenvalues (min … max)".to_string(),
                    value: format!(
                        "{} … {}   cond {}",
                        Self::sci(r.empirical_controllability.min),
                        Self::sci(r.empirical_controllability.max),
                        Self::sci(r.empirical_controllability.condition_number)
                    ),
                },
                MetricRow {
                    label: "hardest-to-drive direction".to_string(),
                    value: format!(
                        "[{}]",
                        Self::vec(&r.empirical_controllability.weakest_direction)
                    ),
                },
                MetricRow {
                    label: "OBSERVABLE".to_string(),
                    value: format!(
                        "{}  (rank {}/{})",
                        if r.observable { "yes" } else { "NO" },
                        r.empirical_observability.numeric_rank,
                        r.state_dim
                    ),
                },
                MetricRow {
                    label: "W_o eigenvalues (min … max)".to_string(),
                    value: format!(
                        "{} … {}   cond {}",
                        Self::sci(r.empirical_observability.min),
                        Self::sci(r.empirical_observability.max),
                        Self::sci(r.empirical_observability.condition_number)
                    ),
                },
                MetricRow {
                    label: "hardest-to-see direction".to_string(),
                    value: format!(
                        "[{}]",
                        Self::vec(&r.empirical_observability.weakest_direction)
                    ),
                },
                MetricRow {
                    label: "shadow vs analytic (rel. eigenvalue err)".to_string(),
                    value: cross,
                },
            ];
            page.add_section(ReportSection {
                heading: r.label.clone(),
                description: None,
                metrics: Some(metrics),
                log: None,
            });
        }

        for n in &artifact.nested {
            let metrics = vec![
                MetricRow {
                    label: "regimes × actions".to_string(),
                    value: format!("{} × {}", n.num_regimes, n.num_actions),
                },
                MetricRow {
                    label: "sensor confidence".to_string(),
                    value: format!("{:.2}", n.sensor_confidence),
                },
                MetricRow {
                    label: "MDP structurally controllable".to_string(),
                    value: format!(
                        "{}  ({:.0}% of state pairs reachable)",
                        n.mdp_structurally_controllable,
                        n.reachable_fraction * 100.0
                    ),
                },
                MetricRow {
                    label: "random-policy reach degree / target".to_string(),
                    value: format!("[{}]", Self::vec(&n.per_target_reach_degree)),
                },
                MetricRow {
                    label: "POMDP structurally observable".to_string(),
                    value: format!("{}", n.pomdp_structurally_observable),
                },
                MetricRow {
                    label: "belief hit-probability / regime".to_string(),
                    value: format!("[{}]", Self::vec(&n.belief_hit_probability)),
                },
                MetricRow {
                    label: "distinguishability degree".to_string(),
                    value: format!(
                        "[{:.2}, {:.2}]",
                        n.distinguishability_min, n.distinguishability_max
                    ),
                },
            ];
            page.add_section(ReportSection {
                heading: format!("Nested · {}", n.label),
                description: None,
                metrics: Some(metrics),
                log: None,
            });
        }

        page.add_section(ReportSection {
            heading: "Run output".to_string(),
            description: None,
            metrics: None,
            log: Some(log.to_string()),
        });
        page.to_html()
    }
}

/// Entry point.
pub fn run() {
    ShadowEvalDemo.run();
}
