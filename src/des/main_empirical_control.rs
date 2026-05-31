//! Port of `src/des/main-empirical-control.ts`.
//!
//! Measures the DEGREE of controllability / observability numerically
//! (Gramians) and empirically (Monte-Carlo trials), rather than the binary
//! Kalman rank verdict.
//!
//! Delegates to `crate::des::general::control_systems::{observability_controllability,
//! empirical_control}` and the iterative DES runner. Monte-Carlo trials are
//! reproducible via the `seed` option fields.

#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use serde_json::{json, Value};

use crate::des::general::control_systems::empirical_control::{
    ControllabilityGramian, DegreeKind, DegreeReportSinkStation, DegreeReportToken,
    DiscreteLinearSystem, DiscreteSystemSourceStation, DiscreteSystemToken, EmpiricalChannels,
    LtiDegreeEvaluatorStation, MdpControllabilityDegree, MdpDegreeEvaluatorStation,
    MdpDegreeSourceStation, MdpDegreeToken, MonteCarloControllability,
    MonteCarloControllabilityOpts, MonteCarloDistinguishability, MonteCarloObservability,
    MonteCarloObservabilityOpts, ObservabilityGramian, PomdpDegreeEvaluatorStation,
    PomdpDegreeSourceStation, PomdpDegreeToken, RandomPolicyOpts,
};
use crate::des::general::control_systems::observability_controllability::{
    MarkovDecisionProcess, MdpSpec, PartiallyObservableProcess, PomdpSpec, StateSpaceModel,
    StateSpaceSpec,
};
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::station::{DESStation, StationRef};
use crate::des::model::RunArtifact;
use crate::des::plugin::UiControl;

struct EmpiricalControlDemo {
    horizon: usize,
    dt: f64,
}

impl EmpiricalControlDemo {
    fn new() -> Self {
        EmpiricalControlDemo {
            horizon: 40,
            dt: 0.02,
        }
    }

    /// The "real system": DC motor, state [i, ω], input V, output ω.
    fn dc_motor(&self) -> DiscreteLinearSystem {
        let (r, l, ke, kt, j, b) = (2.0, 0.5, 0.1, 0.1, 0.02, 0.002);
        let model = StateSpaceModel::new(StateSpaceSpec {
            a: vec![vec![-r / l, -ke / l], vec![kt / j, -b / j]],
            b: vec![vec![1.0 / l], vec![0.0]],
            c: vec![vec![0.0, 1.0]],
            d: None,
        });
        DiscreteLinearSystem::from_continuous(&model, self.dt)
    }

    /// Double integrator (controllable + observable).
    fn double_integrator(&self) -> DiscreteLinearSystem {
        let model = StateSpaceModel::new(StateSpaceSpec {
            a: vec![vec![0.0, 1.0], vec![0.0, 0.0]],
            b: vec![vec![0.0], vec![1.0]],
            c: vec![vec![1.0, 0.0]],
            d: None,
        });
        DiscreteLinearSystem::from_continuous(&model, self.dt)
    }

    /// Decoupled modes: mode 2 is invisible and undrivable.
    fn decoupled(&self) -> DiscreteLinearSystem {
        let model = StateSpaceModel::new(StateSpaceSpec {
            a: vec![vec![-1.0, 0.0], vec![0.0, -2.0]],
            b: vec![vec![1.0], vec![0.0]],
            c: vec![vec![1.0, 0.0]],
            d: None,
        });
        DiscreteLinearSystem::from_continuous(&model, self.dt)
    }

    fn ring_mdp() -> MarkovDecisionProcess {
        MarkovDecisionProcess::new(MdpSpec {
            num_states: 3,
            num_actions: 1,
            transition: vec![vec![
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![1.0, 0.0, 0.0],
            ]],
        })
    }
    fn trap_mdp() -> MarkovDecisionProcess {
        MarkovDecisionProcess::new(MdpSpec {
            num_states: 3,
            num_actions: 1,
            transition: vec![vec![
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![0.0, 0.0, 1.0],
            ]],
        })
    }
    fn distinct_pomdp() -> PartiallyObservableProcess {
        PartiallyObservableProcess::new(PomdpSpec {
            num_states: 2,
            num_actions: 1,
            transition: vec![vec![vec![0.5, 0.5], vec![0.5, 0.5]]],
            num_observations: 2,
            observation: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        })
    }
    fn aliased_pomdp() -> PartiallyObservableProcess {
        PartiallyObservableProcess::new(PomdpSpec {
            num_states: 2,
            num_actions: 1,
            transition: vec![vec![vec![1.0, 0.0], vec![0.0, 1.0]]],
            num_observations: 2,
            observation: vec![vec![0.5, 0.5], vec![0.5, 0.5]],
        })
    }

    fn run(&self) {
        let systems: Vec<(&str, DiscreteLinearSystem)> = vec![
            ("DC motor (real system)", self.dc_motor()),
            ("double integrator", self.double_integrator()),
            ("decoupled modes", self.decoupled()),
        ];

        // ── DES pipeline: Gramian min/max degree reports ──
        let lti_source = Rc::new(RefCell::new(DiscreteSystemSourceStation::new(
            "lti-src",
            systems
                .iter()
                .map(|(label, sys)| {
                    DiscreteSystemToken::new(label.to_string(), sys.clone(), self.horizon)
                })
                .collect(),
        )));
        let lti_eval = Rc::new(RefCell::new(LtiDegreeEvaluatorStation::new("lti-degree")));
        let mdp_source = Rc::new(RefCell::new(MdpDegreeSourceStation::new(
            "mdp-src",
            vec![
                MdpDegreeToken::new("ring MDP".to_string(), Self::ring_mdp()),
                MdpDegreeToken::new("trap MDP".to_string(), Self::trap_mdp()),
            ],
        )));
        let mdp_eval = Rc::new(RefCell::new(MdpDegreeEvaluatorStation::new("mdp-degree")));
        let pomdp_source = Rc::new(RefCell::new(PomdpDegreeSourceStation::new(
            "pomdp-src",
            vec![
                PomdpDegreeToken::new("distinct sensors".to_string(), Self::distinct_pomdp()),
                PomdpDegreeToken::new("aliased sensors".to_string(), Self::aliased_pomdp()),
            ],
        )));
        let pomdp_eval = Rc::new(RefCell::new(PomdpDegreeEvaluatorStation::new(
            "pomdp-degree",
        )));
        let sink = Rc::new(RefCell::new(DegreeReportSinkStation::new("sink")));

        lti_source.borrow_mut().core_mut().pipe(
            lti_eval.clone() as StationRef,
            EmpiricalChannels::SYSTEM,
            EmpiricalChannels::SYSTEM,
        );
        lti_eval.borrow_mut().core_mut().pipe(
            sink.clone() as StationRef,
            EmpiricalChannels::REPORT,
            EmpiricalChannels::REPORT,
        );
        mdp_source.borrow_mut().core_mut().pipe(
            mdp_eval.clone() as StationRef,
            EmpiricalChannels::MDP,
            EmpiricalChannels::MDP,
        );
        mdp_eval.borrow_mut().core_mut().pipe(
            sink.clone() as StationRef,
            EmpiricalChannels::REPORT,
            EmpiricalChannels::REPORT,
        );
        pomdp_source.borrow_mut().core_mut().pipe(
            pomdp_eval.clone() as StationRef,
            EmpiricalChannels::POMDP,
            EmpiricalChannels::POMDP,
        );
        pomdp_eval.borrow_mut().core_mut().pipe(
            sink.clone() as StationRef,
            EmpiricalChannels::REPORT,
            EmpiricalChannels::REPORT,
        );
        run_iterative_des(
            vec![
                lti_source.clone() as StationRef,
                lti_eval.clone() as StationRef,
                mdp_source.clone() as StationRef,
                mdp_eval.clone() as StationRef,
                pomdp_source.clone() as StationRef,
                pomdp_eval.clone() as StationRef,
                sink.clone() as StationRef,
            ],
            IterativeRunOptions {
                shuffle: false,
                max_ticks: Some(20),
                ..Default::default()
            },
        );

        println!("================ Gramian degree reports (DES pipeline) ================");
        for r in &sink.borrow().reports {
            self.print_report(r);
        }

        // ── Direct empirical comparison: analytic Gramian vs Monte-Carlo trials ──
        println!("\n================ Empirical (trial-based) vs analytic ================");
        for (label, sys) in &systems {
            let wc = ControllabilityGramian::new(sys, self.horizon);
            let wo = ObservabilityGramian::new(sys, self.horizon);
            let mc_c = MonteCarloControllability::new(
                sys,
                self.horizon,
                MonteCarloControllabilityOpts {
                    trials: Some(3000),
                    input_bound: Some(1.0),
                    seed: Some(1),
                    ..Default::default()
                },
            )
            .run();
            let mc_o = MonteCarloObservability::new(
                sys,
                self.horizon,
                MonteCarloObservabilityOpts {
                    trials: Some(1500),
                    noise_std: Some(0.02),
                    seed: Some(2),
                    ..Default::default()
                },
            )
            .run();
            println!("\n--- {label} ---");
            println!("  CONTROLLABILITY");
            println!(
                "    W_c eigenvalues (min..max) : [{}]",
                self.vec(&wc.eigenvalues())
            );
            println!(
                "    empirical reach-cloud var  : [{}]  (∝ W_c)",
                self.vec(&mc_c.spread_eigenvalues)
            );
            println!(
                "    least-squares target hit % : {:.1}%   reachRadius={:.3}",
                mc_c.target_success_rate * 100.0,
                mc_c.reach_radius
            );
            println!(
                "    min/max controllability    : {:.2e} / {:.2e}  (cond {})",
                wc.min(),
                wc.max(),
                self.cond(wc.condition_number())
            );
            println!("  OBSERVABILITY");
            println!(
                "    W_o eigenvalues (min..max) : [{}]",
                self.vec(&wo.eigenvalues())
            );
            println!(
                "    recon error (mean/worst)   : {:.4} / {:.4}  @ noise 0.02",
                mc_o.mean_reconstruction_error, mc_o.worst_reconstruction_error
            );
            println!(
                "    min/max observability      : {:.2e} / {:.2e}  (cond {})",
                wo.min(),
                wo.max(),
                self.cond(wo.condition_number())
            );
        }

        // ── MDP hitting times (numerical planning) ──
        println!("\n================ MDP controllability via value iteration ================");
        for (name, mdp) in [("ring", Self::ring_mdp()), ("trap", Self::trap_mdp())] {
            let planner = MdpControllabilityDegree::new(&mdp);
            println!(
                "  {name}: expected steps to reach s0 from [s0,s1,s2] = [{}]",
                self.vec(&planner.expected_hitting_times(0, 10_000, 1e-12))
            );
            println!(
                "        random-policy reach degree per target = [{}]",
                self.vec(&planner.per_target_degree(&RandomPolicyOpts {
                    episodes: Some(600),
                    ..Default::default()
                }))
            );
        }

        // ── POMDP distinguishability (belief tracking) ──
        println!("\n================ POMDP observability via belief tracking ================");
        for (name, pomdp) in [
            ("distinct", Self::distinct_pomdp()),
            ("aliased", Self::aliased_pomdp()),
        ] {
            let r = MonteCarloDistinguishability::new(&pomdp).run(&RandomPolicyOpts {
                episodes: Some(800),
                ..Default::default()
            });
            println!(
                "  {name}: belief hit-prob per state = [{}]   residual entropy = [{}] bits",
                self.vec(&r.hit_probability),
                self.vec(&r.residual_entropy)
            );
        }
    }

    fn build_artifact(&self) -> RunArtifact {
        let systems: Vec<(&str, DiscreteLinearSystem)> = vec![
            ("DC motor (real system)", self.dc_motor()),
            ("double integrator", self.double_integrator()),
            ("decoupled modes", self.decoupled()),
        ];
        let mut frames: Vec<Value> = Vec::new();
        let mut lti_results: Vec<Value> = Vec::new();

        frames.push(json!({
            "t": 0,
            "stage": 0,
            "ltiSystems": systems.len(),
            "mdpSystems": 2,
            "pomdpSystems": 2,
            "caption": "Empirical-control run: LTI Gramian degrees, MDP reachability, and POMDP distinguishability.",
            "shapes": Self::overview_shapes(),
        }));

        for (idx, (label, sys)) in systems.iter().enumerate() {
            let wc = ControllabilityGramian::new(sys, self.horizon);
            let wo = ObservabilityGramian::new(sys, self.horizon);
            let mc_c = MonteCarloControllability::new(
                sys,
                self.horizon,
                MonteCarloControllabilityOpts {
                    trials: Some(3000),
                    input_bound: Some(1.0),
                    seed: Some(1),
                    ..Default::default()
                },
            )
            .run();
            let mc_o = MonteCarloObservability::new(
                sys,
                self.horizon,
                MonteCarloObservabilityOpts {
                    trials: Some(1500),
                    noise_std: Some(0.02),
                    seed: Some(2),
                    ..Default::default()
                },
            )
            .run();

            let wc_eigs = wc.eigenvalues();
            let wo_eigs = wo.eigenvalues();
            let controllability_min = wc.min();
            let controllability_max = wc.max();
            let observability_min = wo.min();
            let observability_max = wo.max();
            let target_hit_pct = mc_c.target_success_rate * 100.0;

            frames.push(json!({
                "t": frames.len(),
                "stage": 1,
                "controllabilityMin": controllability_min,
                "controllabilityMax": controllability_max,
                "observabilityMin": observability_min,
                "observabilityMax": observability_max,
                "targetHitPct": target_hit_pct,
                "reconMean": mc_o.mean_reconstruction_error,
                "caption": format!("{label}: controllability / observability degrees and Monte-Carlo trial estimates."),
                "shapes": Self::lti_shapes(
                    label,
                    idx,
                    controllability_min,
                    controllability_max,
                    observability_min,
                    observability_max,
                    target_hit_pct,
                    mc_o.mean_reconstruction_error,
                    &wc_eigs,
                    &wo_eigs,
                ),
            }));

            lti_results.push(json!({
                "label": label,
                "controllability": {
                    "eigenvalues": Self::json_numbers(&wc_eigs),
                    "min": controllability_min,
                    "max": controllability_max,
                    "conditionNumber": Self::finite_or_string(wc.condition_number()),
                    "empiricalReachVariance": Self::json_numbers(&mc_c.spread_eigenvalues),
                    "targetSuccessRate": mc_c.target_success_rate,
                    "reachRadius": mc_c.reach_radius,
                },
                "observability": {
                    "eigenvalues": Self::json_numbers(&wo_eigs),
                    "min": observability_min,
                    "max": observability_max,
                    "conditionNumber": Self::finite_or_string(wo.condition_number()),
                    "meanReconstructionError": mc_o.mean_reconstruction_error,
                    "worstReconstructionError": mc_o.worst_reconstruction_error,
                },
            }));
        }

        let mdp_specs = vec![("ring", Self::ring_mdp()), ("trap", Self::trap_mdp())];
        let mut mdp_results = Vec::new();
        for (name, mdp) in mdp_specs {
            let planner = MdpControllabilityDegree::new(&mdp);
            let hitting = planner.expected_hitting_times(0, 10_000, 1e-12);
            let reach = planner.per_target_degree(&RandomPolicyOpts {
                episodes: Some(600),
                ..Default::default()
            });
            frames.push(json!({
                "t": frames.len(),
                "stage": 2,
                "reachS0": reach.first().copied().unwrap_or(0.0),
                "reachS1": reach.get(1).copied().unwrap_or(0.0),
                "reachS2": reach.get(2).copied().unwrap_or(0.0),
                "expectedStepsS0": hitting.first().copied().filter(|v| v.is_finite()).unwrap_or(0.0),
                "expectedStepsS1": hitting.get(1).copied().filter(|v| v.is_finite()).unwrap_or(0.0),
                "expectedStepsS2": hitting.get(2).copied().filter(|v| v.is_finite()).unwrap_or(0.0),
                "caption": format!("{name} MDP: expected hitting time to s0 and random-policy reach degree."),
                "shapes": Self::mdp_shapes(name, &hitting, &reach),
            }));
            mdp_results.push(json!({
                "name": name,
                "expectedStepsToS0": Self::json_numbers(&hitting),
                "randomPolicyReachDegree": Self::json_numbers(&reach),
            }));
        }

        let pomdp_specs = vec![
            ("distinct", Self::distinct_pomdp()),
            ("aliased", Self::aliased_pomdp()),
        ];
        let mut pomdp_results = Vec::new();
        for (name, pomdp) in pomdp_specs {
            let r = MonteCarloDistinguishability::new(&pomdp).run(&RandomPolicyOpts {
                episodes: Some(800),
                ..Default::default()
            });
            frames.push(json!({
                "t": frames.len(),
                "stage": 3,
                "hit0": r.hit_probability.first().copied().unwrap_or(0.0),
                "hit1": r.hit_probability.get(1).copied().unwrap_or(0.0),
                "entropy0": r.residual_entropy.first().copied().unwrap_or(0.0),
                "entropy1": r.residual_entropy.get(1).copied().unwrap_or(0.0),
                "caption": format!("{name} POMDP sensors: belief hit probability and residual entropy."),
                "shapes": Self::pomdp_shapes(name, &r.hit_probability, &r.residual_entropy),
            }));
            pomdp_results.push(json!({
                "name": name,
                "beliefHitProbability": Self::json_numbers(&r.hit_probability),
                "residualEntropyBits": Self::json_numbers(&r.residual_entropy),
            }));
        }

        RunArtifact::sim(
            "empirical-control",
            "Empirical Controllability & Observability Player",
            "Playable frame-by-frame view of the LTI, MDP, and POMDP degree checks from the empirical-control run.",
            frames,
            json!({
                "horizon": self.horizon,
                "dt": self.dt,
                "lti": lti_results,
                "mdp": mdp_results,
                "pomdp": pomdp_results,
            }),
            vec![
                UiControl::range("speed", "Speed (fps)", 1.0, 20.0, 1.0, 2.0),
                UiControl::select(
                    "metric",
                    "Feature signal",
                    &[
                        "all",
                        "controllabilityMin",
                        "controllabilityMax",
                        "observabilityMin",
                        "observabilityMax",
                        "targetHitPct",
                        "reconMean",
                    ],
                    "all",
                    Some("metric"),
                ),
            ],
            "LTI, MDP, and POMDP degree checks rendered as a playable structured run.",
        )
    }

    fn overview_shapes() -> Vec<Value> {
        vec![
            Self::text(40.0, 42.0, "Empirical control run", 22.0, "#0f172a"),
            Self::text(
                40.0,
                76.0,
                "The player steps through the same three analysis families the report prints.",
                13.0,
                "#475569",
            ),
            Self::box_step(48.0, 130.0, "LTI systems", "Gramians + Monte-Carlo"),
            Self::box_step(300.0, 130.0, "MDPs", "hitting time + reach degree"),
            Self::box_step(552.0, 130.0, "POMDPs", "belief hit + entropy"),
            Self::line(232.0, 170.0, 288.0, 170.0, "#94a3b8", 2.0),
            Self::line(484.0, 170.0, 540.0, 170.0, "#94a3b8", 2.0),
        ]
    }

    fn lti_shapes(
        label: &str,
        idx: usize,
        c_min: f64,
        c_max: f64,
        o_min: f64,
        o_max: f64,
        hit_pct: f64,
        recon_mean: f64,
        wc_eigs: &[f64],
        wo_eigs: &[f64],
    ) -> Vec<Value> {
        let colors = ["#2563eb", "#16a34a", "#dc2626"];
        let accent = colors[idx % colors.len()];
        let c_scale = c_max.max(1e-12);
        let o_scale = o_max.max(1e-12);
        let mut s = vec![
            Self::text(36.0, 35.0, label, 20.0, "#0f172a"),
            Self::text(
                36.0,
                62.0,
                &format!(
                    "Wc eigs [{}]    Wo eigs [{}]",
                    Self::fmt_vec(wc_eigs),
                    Self::fmt_vec(wo_eigs)
                ),
                12.0,
                "#475569",
            ),
            Self::text(36.0, 106.0, "controllability", 13.0, "#334155"),
            Self::text(36.0, 210.0, "observability", 13.0, "#334155"),
        ];
        s.extend(Self::bar_pair(
            170.0, 92.0, c_min, c_max, c_scale, accent, "Wc min", "Wc max",
        ));
        s.extend(Self::bar_pair(
            170.0, 196.0, o_min, o_max, o_scale, "#7c3aed", "Wo min", "Wo max",
        ));
        s.push(Self::text(
            36.0,
            328.0,
            &format!("least-squares target hit: {hit_pct:.1}%"),
            14.0,
            "#0f172a",
        ));
        s.push(Self::text(
            36.0,
            356.0,
            &format!("mean noisy reconstruction error: {recon_mean:.4}"),
            14.0,
            "#0f172a",
        ));
        s
    }

    fn mdp_shapes(name: &str, hitting: &[f64], reach: &[f64]) -> Vec<Value> {
        let pos = [(155.0, 180.0), (370.0, 105.0), (585.0, 180.0)];
        let edges = if name == "ring" {
            vec![(0usize, 1usize), (1, 2), (2, 0)]
        } else {
            vec![(0usize, 1usize), (1, 2), (2, 2)]
        };
        let mut s = vec![
            Self::text(36.0, 40.0, &format!("{name} MDP"), 20.0, "#0f172a"),
            Self::text(
                36.0,
                68.0,
                "Target is s0. Node fill encodes random-policy reach degree.",
                13.0,
                "#475569",
            ),
        ];
        for (from, to) in edges {
            let (x1, y1) = pos[from];
            let (x2, y2) = pos[to];
            let dx: f64 = x2 - x1;
            let dy: f64 = y2 - y1;
            let len = (dx * dx + dy * dy).sqrt().max(1.0);
            s.push(Self::line(
                x1 + dx / len * 28.0,
                y1 + dy / len * 28.0,
                x2 - dx / len * 28.0,
                y2 - dy / len * 28.0,
                "#94a3b8",
                2.0,
            ));
        }
        for (i, (x, y)) in pos.iter().enumerate() {
            let degree = reach.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            let fill = if degree > 0.99 {
                "#2563eb"
            } else if degree > 0.5 {
                "#f59e0b"
            } else {
                "#ef4444"
            };
            s.push(json!({
                "kind": "circle", "x": x, "y": y, "r": 34.0, "fill": fill,
                "stroke": "#0f172a", "strokeWidth": 2.0,
            }));
            s.push(Self::centered_text(
                *x,
                *y + 5.0,
                &format!("s{i}"),
                16.0,
                "#ffffff",
            ));
            let h = hitting.get(i).copied().unwrap_or(f64::INFINITY);
            s.push(Self::centered_text(
                *x,
                *y + 58.0,
                &format!("E[t]={}", Self::fmt_num(h)),
                12.0,
                "#334155",
            ));
            s.push(Self::centered_text(
                *x,
                *y + 76.0,
                &format!("reach={degree:.2}"),
                12.0,
                "#334155",
            ));
        }
        s
    }

    fn pomdp_shapes(name: &str, hit: &[f64], entropy: &[f64]) -> Vec<Value> {
        let mut s = vec![
            Self::text(
                36.0,
                40.0,
                &format!("{name} POMDP sensors"),
                20.0,
                "#0f172a",
            ),
            Self::text(
                36.0,
                68.0,
                "Blue bars are correct-belief hit probability; red bars are residual entropy.",
                13.0,
                "#475569",
            ),
            Self::line(80.0, 285.0, 640.0, 285.0, "#cbd5e1", 2.0),
        ];
        for i in 0..hit.len().max(entropy.len()).max(1) {
            let x = 150.0 + i as f64 * 230.0;
            let hp = hit.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            let en = entropy.get(i).copied().unwrap_or(0.0).clamp(0.0, 2.0);
            let hp_h = hp * 170.0;
            let en_h = (en / 2.0) * 170.0;
            s.push(json!({
                "kind": "rect", "x": x, "y": 285.0 - hp_h, "w": 54.0, "h": hp_h.max(1.0),
                "rx": 5.0, "fill": "#2563eb",
            }));
            s.push(json!({
                "kind": "rect", "x": x + 70.0, "y": 285.0 - en_h, "w": 54.0, "h": en_h.max(1.0),
                "rx": 5.0, "fill": "#dc2626",
            }));
            s.push(Self::centered_text(
                x + 27.0,
                306.0,
                &format!("hit s{i}"),
                12.0,
                "#334155",
            ));
            s.push(Self::centered_text(
                x + 97.0,
                306.0,
                &format!("H s{i}"),
                12.0,
                "#334155",
            ));
            s.push(Self::centered_text(
                x + 27.0,
                285.0 - hp_h - 8.0,
                &format!("{hp:.2}"),
                12.0,
                "#334155",
            ));
            s.push(Self::centered_text(
                x + 97.0,
                285.0 - en_h - 8.0,
                &format!("{en:.2}"),
                12.0,
                "#334155",
            ));
        }
        s
    }

    fn box_step(x: f64, y: f64, heading: &str, body: &str) -> Value {
        json!({
            "kind": "rect", "x": x, "y": y, "w": 184.0, "h": 82.0, "rx": 8.0,
            "fill": "#e0f2fe", "stroke": "#0284c7", "strokeWidth": 2.0,
            "label": format!("{heading}\n{body}"),
        })
    }

    fn bar_pair(
        x: f64,
        y: f64,
        min_value: f64,
        max_value: f64,
        scale: f64,
        color: &str,
        min_label: &str,
        max_label: &str,
    ) -> Vec<Value> {
        let min_w = Self::bar_width(min_value, scale);
        let max_w = Self::bar_width(max_value, scale);
        vec![
            Self::text(x, y + 14.0, min_label, 12.0, "#64748b"),
            json!({"kind": "rect", "x": x + 78.0, "y": y, "w": min_w, "h": 20.0, "rx": 4.0, "fill": color}),
            Self::text(
                x + 88.0 + min_w,
                y + 15.0,
                &format!("{:.2e}", min_value),
                11.0,
                "#334155",
            ),
            Self::text(x, y + 48.0, max_label, 12.0, "#64748b"),
            json!({"kind": "rect", "x": x + 78.0, "y": y + 34.0, "w": max_w, "h": 20.0, "rx": 4.0, "fill": color}),
            Self::text(
                x + 88.0 + max_w,
                y + 49.0,
                &format!("{:.2e}", max_value),
                11.0,
                "#334155",
            ),
        ]
    }

    fn bar_width(value: f64, scale: f64) -> f64 {
        if value <= 0.0 || !value.is_finite() || scale <= 0.0 {
            2.0
        } else {
            (value / scale).sqrt().clamp(0.0, 1.0) * 250.0
        }
    }

    fn text(x: f64, y: f64, text: &str, font_size: f64, fill: &str) -> Value {
        json!({
            "kind": "text", "x": x, "y": y, "text": text,
            "fontSize": font_size, "fill": fill,
        })
    }

    fn centered_text(x: f64, y: f64, text: &str, font_size: f64, fill: &str) -> Value {
        json!({
            "kind": "text", "x": x, "y": y, "text": text,
            "anchor": "middle", "fontSize": font_size, "fill": fill,
        })
    }

    fn line(x1: f64, y1: f64, x2: f64, y2: f64, stroke: &str, stroke_width: f64) -> Value {
        json!({
            "kind": "line", "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "stroke": stroke, "strokeWidth": stroke_width,
        })
    }

    fn json_numbers(values: &[f64]) -> Vec<Value> {
        values.iter().map(|&v| Self::finite_or_string(v)).collect()
    }

    fn finite_or_string(value: f64) -> Value {
        if value.is_finite() {
            json!(value)
        } else {
            json!("infinity")
        }
    }

    fn fmt_num(value: f64) -> String {
        if value.is_finite() {
            format!("{value:.2}")
        } else {
            "∞".to_string()
        }
    }

    fn fmt_vec(values: &[f64]) -> String {
        values
            .iter()
            .map(|&v| {
                if v.is_finite() {
                    format!("{v:.4}")
                } else {
                    "∞".to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn print_report(&self, r: &DegreeReportToken) {
        let kind = match r.kind {
            DegreeKind::LtiDegree => "lti",
            DegreeKind::MdpDegree => "mdp",
            DegreeKind::PomdpDegree => "pomdp",
        };
        println!("[{}] {}", kind, r.label);
        println!("    {}", r.detail);
    }

    fn vec(&self, v: &[f64]) -> String {
        v.iter()
            .map(|x| {
                if x.is_finite() {
                    format!("{x:.4}")
                } else {
                    "∞".to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn cond(&self, c: f64) -> String {
        if c.is_finite() {
            format!("{c:.1e}")
        } else {
            "∞".to_string()
        }
    }
}

/// Entry point (TS top-level script).
pub fn run() {
    EmpiricalControlDemo::new().run();
}

pub fn build_run_artifact() -> RunArtifact {
    EmpiricalControlDemo::new().build_artifact()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empirical_control_artifact_is_playable() {
        let artifact = build_run_artifact();
        assert_eq!(artifact.kind, "empirical-control");
        assert!(artifact.frames.len() >= 8);
        let html = artifact.to_player_html();
        assert!(html.contains("\"player\":\"sim\""));
        assert!(html.contains("Empirical Controllability"));
    }
}
