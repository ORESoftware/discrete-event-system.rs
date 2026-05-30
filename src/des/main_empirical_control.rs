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

struct EmpiricalControlDemo {
    horizon: usize,
    dt: f64,
}

impl EmpiricalControlDemo {
    fn new() -> Self {
        EmpiricalControlDemo { horizon: 40, dt: 0.02 }
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
            transition: vec![vec![vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0], vec![1.0, 0.0, 0.0]]],
        })
    }
    fn trap_mdp() -> MarkovDecisionProcess {
        MarkovDecisionProcess::new(MdpSpec {
            num_states: 3,
            num_actions: 1,
            transition: vec![vec![vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0], vec![0.0, 0.0, 1.0]]],
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
                .map(|(label, sys)| DiscreteSystemToken::new(label.to_string(), sys.clone(), self.horizon))
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
        let pomdp_eval = Rc::new(RefCell::new(PomdpDegreeEvaluatorStation::new("pomdp-degree")));
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
            IterativeRunOptions { shuffle: false, max_ticks: Some(20), ..Default::default() },
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
            println!("    W_c eigenvalues (min..max) : [{}]", self.vec(&wc.eigenvalues()));
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
            println!("    W_o eigenvalues (min..max) : [{}]", self.vec(&wo.eigenvalues()));
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
        for (name, pomdp) in [("distinct", Self::distinct_pomdp()), ("aliased", Self::aliased_pomdp())] {
            let r = MonteCarloDistinguishability::new(&pomdp)
                .run(&RandomPolicyOpts { episodes: Some(800), ..Default::default() });
            println!(
                "  {name}: belief hit-prob per state = [{}]   residual entropy = [{}] bits",
                self.vec(&r.hit_probability),
                self.vec(&r.residual_entropy)
            );
        }
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
            .map(|x| if x.is_finite() { format!("{x:.4}") } else { "∞".to_string() })
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
