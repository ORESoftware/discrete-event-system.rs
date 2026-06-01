//! Port of `src/des/main-observability-controllability.ts`.
//!
//! Runnable demo of the general controllability / observability evaluator
//! across linear state-space models, an MDP (reachability), and POMDPs
//! (distinguishability).
//!
//! Conversion notes:
//!   - `class ObsCtrlDemo` → struct + impl; top-level run → [`run`].
//!   - delegates to `general::control_systems::observability_controllability`
//!     and `general::control_systems::dc_motor`.

use std::cell::RefCell;
use std::rc::Rc;

use crate::des::general::control_systems::dc_motor::{DcMotorDynamics, DcMotorParams};
use crate::des::general::control_systems::observability_controllability::{
    ControllabilityEvaluatorStation, EvaluationKind, EvaluationSinkStation, MarkovDecisionProcess,
    MdpControllabilityEvaluatorStation, MdpInformationEvaluatorStation, MdpSourceStation, MdpSpec,
    MdpToken, ObsCtrlChannels, ObservabilityEvaluatorStation, PartiallyObservableProcess,
    PomdpInformationEvaluatorStation, PomdpObservabilityEvaluatorStation, PomdpSourceStation,
    PomdpSpec, PomdpToken, StateSpaceModel, StateSpaceSourceStation, StateSpaceSpec,
    StateSpaceToken,
};
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::station::{DESStation, StationRef};

struct ObsCtrlDemo;

impl ObsCtrlDemo {
    fn run(&self) {
        let lti_tokens = self.build_lti_models();
        let mdp_tokens = self.build_mdp_models();
        let pomdp_tokens = self.build_pomdp_models();

        let lti_source = Rc::new(RefCell::new(StateSpaceSourceStation::new(
            "lti-source",
            lti_tokens.clone(),
        )));
        let mdp_source = Rc::new(RefCell::new(MdpSourceStation::new(
            "mdp-source",
            mdp_tokens.clone(),
        )));
        let pomdp_source = Rc::new(RefCell::new(PomdpSourceStation::new(
            "pomdp-source",
            pomdp_tokens.clone(),
        )));

        let ctrl_eval = Rc::new(RefCell::new(ControllabilityEvaluatorStation::new(
            "ctrl-eval",
        )));
        let obs_eval = Rc::new(RefCell::new(ObservabilityEvaluatorStation::new("obs-eval")));
        let mdp_eval = Rc::new(RefCell::new(MdpControllabilityEvaluatorStation::new(
            "mdp-eval",
        )));
        let mdp_info = Rc::new(RefCell::new(MdpInformationEvaluatorStation::new(
            "mdp-info",
        )));
        let pomdp_eval = Rc::new(RefCell::new(PomdpObservabilityEvaluatorStation::new(
            "pomdp-eval",
        )));
        let pomdp_info = Rc::new(RefCell::new(PomdpInformationEvaluatorStation::new(
            "pomdp-info",
        )));
        let sink = Rc::new(RefCell::new(EvaluationSinkStation::new("sink")));

        let lti_src_ref: StationRef = lti_source.clone();
        let mdp_src_ref: StationRef = mdp_source.clone();
        let pomdp_src_ref: StationRef = pomdp_source.clone();
        let ctrl_ref: StationRef = ctrl_eval;
        let obs_ref: StationRef = obs_eval;
        let mdp_eval_ref: StationRef = mdp_eval;
        let mdp_info_ref: StationRef = mdp_info;
        let pomdp_eval_ref: StationRef = pomdp_eval;
        let pomdp_info_ref: StationRef = pomdp_info;
        let sink_ref: StationRef = sink.clone();

        lti_source.borrow_mut().core_mut().pipe(
            ctrl_ref.clone(),
            ObsCtrlChannels::MODEL_LTI,
            ObsCtrlChannels::MODEL_LTI,
        );
        lti_source.borrow_mut().core_mut().pipe(
            obs_ref.clone(),
            ObsCtrlChannels::MODEL_LTI,
            ObsCtrlChannels::MODEL_LTI,
        );
        mdp_source.borrow_mut().core_mut().pipe(
            mdp_eval_ref.clone(),
            ObsCtrlChannels::MODEL_MDP,
            ObsCtrlChannels::MODEL_MDP,
        );
        mdp_source.borrow_mut().core_mut().pipe(
            mdp_info_ref.clone(),
            ObsCtrlChannels::MODEL_MDP,
            ObsCtrlChannels::MODEL_MDP,
        );
        pomdp_source.borrow_mut().core_mut().pipe(
            pomdp_eval_ref.clone(),
            ObsCtrlChannels::MODEL_POMDP,
            ObsCtrlChannels::MODEL_POMDP,
        );
        pomdp_source.borrow_mut().core_mut().pipe(
            pomdp_info_ref.clone(),
            ObsCtrlChannels::MODEL_POMDP,
            ObsCtrlChannels::MODEL_POMDP,
        );
        for ev in [
            &ctrl_ref,
            &obs_ref,
            &mdp_eval_ref,
            &mdp_info_ref,
            &pomdp_eval_ref,
            &pomdp_info_ref,
        ] {
            ev.borrow_mut().core_mut().pipe(
                sink_ref.clone(),
                ObsCtrlChannels::RESULT,
                ObsCtrlChannels::RESULT,
            );
        }

        run_iterative_des(
            vec![
                lti_src_ref,
                mdp_src_ref,
                pomdp_src_ref,
                ctrl_ref,
                obs_ref,
                mdp_eval_ref,
                mdp_info_ref,
                pomdp_eval_ref,
                pomdp_info_ref,
                sink_ref,
            ],
            IterativeRunOptions {
                shuffle: false,
                max_ticks: Some(10),
                ..Default::default()
            },
        );

        self.report(&sink.borrow(), &lti_tokens, &mdp_tokens, &pomdp_tokens);
    }

    fn build_lti_models(&self) -> Vec<StateSpaceToken> {
        let mut tokens: Vec<StateSpaceToken> = Vec::new();
        // The query's worked example: double integrator. Both rank 2.
        tokens.push(StateSpaceToken::new(
            "double-integrator (query example)".to_string(),
            StateSpaceModel::new(StateSpaceSpec {
                a: vec![vec![0.0, 1.0], vec![0.0, 0.0]],
                b: vec![vec![0.0], vec![1.0]],
                c: vec![vec![1.0, 0.0]],
                d: None,
            }),
        ));
        // Diagonal plant, input only reaches x1 → uncontrollable; output only sees
        // x1 → unobservable. A clean "neither" example.
        tokens.push(StateSpaceToken::new(
            "decoupled modes (B,C reach one mode)".to_string(),
            StateSpaceModel::new(StateSpaceSpec {
                a: vec![vec![1.0, 0.0], vec![0.0, 2.0]],
                b: vec![vec![1.0], vec![0.0]],
                c: vec![vec![1.0, 0.0]],
                d: None,
            }),
        ));
        // Controllable but NOT observable: both modes driven, output sees only x1.
        tokens.push(StateSpaceToken::new(
            "controllable, not observable".to_string(),
            StateSpaceModel::new(StateSpaceSpec {
                a: vec![vec![1.0, 0.0], vec![0.0, 2.0]],
                b: vec![vec![1.0], vec![1.0]],
                c: vec![vec![1.0, 0.0]],
                d: None,
            }),
        ));
        // The DC motor (R,L,Ke,Kt,J,B) — physically controllable & observable.
        let motor = DcMotorDynamics::new(DcMotorParams {
            resistance: 2.0,
            inductance: 0.5,
            back_emf_constant: 0.1,
            torque_constant: 0.1,
            inertia: 0.02,
            friction: 0.002,
        })
        .state_space();
        tokens.push(StateSpaceToken::new(
            "DC motor (V → ω)".to_string(),
            StateSpaceModel::new(StateSpaceSpec {
                a: motor.a,
                b: motor.b,
                c: motor.c,
                d: Some(motor.d),
            }),
        ));
        tokens
    }

    fn build_mdp_models(&self) -> Vec<MdpToken> {
        // 3-state controllable ring: action 0 advances s→s+1 (mod 3).
        let ring = MarkovDecisionProcess::new(MdpSpec {
            num_states: 3,
            num_actions: 1,
            transition: vec![vec![
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![1.0, 0.0, 0.0],
            ]],
        });
        // 3-state with an absorbing trap (state 2): not strongly connected.
        let trap = MarkovDecisionProcess::new(MdpSpec {
            num_states: 3,
            num_actions: 1,
            transition: vec![vec![
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![0.0, 0.0, 1.0],
            ]],
        });
        vec![
            MdpToken::new("ring MDP (strongly connected)".to_string(), ring),
            MdpToken::new("trap MDP (absorbing state 2)".to_string(), trap),
        ]
    }

    fn build_pomdp_models(&self) -> Vec<PomdpToken> {
        // Observable: distinct observations per state (identity-like sensor).
        let distinct = PartiallyObservableProcess::new(PomdpSpec {
            num_states: 2,
            num_actions: 1,
            transition: vec![vec![vec![0.5, 0.5], vec![0.5, 0.5]]],
            num_observations: 2,
            observation: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        });
        // Aliased: same observation distribution AND stay put → never distinguishable.
        let aliased = PartiallyObservableProcess::new(PomdpSpec {
            num_states: 2,
            num_actions: 1,
            transition: vec![vec![vec![1.0, 0.0], vec![0.0, 1.0]]],
            num_observations: 2,
            observation: vec![vec![0.5, 0.5], vec![0.5, 0.5]],
        });
        vec![
            PomdpToken::new("distinct-sensor POMDP".to_string(), distinct),
            PomdpToken::new("aliased-sensor POMDP".to_string(), aliased),
        ]
    }

    fn report(
        &self,
        sink: &EvaluationSinkStation,
        lti: &[StateSpaceToken],
        mdp: &[MdpToken],
        pomdp: &[PomdpToken],
    ) {
        println!();
        println!(
            "================================================================================"
        );
        println!(" Observability & Controllability — general structural evaluator");
        println!(
            "================================================================================"
        );

        println!();
        println!(" LINEAR STATE-SPACE  (Kalman rank tests)");
        println!(
            " --------------------------------------------------------------------------------"
        );
        for t in lti {
            let rs = sink.for_label(&t.label);
            let ctrl = rs
                .iter()
                .find(|r| r.kind == EvaluationKind::Controllability)
                .expect("controllability verdict");
            let obs = rs
                .iter()
                .find(|r| r.kind == EvaluationKind::Observability)
                .expect("observability verdict");
            println!("   {}", t.label);
            println!(
                "      controllable : {}   ({})",
                self.verdict(ctrl.full),
                ctrl.detail
            );
            println!(
                "      observable   : {}   ({})",
                self.verdict(obs.full),
                obs.detail
            );
        }

        println!();
        println!(" MDP  (reachability ≈ controllability)");
        println!(
            " --------------------------------------------------------------------------------"
        );
        for t in mdp {
            let rs = sink.for_label(&t.label);
            let r = rs
                .iter()
                .find(|x| x.kind == EvaluationKind::MdpControllability)
                .expect("mdp-controllability verdict");
            let info = rs
                .iter()
                .find(|x| x.kind == EvaluationKind::MdpTransitionEntropy)
                .expect("mdp-transition entropy verdict");
            println!("   {}", t.label);
            println!(
                "      controllable : {}   ({})",
                self.verdict(r.full),
                r.detail
            );
            println!("      transition H :       ({})", info.detail);
        }

        println!();
        println!(" POMDP  (distinguishability ≈ observability)");
        println!(
            " --------------------------------------------------------------------------------"
        );
        for t in pomdp {
            let rs = sink.for_label(&t.label);
            let r = rs
                .iter()
                .find(|x| x.kind == EvaluationKind::PomdpObservability)
                .expect("pomdp-observability verdict");
            let info = rs
                .iter()
                .find(|x| x.kind == EvaluationKind::PomdpObservationInformation)
                .expect("pomdp-observation information verdict");
            let aliasing = t.pomdp.indistinguishable_pairs();
            println!("   {}", t.label);
            println!(
                "      observable   : {}   ({})",
                self.verdict(r.full),
                r.detail
            );
            println!("      sensor info  :       ({})", info.detail);
            if !aliasing.is_empty() {
                let pairs: Vec<String> = aliasing
                    .iter()
                    .map(|p| format!("({},{})", p.0, p.1))
                    .collect();
                println!("      aliased state pairs: {}", pairs.join(" "));
            }
        }
        println!();
        println!(
            "================================================================================"
        );
        println!();
    }

    fn verdict(&self, ok: bool) -> &'static str {
        if ok {
            "YES"
        } else {
            "NO "
        }
    }
}

/// Entry point (`main()` in the TS source).
pub fn run() {
    ObsCtrlDemo.run();
}
