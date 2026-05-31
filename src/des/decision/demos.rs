//! Canonical decision-process demos: a control MDP (machine maintenance) and a
//! belief POMDP (the classic tiger). Each returns a canonical serde spec, so the
//! same objects exercise validation, solving, rollout, viz, and the citizen
//! contract.

use super::spec::{MdpSpec, MdpTransition, PomdpSpec, MDP_SCHEMA, POMDP_SCHEMA};

/// Machine-maintenance MDP: condition degrades while operating; repairing
/// restores it at a cost. The optimal policy operates while healthy and repairs
/// once degraded — a compact "predict degradation + control maintenance" model.
///
/// States 0..=4 = condition (new → broken). Actions: 0 = operate, 1 = repair.
pub fn machine_maintenance_mdp() -> MdpSpec {
    let revenue = [10.0, 8.0, 6.0, 3.0, -5.0];
    let repair_cost = 4.0;
    let degrade_p = 0.3;
    let n = revenue.len();

    let mut transitions = Vec::with_capacity(n);
    for c in 0..n {
        // Operate.
        let operate = if c + 1 < n {
            vec![
                MdpTransition { prob: 1.0 - degrade_p, reward: revenue[c], next: c },
                MdpTransition { prob: degrade_p, reward: revenue[c], next: c + 1 },
            ]
        } else {
            // Broken: operating loses money and stays broken.
            vec![MdpTransition { prob: 1.0, reward: revenue[c], next: c }]
        };
        // Repair: back to new (condition 0) at a cost.
        let repair = vec![MdpTransition { prob: 1.0, reward: -repair_cost, next: 0 }];
        transitions.push(vec![operate, repair]);
    }

    MdpSpec {
        schema: MDP_SCHEMA.to_string(),
        num_states: n,
        transitions,
        discount: 0.9,
        terminal: vec![],
        state_labels: vec![
            "new".into(),
            "good".into(),
            "fair".into(),
            "poor".into(),
            "broken".into(),
        ],
        action_labels: vec!["operate".into(), "repair".into()],
    }
}

/// The classic tiger POMDP: two doors, a tiger behind one. `listen` gives a
/// noisy (85% correct) hint; opening the correct door yields +10, the wrong one
/// −100, and resets the world. Demonstrates value-of-information: a good policy
/// listens until confident, then opens.
pub fn tiger_pomdp() -> PomdpSpec {
    PomdpSpec {
        schema: POMDP_SCHEMA.to_string(),
        num_states: 2,
        num_actions: 3,
        num_observations: 2,
        // listen: stay; open-*: reset to 50/50.
        transition: vec![
            vec![vec![1.0, 0.0], vec![0.5, 0.5], vec![0.5, 0.5]],
            vec![vec![0.0, 1.0], vec![0.5, 0.5], vec![0.5, 0.5]],
        ],
        // listen: 85% correct; open-*: uninformative.
        observation: vec![
            vec![vec![0.85, 0.15], vec![0.5, 0.5], vec![0.5, 0.5]],
            vec![vec![0.15, 0.85], vec![0.5, 0.5], vec![0.5, 0.5]],
        ],
        reward: vec![vec![-1.0, -100.0, 10.0], vec![-1.0, 10.0, -100.0]],
        discount: 0.95,
        initial_belief: None,
        state_labels: vec!["tiger-left".into(), "tiger-right".into()],
        action_labels: vec!["listen".into(), "open-left".into(), "open-right".into()],
        observation_labels: vec!["hear-left".into(), "hear-right".into()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::decision::solve::{solve_mdp, solve_pomdp, MdpMethod, PomdpMethod};
    use crate::des::decision::{rollout_mdp, rollout_pomdp};
    use crate::des::general::belief::DiscreteBelief;
    use crate::des::general::pomdp::belief_update;

    #[test]
    fn maintenance_policy_operates_when_healthy_and_repairs_when_broken() {
        let spec = machine_maintenance_mdp();
        assert!(spec.validate().is_ok());
        let sol = solve_mdp(&spec, MdpMethod::ValueIteration).unwrap();
        assert_eq!(sol.policy[0], 0, "should operate when new");
        assert_eq!(sol.policy[4], 1, "should repair when broken");
        // Value is non-increasing as the condition worsens, and the best state is
        // strictly more valuable than the broken one. (Once "repair" is optimal,
        // the worse conditions share a value, since repair resets to condition 0.)
        for c in 0..4 {
            assert!(
                sol.value[c] >= sol.value[c + 1] - 1e-9,
                "V[{c}]={} should be >= V[{}]={}",
                sol.value[c],
                c + 1,
                sol.value[c + 1]
            );
        }
        assert!(
            sol.value[0] > sol.value[4] + 1.0,
            "new should be worth strictly more than broken: V[0]={}, V[4]={}",
            sol.value[0],
            sol.value[4]
        );
    }

    #[test]
    fn maintenance_rollout_is_finite_and_consistent() {
        let spec = machine_maintenance_mdp();
        let sol = solve_mdp(&spec, MdpMethod::ValueIteration).unwrap();
        let trace = rollout_mdp(&spec, &sol.policy, 0, 30, 7);
        assert_eq!(trace.states.len(), trace.actions.len() + 1);
        assert_eq!(trace.rewards.len(), trace.actions.len());
        assert!(trace.discounted_return.is_finite());
    }

    #[test]
    fn tiger_qmdp_listens_when_uncertain_and_opens_when_confident() {
        let spec = tiger_pomdp();
        let mut plan = solve_pomdp(&spec, PomdpMethod::Qmdp, 3).unwrap();

        // Uniform belief → listen (action 0): opening is too risky.
        let uniform = DiscreteBelief::new(vec![0usize, 1], Some(&[0.5, 0.5]));
        assert_eq!(plan.act(&uniform), 0, "should listen when uncertain");

        // Confident the tiger is on the left → open the right (safe) door (action 2).
        let confident_left = DiscreteBelief::new(vec![0usize, 1], Some(&[0.95, 0.05]));
        assert_eq!(plan.act(&confident_left), 2, "should open the safe door");
    }

    #[test]
    fn tiger_belief_sharpens_after_consistent_observations() {
        let spec = tiger_pomdp();
        let closure = spec.to_pomdp_spec();
        let b0 = DiscreteBelief::new(vec![0usize, 1], Some(&[0.5, 0.5]));
        // Listen (action 0), hear-left (obs 0) twice.
        let b1 = belief_update(&closure, &b0, 0, 0);
        let b2 = belief_update(&closure, &b1, 0, 0);
        assert!(b1.weights[0] > 0.5, "belief should shift toward tiger-left");
        assert!(b2.weights[0] > b1.weights[0], "belief should sharpen further");
        assert!(b2.entropy() < b0.entropy(), "entropy should drop");
    }

    #[test]
    fn tiger_rollout_records_beliefs_and_observations() {
        let spec = tiger_pomdp();
        let mut plan = solve_pomdp(&spec, PomdpMethod::Lookahead, 3).unwrap();
        let trace = rollout_pomdp(&spec, &mut plan, Some(0), 12, 3);
        assert_eq!(trace.beliefs.len(), trace.states.len());
        assert_eq!(trace.observations.len(), trace.actions.len());
        assert!(trace.discounted_return.is_finite());
    }
}
