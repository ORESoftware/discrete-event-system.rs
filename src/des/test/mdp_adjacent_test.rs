//! Port of src/des/test/mdp-adjacent-test.ts
//!
//! End-to-end tests for the nine MDP-adjacent models:
//!   1. inventory-dp            — finite-horizon dynamic programming
//!   2. mountain-car-vfa        — approximate dynamic programming (linear VFA)
//!   3. tiger-pomdp             — POMDP belief-state planning
//!   4. grid-localization-pomdp — multi-dimensional POMDP belief lookahead
//!   5. four-rooms-smdp         — Semi-MDP / options framework
//!   6. actor-critic-grid       — Actor-Critic on tabular GridWorld
//!   7. blackjack-mc            — Monte Carlo on-policy control
//!   8. stag-hunt               — multi-agent IQL on a coordination game
//!   9. double-integrator-lqr   — LQR / stochastic control via Riccati DARE
//!
//! The thresholds are intentionally conservative so they pass deterministically
//! while still verifying the core invariant each algorithm should satisfy.

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::actor_critic_gridworld::{
        run_actor_critic_gridworld, ActorCriticTrainOpts,
    };
    use crate::des::general::blackjack::{run_blackjack_mc, BlackjackTrainOpts};
    use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
    use crate::des::general::des_base::station::StationRef;
    use crate::des::general::double_integrator_lqr::{
        run_double_integrator_lqr, DoubleIntegratorOpts,
    };
    use crate::des::general::four_rooms::{run_four_rooms_smdp, FourRoomsTrainOpts};
    use crate::des::general::grid_localization_pomdp::{
        run_grid_localization_pomdp, GridLocalizationActionKind, GridLocalizationParams,
    };
    use crate::des::general::inventory_dp::{
        simulate_inventory, solve_inventory_dp, InventoryDPStation, InventoryProblem,
    };
    use crate::des::general::mountain_car::{run_mountain_car, MountainCarTrainOpts};
    use crate::des::general::stag_hunt::{run_stag_hunt, StagHuntOpts};
    use crate::des::general::tiger_pomdp::{
        build_tiger_spec, simulate_tiger, TigerOpts, TigerSimOpts, TigerSolver, ACT_LISTEN,
    };
    use std::cell::RefCell;
    use std::rc::Rc;

    // -------------------------------------------------------------------------
    // 1. INVENTORY-DP (finite-horizon DP / Bellman backward induction)
    // -------------------------------------------------------------------------

    fn inventory_problem() -> InventoryProblem {
        // 5-period inventory with truncated Poisson(λ=3) demand.
        let pmf = [
            0.0498, 0.1494, 0.2240, 0.2240, 0.1680, 0.1008, 0.0504, 0.0336,
        ];
        let norm: f64 = pmf.iter().sum();
        let demand_pmf: Vec<f64> = pmf.iter().map(|x| x / norm).collect();
        InventoryProblem {
            horizon: 5,
            s_max: 10,
            demand_pmf,
            price: 8.0,
            cost: 3.0,
            fixed_cost: 4.0,
            hold_cost: 1.0,
            stockout_cost: 5.0,
            salvage_value: 1.0,
            discount: Some(1.0),
            initial_inventory: 0,
        }
    }

    #[test]
    fn inventory_dp_backward_sweep_ticks_equal_horizon() {
        let p = inventory_problem();
        let r = solve_inventory_dp(&p, Some(7));
        assert_eq!(r.ticks, p.horizon);
    }

    #[test]
    fn inventory_dp_value_non_decreasing_in_stock() {
        let p = inventory_problem();
        let r = solve_inventory_dp(&p, Some(7));
        let v0 = &r.v[0];
        for s in 1..v0.len() {
            assert!(v0[s] >= v0[s - 1] - 1e-6, "V(t=0) must be non-decreasing in s");
        }
    }

    #[test]
    fn inventory_dp_policy_entries_feasible() {
        let p = inventory_problem();
        let r = solve_inventory_dp(&p, Some(7));
        for t in 0..p.horizon {
            for s in 0..r.policy[t].len() {
                let a = r.policy[t][s];
                // a is usize so a >= 0 trivially; check a + s <= S_max.
                assert!(a + s <= p.s_max, "infeasible action at (t={}, s={})", t, s);
            }
        }
    }

    #[test]
    fn inventory_dp_mc_estimate_matches_value() {
        let p = inventory_problem();
        let r = solve_inventory_dp(&p, Some(7));
        let reps = 200usize;
        let mut mc_sum = 0.0;
        for rep in 0..reps {
            let sim = simulate_inventory(&p, &r.policy, (rep + 100) as u32);
            mc_sum += sim.total_reward;
        }
        let mc = mc_sum / reps as f64;
        assert!(
            (mc - r.expected_reward).abs() < 2.0,
            "MC={} vs V*={}",
            mc,
            r.expected_reward
        );
    }

    #[test]
    fn inventory_dp_intrinsic_validators_pass() {
        let p = inventory_problem();
        let station = InventoryDPStation::new(p);
        let summary = run_iterative_des(
            vec![Rc::new(RefCell::new(station)) as StationRef],
            IterativeRunOptions::default(),
        );
        assert!(summary.validation.is_some(), "validators should be attached");
        assert_eq!(summary.validation_ok, Some(true), "all invariants must pass");
    }

    // -------------------------------------------------------------------------
    // 2. MOUNTAIN-CAR-VFA (approximate DP, linear VFA + tile coding)
    // -------------------------------------------------------------------------

    #[test]
    fn mountain_car_vfa_learns() {
        let r = run_mountain_car(MountainCarTrainOpts {
            num_episodes: 80,
            alpha: Some(0.5),
            gamma: Some(1.0),
            epsilon: Some(0.0),
            epsilon_decay: Some(1.0),
            epsilon_min: Some(0.0),
            num_tilings: Some(8),
            num_tiles_per_dim: Some(8),
            max_steps_per_episode: Some(1000),
            seed: Some(1),
        });

        assert_eq!(r.reward_history.len(), 80, "rewardHistory length == numEpisodes");

        // First 5 vs last 20 episode lengths — should improve substantially.
        let first5: f64 = r.length_history[0..5].iter().sum::<f64>() / 5.0;
        let last_slice = &r.length_history[r.length_history.len() - 20..];
        let last20: f64 = last_slice.iter().sum::<f64>() / 20.0;
        assert!(first5 > last20, "episode length should decrease: {first5} -> {last20}");

        // All returns negative (per-step -1, no goal yet).
        assert!(
            r.reward_history.iter().all(|&x| x <= 0.0),
            "all returns should be <= 0"
        );

        // theta is non-trivial after 80 episodes.
        assert!(r.theta_norm > 0.0, "||theta|| should be > 0 after training");
    }

    // -------------------------------------------------------------------------
    // 3. TIGER-POMDP (POMDP belief-state planning)
    // -------------------------------------------------------------------------

    fn tiger_one_step(seed: u32) -> crate::des::general::tiger_pomdp::TigerSimResult {
        simulate_tiger(TigerSimOpts {
            spec: Some(build_tiger_spec(&TigerOpts::default())),
            solver: TigerSolver::OneStepLookahead,
            num_steps: 50,
            seed: Some(seed),
            initial_state: None,
            initial_belief: None,
        })
    }

    #[test]
    fn tiger_pomdp_one_step_lookahead_mostly_listens() {
        let r1 = tiger_one_step(1);
        let listens = r1.actions.iter().filter(|&&a| a == ACT_LISTEN).count();
        let listen_frac = listens as f64 / r1.actions.len() as f64;
        assert!(listen_frac > 0.5, "listen fraction = {listen_frac}");
    }

    #[test]
    fn tiger_pomdp_avoids_most_catastrophic_opens() {
        let r1 = tiger_one_step(1);
        assert!(
            r1.num_bad_opens <= 5,
            "bad opens = {} / {} opens / {} steps",
            r1.num_bad_opens,
            r1.num_opens,
            r1.steps
        );
    }

    #[test]
    fn tiger_pomdp_average_return_is_finite() {
        let mut avg = 0.0;
        for s in 0..10 {
            avg += tiger_one_step(s + 1).total_return;
        }
        avg /= 10.0;
        assert!(avg.is_finite(), "avg discounted return should be finite: {avg}");
    }

    // -------------------------------------------------------------------------
    // 4. GRID-LOCALIZATION-POMDP (2D hidden-state POMDP)
    // -------------------------------------------------------------------------

    #[test]
    fn grid_localization_pomdp_localizes() {
        let params = GridLocalizationParams {
            width: 3,
            height: 3,
            horizon: Some(3),
            num_steps: Some(8),
            seed: Some(7),
            hidden_target: Some((2, 1)),
            initial_belief: None,
            scan_accuracy: Some(1.0),
            inspect_accuracy: Some(1.0),
            scan_cost: None,
            inspect_correct_reward: None,
            inspect_wrong_penalty: None,
            discount: None,
        };
        let r = run_grid_localization_pomdp(&params);

        let first = &r.trace[0];
        let last = &r.trace[r.trace.len() - 1];

        // 2D POMDP state space is Cartesian 3x3.
        assert_eq!(r.state_space.num_states, 9);
        assert_eq!(r.state_space.dimensions.len(), 2);

        // Belief-lookahead gathers information before inspecting.
        assert!(
            first.action.kind != GridLocalizationActionKind::Inspect,
            "first action should not be inspect: {}",
            first.action.label
        );

        // Perfect row/column scans reduce entropy below the uniform max.
        assert!(last.entropy < (9.0_f64).ln(), "Hf={}", last.entropy);

        // Posterior concentrates on the hidden target.
        assert!(
            last.hidden_probability > 0.95,
            "P(hidden)={}",
            last.hidden_probability
        );

        // Planner finds the hidden target.
        assert!(r.found, "planner should find the hidden target");
    }

    // -------------------------------------------------------------------------
    // 5. FOUR-ROOMS-SMDP (Semi-MDP, options framework)
    // -------------------------------------------------------------------------

    #[test]
    fn four_rooms_smdp_solves() {
        let r = run_four_rooms_smdp(FourRoomsTrainOpts {
            num_episodes: 800,
            alpha: Some(0.3),
            gamma: Some(0.99),
            epsilon: Some(0.2),
            epsilon_decay: Some(0.99),
            epsilon_min: Some(0.02),
            max_steps_per_episode: Some(2000),
            slip: Some(0.0),
            include_primitive: Some(true),
            init_q: Some(0.05),
            seed: Some(1),
        });

        assert_eq!(r.reward_history.len(), 800);
        assert!(r.greedy_reached_goal, "greedy policy should reach the goal");
        assert!(
            r.greedy_episode_length <= 200.0,
            "greedy episode length should be <= 200: {}",
            r.greedy_episode_length
        );
    }

    // -------------------------------------------------------------------------
    // 6. ACTOR-CRITIC-GRID (Actor-Critic on GridWorld)
    // -------------------------------------------------------------------------

    #[test]
    fn actor_critic_gridworld_solves() {
        let r = run_actor_critic_gridworld(ActorCriticTrainOpts {
            num_episodes: 1500,
            alpha_v: Some(0.1),
            alpha_p: Some(0.1),
            gamma: Some(0.95),
            entropy_coef: None,
            max_steps_per_episode: Some(100),
            width: Some(4),
            height: Some(4),
            seed: Some(1),
        });

        assert_eq!(r.reward_history.len(), 1500);

        // Last 50 returns positive on average (goal +10 dominates step -1).
        let last = &r.reward_history[r.reward_history.len() - 50..];
        let mean_last: f64 = last.iter().sum::<f64>() / last.len() as f64;
        assert!(mean_last > 0.0, "mean return (last 50) should be > 0: {mean_last}");

        assert!(r.greedy_reached, "greedy policy should reach the goal");
    }

    // -------------------------------------------------------------------------
    // 7. BLACKJACK-MC (first-visit Monte Carlo control)
    // -------------------------------------------------------------------------

    #[test]
    fn blackjack_mc_control() {
        let r = run_blackjack_mc(BlackjackTrainOpts {
            num_episodes: 50_000,
            epsilon: Some(0.1),
            epsilon_decay: Some(1.0),
            epsilon_min: Some(0.05),
            first_visit: Some(true),
            gamma: Some(1.0),
            eval_episodes: Some(3000),
            seed: Some(1),
        });

        assert!(
            r.greedy_mean_return > r.baseline_mean_return,
            "greedy={} should beat baseline={}",
            r.greedy_mean_return,
            r.baseline_mean_return
        );
        assert!(
            r.baseline_mean_return >= -0.40 && r.baseline_mean_return <= -0.20,
            "baseline in canonical band [-0.40,-0.20]: {}",
            r.baseline_mean_return
        );
        assert!(
            r.greedy_mean_return >= -0.10,
            "greedy in canonical band (>= -0.10): {}",
            r.greedy_mean_return
        );
        assert!(r.visited_cells > 200, "visited {} / 400 cells", r.visited_cells);
    }

    // -------------------------------------------------------------------------
    // 8. STAG-HUNT (independent Q-learning, 2 agents)
    // -------------------------------------------------------------------------

    #[test]
    fn stag_hunt_coordinates() {
        let r = run_stag_hunt(&StagHuntOpts {
            num_episodes: 5000,
            alpha: Some(0.05),
            gamma: Some(0.0),
            epsilon: Some(0.2),
            epsilon_decay: Some(0.999),
            epsilon_min: Some(0.01),
            seed: Some(1),
        });

        assert_eq!(r.reward_history.len(), 5000);

        // Coordination: end up at one of the two pure NE.
        assert!(
            r.coordinated_on_stag || r.coordinated_on_hare,
            "agents should coordinate on a Nash equilibrium: final=[{}, {}]",
            r.final_joint_action[0],
            r.final_joint_action[1]
        );

        // Recent mean returns: worst NE (Hare,Hare)=3,3; best (Stag,Stag)=4,4.
        assert!(
            r.recent_mean_return[0] >= 2.5 && r.recent_mean_return[1] >= 2.5,
            "recent mean returns should be >= 2.5: [{}, {}]",
            r.recent_mean_return[0],
            r.recent_mean_return[1]
        );
    }

    // -------------------------------------------------------------------------
    // 9. DOUBLE-INTEGRATOR-LQR (Riccati DARE)
    // -------------------------------------------------------------------------

    #[test]
    fn double_integrator_lqr_optimal() {
        let r = run_double_integrator_lqr(DoubleIntegratorOpts {
            dt: Some(0.1),
            q_pos: Some(1.0),
            q_vel: Some(0.1),
            r_u: Some(0.01),
            noise_std: Some(0.0), // deterministic for theory-vs-realised match
            x0: Some([3.0, 0.0]),
            num_steps: Some(200),
            u_sat: Some(100.0),
            gamma: Some(1.0),
            seed: Some(1),
        })
        .expect("valid LQR parameters");

        // Riccati iteration converged.
        assert!(r.riccati_residual < 1e-8, "residual={}", r.riccati_residual);

        // K is 1×2 (m × n).
        assert_eq!(r.k.len(), 1);
        assert_eq!(r.k[0].len(), 2);

        // Both gains positive (point mass with positive Q).
        assert!(r.k[0][0] > 0.0 && r.k[0][1] > 0.0, "K entries should be positive");

        // Trajectory drives state → 0.
        let last = r.trajectory.last().expect("non-empty trajectory");
        let final_norm = (last[0] * last[0] + last[1] * last[1]).sqrt();
        assert!(final_norm < 0.05, "|x(T)|={final_norm}");

        // Realised cost ≤ DARE cost-to-go (LQR optimality); small FP slack.
        assert!(
            r.total_cost <= r.riccati_cost_from_x0 * (1.0 + 1e-3) + 1e-3,
            "realised={} vs DARE={}",
            r.total_cost,
            r.riccati_cost_from_x0
        );
    }
}
