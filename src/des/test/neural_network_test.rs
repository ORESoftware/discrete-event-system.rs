//! Port of src/des/test/neural-network-test.ts
//!
//! Neural-net support across the hybrid boundary: supervised DES training (XOR),
//! neural Q-learning for MDPs, and neural ODE solves. All three are exercised by
//! their direct entry points.
//!
//! PORT NOTE: the two "station queue semantics" groups (the `NeuralNetworkStation`
//! inference pipe and the `NeuralODESolverStation` solution pipe, the latter of
//! which subclasses `NeuralPredictionSink`) drive a full `run_iterative_des`
//! pipeline and a TS subclass override; those are deferred. The numerical
//! behaviour they assert is covered by the direct `solve_neural_ode` group.
#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::des_base::environment::{PureEnvironment, StepResult};
    use crate::des::general::neural_network::{
        run_neural_q_learning_des, run_xor_neural_net_des, solve_neural_ode, ActivationName,
        DenseLayerConfig, FeedForwardNetwork, NeuralODEOptions, NeuralODESolverName,
        NeuralQLearningRunParams, XorNeuralNetOptions,
    };
    use crate::des::general::rl_environments::{
        eval_policy, Corridor, Environment, EvalPolicyOptions,
    };
    use crate::des::shared::capabilities::SeededRandom;

    // Bridges the pure `Environment` trait to the `PureEnvironment<f64, usize>`
    // the neural Q-learning runner requires (state indices carried as `f64`).
    struct PureEnvAdapter<E: Environment> {
        env: E,
    }

    impl<E: Environment> PureEnvironment<f64, usize> for PureEnvAdapter<E> {
        fn num_states(&self) -> usize {
            self.env.num_states()
        }
        fn num_actions(&self) -> usize {
            self.env.num_actions()
        }
        fn reset(&mut self) -> f64 {
            self.env.reset() as f64
        }
        fn step(&mut self, state: f64, action: usize) -> StepResult<f64> {
            let o = self.env.step(state as usize, action);
            StepResult {
                next_state: o.next_state as f64,
                reward: o.reward,
                done: o.done,
            }
        }
    }

    fn mean(xs: &[f64]) -> f64 {
        xs.iter().sum::<f64>() / (xs.len().max(1) as f64)
    }

    // Feed-forward network + supervised DES training (XOR).
    #[test]
    fn xor_supervised_training() {
        let r = run_xor_neural_net_des(XorNeuralNetOptions {
            seed: Some(7),
            epochs: Some(8000),
            learning_rate: Some(0.3),
            hidden_layers: Some(vec![4]),
            ..Default::default()
        });
        let n = r.loss_history.len();
        let first = mean(&r.loss_history[0..100]);
        let last = mean(&r.loss_history[n - 100..]);
        let y: Vec<f64> = r.predictions.iter().map(|v| v[0]).collect();
        let classified = y[0] < 0.2 && y[1] > 0.8 && y[2] > 0.8 && y[3] < 0.2;
        assert!(last < first / 10.0, "first={first} last={last}");
        assert!(classified, "predictions={y:?}");
        assert!(r.network.num_parameters() > 0);
    }

    // Neural Q-learning over an MDP (Corridor).
    #[test]
    fn neural_q_learning() {
        let r = run_neural_q_learning_des(
            Box::new(PureEnvAdapter {
                env: Corridor::new(6, 0),
            }),
            NeuralQLearningRunParams {
                num_episodes: 600,
                alpha: 0.25,
                gamma: 0.95,
                epsilon: 0.8,
                epsilon_min: Some(0.02),
                epsilon_decay: Some(0.99),
                max_steps_per_episode: Some(40),
                seed: Some(1),
                network: None,
                hidden_layers: None,
                hidden_activation: None,
                state_encoder: None,
            },
        );

        let eval_env = Corridor::new(6, 0);
        let mut rng = SeededRandom::new(1);
        let e = eval_policy(
            &eval_env,
            |s, _rng| r.policy[s],
            &mut rng,
            EvalPolicyOptions {
                num_episodes: 50,
                max_steps_per_episode: 40,
                gamma: 1.0,
            },
        );

        assert_eq!(r.total_episodes, 600);
        assert_eq!(e.success_rate, 1.0);
        assert_eq!(r.policy[0], 1);
        assert_eq!(r.loss_history.len() as u64, r.total_steps);
    }

    // Neural ODE vector field: dy/dt = -rate * y, RK4 tracks exponential decay.
    #[test]
    fn neural_ode_decay() {
        let rate = 0.5;
        let net = FeedForwardNetwork::new(vec![DenseLayerConfig {
            weights: vec![vec![-rate]],
            biases: vec![0.0],
            activation: ActivationName::Linear,
        }]);
        let trace = solve_neural_ode(
            &net,
            &NeuralODEOptions {
                y0: vec![1.0],
                t0: 0.0,
                t1: 2.0,
                dt: 0.05,
                solver: Some(NeuralODESolverName::Rk4),
                include_time: None,
                rk45: None,
            },
        );
        let final_y = trace.y[trace.y.len() - 1][0];
        let exact = (-rate * 2.0).exp();
        assert!(
            (final_y - exact).abs() < 1e-7,
            "final={final_y} exact={exact}"
        );
    }
}
