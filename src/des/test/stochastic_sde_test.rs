//! Port of src/des/test/stochastic-sde-test.ts
//!
//! Tests stochastic-sde and sde-learning: Euler–Maruyama moments, OU stationary
//! statistics, maximum-likelihood SDE recovery, an MLP regressor, a denoising
//! diffusion model, the deterministic motor limit, and the EnKF.
//!
//! PORT NOTE: group [4] (the full SdePlantStation → EnsembleKalmanFilterStation
//! → SdeEstimateSinkStation pipeline driven by `run_iterative_des`) is deferred;
//! the EnKF's filtering behaviour is still exercised directly in group [8].
#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::control_systems::empirical_control::Mulberry32;
    use crate::des::general::control_systems::sde_learning::{
        DenoisingDiffusionModel, DiffusionOptions, DiffusionTrainOptions, EnkfOptions,
        EnsembleKalmanFilter, GbmFamily, Mlp, OuFamily, SdeMaximumLikelihoodEstimator,
        SdeMleOptions,
    };
    use crate::des::general::control_systems::stochastic_sde::{
        EulerMaruyamaIntegrator, GeometricBrownianMotion, OrnsteinUhlenbeck, StochasticDcMotor,
        StochasticDcMotorSpec,
    };

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / 1e-9_f64.max(b.abs())
    }

    fn mle_opts(iterations: usize, learning_rate: f64) -> SdeMleOptions {
        SdeMleOptions {
            iterations: Some(iterations),
            learning_rate: Some(learning_rate),
            fd_eps: None,
        }
    }

    fn motor_spec(current_noise: f64, speed_noise: f64) -> StochasticDcMotorSpec {
        StochasticDcMotorSpec {
            resistance: 2.0,
            inductance: 0.5,
            back_emf_constant: 0.1,
            torque_constant: 0.1,
            inertia: 0.02,
            friction: 0.002,
            voltage: 12.0,
            load_torque: None,
            current_noise,
            speed_noise,
        }
    }

    // [1] Euler–Maruyama — GBM moments vs analytic
    #[test]
    fn gbm_moments_vs_analytic() {
        let gbm = GeometricBrownianMotion::new(0.1, 0.3);
        let em = EulerMaruyamaIntegrator::new();
        let (x0, dt, steps, paths) = (1.0, 0.002, 800usize, 3000u32);
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for p in 0..paths {
            let mut rng = Mulberry32::new(500 + p);
            let sde = em.simulate(&gbm, &[x0], dt, steps, &mut rng);
            let x_t = sde.path[sde.path.len() - 1][0];
            sum += x_t;
            sum_sq += x_t * x_t;
        }
        let t = dt * steps as f64;
        let emp_mean = sum / paths as f64;
        let emp_var = sum_sq / paths as f64 - emp_mean * emp_mean;
        assert!(rel(emp_mean, gbm.mean_at(x0, t)) < 0.03);
        assert!(rel(emp_var, gbm.var_at(x0, t)) < 0.12);

        let a = em
            .simulate(&gbm, &[1.0], 0.01, 10, &mut Mulberry32::new(1))
            .path;
        let b = em
            .simulate(&gbm, &[1.0], 0.01, 10, &mut Mulberry32::new(1))
            .path;
        assert_eq!(a, b);
    }

    // [2] Ornstein–Uhlenbeck — stationary variance
    #[test]
    fn ou_stationary_variance() {
        let ou = OrnsteinUhlenbeck::new(1.0, 0.0, 0.5);
        let sde = EulerMaruyamaIntegrator::new().simulate(
            &ou,
            &[0.0],
            0.01,
            60000,
            &mut Mulberry32::new(3),
        );
        let tail: Vec<f64> = sde.path[sde.path.len() / 2..]
            .iter()
            .map(|x| x[0])
            .collect();
        let mean = tail.iter().sum::<f64>() / tail.len() as f64;
        let variance = tail.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / tail.len() as f64;
        assert!((mean - ou.stationary_mean()).abs() < 0.05);
        assert!(rel(variance, ou.stationary_variance()) < 0.15);
    }

    // [3] ML-1: maximum-likelihood SDE parameter recovery
    #[test]
    fn mle_gbm_recovery() {
        let gbm = GeometricBrownianMotion::new(0.15, 0.25);
        let sde = EulerMaruyamaIntegrator::new().simulate(
            &gbm,
            &[1.0],
            0.004,
            6000,
            &mut Mulberry32::new(11),
        );
        let fit = SdeMaximumLikelihoodEstimator::new(mle_opts(1000, 0.05))
            .fit(&GbmFamily, &sde.times, &sde.path);
        assert!(rel(fit.params["sigma"], 0.25) < 0.1);
        assert!((fit.params["mu"] - 0.15).abs() < 0.15);

        let est = SdeMaximumLikelihoodEstimator::new(SdeMleOptions {
            iterations: None,
            learning_rate: None,
            fd_eps: None,
        });
        let nll_truth =
            est.neg_log_likelihood(&GbmFamily, &[0.15, 0.25_f64.ln()], &sde.times, &sde.path);
        assert!(fit.final_neg_log_lik <= nll_truth + 1e-6);
    }

    // [5] MLP — learns a nonlinear function
    #[test]
    fn mlp_learns_sin() {
        let mut rng = Mulberry32::new(1);
        let mut net = Mlp::new(1, 32, &mut rng);
        for _ in 0..8000 {
            let x = rng.uniform(2.0);
            net.train_example(&[x], x.sin(), 0.02);
        }
        let mut sse = 0.0;
        let m = 200;
        let mut r2 = Mulberry32::new(99);
        for _ in 0..m {
            let x = r2.uniform(2.0);
            let d = net.predict(&[x]) - x.sin();
            sse += d * d;
        }
        assert!((sse / m as f64) < 0.01, "mse={}", sse / m as f64);
    }

    // [6] ML-3: denoising diffusion learns a unimodal target
    #[test]
    fn diffusion_learns_unimodal() {
        let mut rng = Mulberry32::new(5);
        let data: Vec<f64> = (0..2000).map(|_| 3.0 + rng.normal() * 0.6).collect();
        let mut model = DenoisingDiffusionModel::new(DiffusionOptions {
            steps: Some(80),
            beta_min: None,
            beta_max: Some(0.2),
            hidden: Some(64),
            seed: Some(2),
        });
        model.train(
            &data,
            DiffusionTrainOptions {
                iterations: Some(25000),
                learning_rate: Some(0.006),
            },
        );
        let s = DenoisingDiffusionModel::summarise(&model.sample(2000));
        assert!((s.mean - 3.0).abs() < 0.5, "mean={}", s.mean);
        assert!((s.std - 0.6).abs() < 0.35, "std={}", s.std);
    }

    // [7] Deterministic limit — zero-noise motor → analytic steady state
    #[test]
    fn deterministic_motor_steady_state() {
        let (r, ke, kt, bf, v) = (2.0, 0.1, 0.1, 0.002, 12.0);
        let motor = StochasticDcMotor::new(motor_spec(0.0, 0.0));
        let sde = EulerMaruyamaIntegrator::new().simulate(
            &motor,
            &[0.0, 0.0],
            0.001,
            20000,
            &mut Mulberry32::new(1),
        );
        let last = &sde.path[sde.path.len() - 1];
        let (i_end, w_end) = (last[0], last[1]);
        let w_star = v / (ke + (r * bf) / kt);
        let i_star = (bf * w_star) / kt;
        assert!(rel(w_end, w_star) < 0.01, "w={w_end} target={w_star}");
        assert!(rel(i_end, i_star) < 0.02, "i={i_end} target={i_star}");
    }

    // [8] EnKF — posterior uncertainty shrinks
    #[test]
    fn enkf_uncertainty_shrinks() {
        let motor = StochasticDcMotor::new(motor_spec(0.4, 0.5));
        let filter_opts = EnkfOptions {
            ensemble_size: Some(200),
            observation_matrix: vec![vec![0.0, 1.0]],
            observation_noise_var: vec![0.36],
            initial_mean: vec![0.0, 0.0],
            initial_std: vec![3.0, 6.0],
            seed: Some(1),
        };
        let mut filter = EnsembleKalmanFilter::new(Box::new(motor), 0.01, filter_opts);
        let var0 = filter.variance();
        for k in 0..200 {
            filter.step(&[80.0 + (k % 5) as f64 * 0.1]);
        }
        let var_n = filter.variance();
        assert!(var_n[1] < var0[1]);
        assert!(var_n[0] < var0[0]);
        assert!(filter.mean()[0].is_finite() && filter.mean()[1].is_finite());
    }

    // [9] ML-1 — Ornstein–Uhlenbeck parameter recovery
    #[test]
    fn mle_ou_recovery() {
        let ou = OrnsteinUhlenbeck::new(0.8, 1.5, 0.4);
        let sde = EulerMaruyamaIntegrator::new().simulate(
            &ou,
            &[0.0],
            0.01,
            8000,
            &mut Mulberry32::new(21),
        );
        let fit = SdeMaximumLikelihoodEstimator::new(mle_opts(600, 0.05))
            .fit(&OuFamily, &sde.times, &sde.path);
        assert!(rel(fit.params["sigma"], 0.4) < 0.1);
        assert!(rel(fit.params["theta"], 0.8) < 0.3);
        assert!((fit.params["mu"] - 1.5).abs() < 0.3);
    }

    // [10] Diffusion schedule + Brownian increment statistics
    #[test]
    fn diffusion_schedule_and_brownian() {
        let model = DenoisingDiffusionModel::new(DiffusionOptions {
            steps: Some(100),
            beta_min: None,
            beta_max: Some(0.2),
            hidden: None,
            seed: Some(1),
        });
        assert!(model.terminal_signal_retention() < 0.05);
        assert_eq!(model.num_steps(), 100);

        let em = EulerMaruyamaIntegrator::new();
        let mut rng = Mulberry32::new(8);
        let dt = 0.05;
        let n = 20000;
        let mut s = 0.0;
        let mut s2 = 0.0;
        for _ in 0..n {
            let d_w = em.brownian_increment(1, dt, &mut rng)[0];
            s += d_w;
            s2 += d_w * d_w;
        }
        let mean = s / n as f64;
        let variance = s2 / n as f64 - mean * mean;
        assert!(mean.abs() < 0.01, "mean={mean}");
        assert!(rel(variance, dt) < 0.05, "var={variance} dt={dt}");
    }
}
