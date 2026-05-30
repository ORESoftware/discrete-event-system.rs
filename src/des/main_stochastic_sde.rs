//! Port of `src/des/main-stochastic-sde.ts`.
//!
//! Euler–Maruyama SDE engine plus three ML algorithms on top of it:
//!   0. Euler–Maruyama vs analytic GBM moments (the random-process engine).
//!   1. MLE system-id: recover μ, σ of a GBM path by maximum likelihood.
//!   2. Ensemble Kalman Filter recovering a hidden DC-motor current from
//!      speed-only measurements, wired as a DES pipeline.
//!   3. Denoising-diffusion generative model learning a bimodal target.
//!
//! Conversion notes:
//!   - Wiener increments / EnKF / diffusion sampling route through the seeded
//!     `Mulberry32` from `general::control_systems::empirical_control`.
//!   - station classes → struct + impl `DESStation`.
//!   - `class StochasticSdeDemo` → struct + impl; top-level run → [`run`].

use std::cell::RefCell;
use std::rc::Rc;

use crate::des::general::control_systems::empirical_control::Mulberry32;
use crate::des::general::control_systems::sde_learning::{
    DenoisingDiffusionModel, DiffusionOptions, DiffusionTrainOptions, EnkfOptions,
    EnsembleKalmanFilter, EnsembleKalmanFilterStation, GbmFamily, SdeMaximumLikelihoodEstimator,
    SdeMleOptions,
};
use crate::des::general::control_systems::stochastic_sde::{
    EulerMaruyamaIntegrator, GeometricBrownianMotion, SdeChannels, SdeEstimateSinkStation,
    SdePlantOptions, SdePlantStation, StochasticDcMotor, StochasticDcMotorSpec,
};
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::station::{DESStation, StationRef};

/// `Number.prototype` default string for an integer-valued f64 (no trailing
/// `.0`), used for `${trueMu}` / `${trueSigma}`.
fn num_str(x: f64) -> String {
    if x.fract() == 0.0 && x.is_finite() {
        format!("{}", x as i64)
    } else {
        format!("{}", x)
    }
}

struct StochasticSdeDemo;

impl StochasticSdeDemo {
    fn run(&self) {
        self.engine_check();
        self.mle_system_id();
        self.enkf_filtering();
        self.diffusion_generative();
    }

    // 0. Engine — empirical moments vs the closed-form GBM solution.
    fn engine_check(&self) {
        println!("================ 0. SDE engine: GBM Euler–Maruyama vs analytic ================");
        let gbm = GeometricBrownianMotion::new(0.1, 0.3);
        let em = EulerMaruyamaIntegrator::new();
        let x0 = 1.0;
        let dt = 0.002;
        let steps = 1000usize;
        let paths = 4000usize;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for p in 0..paths {
            let mut rng = Mulberry32::new(1000 + p as u32);
            let res = em.simulate(&gbm, &[x0], dt, steps, &mut rng);
            let x_t = res.path[res.path.len() - 1][0];
            sum += x_t;
            sum_sq += x_t * x_t;
        }
        let emp_mean = sum / paths as f64;
        let emp_var = sum_sq / paths as f64 - emp_mean * emp_mean;
        let t = dt * steps as f64;
        println!(
            "  T={:.2}  E[X_T]: empirical {:.4} vs analytic {:.4}",
            t,
            emp_mean,
            gbm.mean_at(x0, t)
        );
        println!(
            "           Var[X_T]: empirical {:.4} vs analytic {:.4}",
            emp_var,
            gbm.var_at(x0, t)
        );
    }

    // 1. ML system identification — MLE of GBM drift/diffusion from one path.
    fn mle_system_id(&self) {
        println!();
        println!("================ 1. ML system-id: maximum-likelihood SDE fit ================");
        let true_mu = 0.12;
        let true_sigma = 0.3;
        let gbm = GeometricBrownianMotion::new(true_mu, true_sigma);
        let mut rng = Mulberry32::new(77);
        let sim = EulerMaruyamaIntegrator::new().simulate(&gbm, &[1.0], 0.004, 6000, &mut rng);
        let est = SdeMaximumLikelihoodEstimator::new(SdeMleOptions {
            iterations: Some(1500),
            learning_rate: Some(0.05),
            fd_eps: None,
        });
        let fit = est.fit(&GbmFamily, &sim.times, &sim.path);
        println!("  true   : mu={}, sigma={}", num_str(true_mu), num_str(true_sigma));
        println!(
            "  learned: mu={:.4}, sigma={:.4}   (NLL={:.1}, {} Adam steps)",
            fit.params["mu"], fit.params["sigma"], fit.final_neg_log_lik, fit.iterations
        );
    }

    // 2. ML filtering — EnKF recovers hidden current from speed-only measurements.
    fn enkf_filtering(&self) {
        println!();
        println!("================ 2. ML filtering: Ensemble Kalman Filter (DES pipeline) ================");
        let spec = StochasticDcMotorSpec {
            resistance: 2.0,
            inductance: 0.5,
            back_emf_constant: 0.1,
            torque_constant: 0.1,
            inertia: 0.02,
            friction: 0.002,
            voltage: 12.0,
            load_torque: None,
            current_noise: 0.4,
            speed_noise: 0.5,
        };
        let dt = 0.01;
        let steps = 500usize;
        let h: Vec<Vec<f64>> = vec![vec![0.0, 1.0]]; // observe ω only; current i is hidden
        let plant = Rc::new(RefCell::new(SdePlantStation::new(
            "motor-plant",
            SdePlantOptions {
                system: Box::new(StochasticDcMotor::new(spec.clone())),
                x0: vec![0.0, 0.0],
                dt,
                steps,
                observation_matrix: Some(h.clone()),
                observation_noise_std: Some(vec![0.6]),
                seed: Some(5),
            },
        )));
        let filter = EnsembleKalmanFilter::new(
            Box::new(StochasticDcMotor::new(spec.clone())),
            dt,
            EnkfOptions {
                ensemble_size: Some(150),
                observation_matrix: h.clone(),
                observation_noise_var: vec![0.36],
                initial_mean: vec![0.0, 0.0],
                initial_std: vec![2.0, 5.0],
                seed: Some(9),
            },
        );
        let enkf = Rc::new(RefCell::new(EnsembleKalmanFilterStation::new("enkf", filter)));
        let sink = Rc::new(RefCell::new(SdeEstimateSinkStation::new("sink")));

        let plant_ref: StationRef = plant.clone();
        let enkf_ref: StationRef = enkf.clone();
        let sink_ref: StationRef = sink.clone();

        plant.borrow_mut().core_mut().pipe(
            enkf_ref.clone(),
            SdeChannels::OBSERVATION,
            SdeChannels::OBSERVATION,
        );
        plant.borrow_mut().core_mut().pipe(
            sink_ref.clone(),
            SdeChannels::STATE,
            SdeChannels::STATE,
        );
        enkf.borrow_mut().core_mut().pipe(
            sink_ref.clone(),
            SdeChannels::ESTIMATE,
            SdeChannels::ESTIMATE,
        );

        run_iterative_des(
            vec![plant_ref, enkf_ref, sink_ref],
            IterativeRunOptions { shuffle: false, max_ticks: Some(steps + 5), ..Default::default() },
        );

        let sink_b = sink.borrow();
        let rmse = sink_b.rmse_by_dimension();
        // Baseline current RMSE if you just guessed the mean current (no filter).
        let n = sink_b.truth.len() as f64;
        let mean_i = sink_b.truth.iter().map(|t| t.state[0]).sum::<f64>() / n;
        let base_i = (sink_b
            .truth
            .iter()
            .map(|t| {
                let d = t.state[0] - mean_i;
                d * d
            })
            .sum::<f64>()
            / n)
            .sqrt();
        println!("  observed: speed ω (noisy, σ=0.6);  hidden: current i");
        println!("  EnKF RMSE  → current i = {:.4},  speed ω = {:.4}", rmse[0], rmse[1]);
        println!(
            "  baseline   → current i (guess mean) = {:.4}   ⇒ filter recovers the hidden state",
            base_i
        );
    }

    // 3. ML generative — denoising diffusion learns a bimodal target.
    fn diffusion_generative(&self) {
        println!();
        println!("================ 3. ML generative: score-based diffusion (reverse SDE) ================");
        let mut rng = Mulberry32::new(2024);
        let mut data: Vec<f64> = Vec::new();
        for _ in 0..3000 {
            let mode = if rng.next() < 0.5 { -2.0 } else { 2.0 };
            data.push(mode + rng.normal() * 0.4);
        }
        let mut model = DenoisingDiffusionModel::new(DiffusionOptions {
            steps: Some(100),
            beta_min: None,
            beta_max: Some(0.2),
            hidden: Some(128),
            seed: Some(3),
        });
        let loss = model.train(
            &data,
            DiffusionTrainOptions { iterations: Some(60000), learning_rate: Some(0.004) },
        );
        let samples = model.sample(3000);
        let data_stats = DenoisingDiffusionModel::summarise(&data);
        let gen_stats = DenoisingDiffusionModel::summarise(&samples);
        let near_neg = samples.iter().filter(|s| **s < 0.0).count() as f64 / samples.len() as f64;
        println!(
            "  target  : bimodal N(±2, 0.4²)   data mean/std = {:.3} / {:.3}",
            data_stats.mean, data_stats.std
        );
        println!(
            "  learned : sample mean/std = {:.3} / {:.3}   (final DSM loss {:.4})",
            gen_stats.mean, gen_stats.std, loss
        );
        println!(
            "  modes   : {:.0}% near −2, {:.0}% near +2  (target ≈ 50/50)",
            near_neg * 100.0,
            (1.0 - near_neg) * 100.0
        );
    }
}

/// Entry point (`main()` in the TS source).
pub fn run() {
    StochasticSdeDemo.run();
}
