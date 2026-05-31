//! The **same** M/M/1 queue, but built on the engine's existing *time-stepped*
//! entity network (no FEL). This module only *calls* the existing
//! `entity_source` / `entity_processing` / `entity_sink` stations and the global
//! `SimClock`; it modifies none of them.
//!
//! How the existing engine models a queue: every Δt tick, the source draws
//! `Poisson(λ·Δt)` arrivals and the single-server processor draws
//! `Poisson(μ·Δt)` service completions (`ExponentialRandomVariable::
//! get_next_event_quantity` is Knuth's product-of-uniforms Poisson sampler).
//! The processor maintains time-weighted histograms of its input (waiting) and
//! processing (in-service) queue lengths, plus busy/idle accounting. From those
//! we recover the queueing statistics:
//!
//! * `ρ`  = busy / (busy + idle)               — `get_server_utilization()`
//! * `Lq` = Σ k·time(k) / T over the input histogram (mean waiting-line length)
//! * `L`  = `Lq` + mean number in service
//! * `Wq` = `Lq` / X,  `W` = `L` / X           — Little's law, X = throughput
//!
//! As Δt → 0 the per-tick Poisson counts approach at-most-one-event Bernoulli
//! trials and the results converge to the exact continuous-time M/M/1; larger Δt
//! permits multiple "simultaneous" arrivals/departures per tick (the
//! discretization error this comparison exposes).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use serde::Serialize;

use crate::des::entity_processing::processing::EntityProcessor;
use crate::des::entity_routing::output_routing_policy::OutputRoutingPolicy;
use crate::des::entity_sink::sink::EntitySink;
use crate::des::entity_source::source::EntitySource;
use crate::des::general::general::fisher_yates_shuffle;
use crate::des::general::time_accrued::{
    bump_time_accrued_by_time_step, reset_global_clock, set_step_size,
};
use crate::des::r#abstract::interfaces::{HasInput, HasOutput};
use crate::des::r#abstract::r#abstract::Entity;
use crate::des::random_variables::rv::{ExponentialRandomVariable, RandomVariable};
use crate::des::shared::capabilities::SeededRandom;
use crate::des::shared::precision::{bgn, to_f64, Decimal};

/// Steady-state estimates produced by a time-stepped M/M/1 run.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TimeSteppedMm1Result {
    pub lambda: f64,
    pub mu: f64,
    /// Fixed time step Δt (seconds).
    pub dt: f64,
    pub num_ticks: u64,
    pub seed: u32,
    pub rho: f64,
    pub l: f64,
    pub lq: f64,
    pub w: f64,
    pub wq: f64,
    pub throughput: f64,
    pub num_served: i64,
    /// Total ticks executed (the time-step work metric).
    pub ticks: u64,
    /// Total per-station updates = ticks × #stations (the real work done).
    pub station_updates: u64,
}

fn exp_rv(rate: f64, dt: Decimal, seed: u32) -> Box<dyn RandomVariable> {
    Box::new(ExponentialRandomVariable::new(
        bgn(rate),
        dt,
        Box::new(SeededRandom::new(seed)),
    ))
}

/// Time-average of a length-keyed, time-weighted histogram: Σ k·time(k) / Σ time.
fn histogram_time_average(hist: &HashMap<i64, Decimal>) -> f64 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (length, time) in hist {
        let time = to_f64(*time);
        numerator += (*length as f64) * time;
        denominator += time;
    }
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Build Source → (single-server) Processor → Sink and run it for `num_ticks`
/// ticks of Δt, then recover the M/M/1 statistics. Requires `lambda < mu`.
pub fn run_time_stepped_mm1(
    lambda: f64,
    mu: f64,
    dt: f64,
    num_ticks: u64,
    seed: u32,
) -> TimeSteppedMm1Result {
    assert!(lambda > 0.0 && mu > 0.0, "rates must be positive");
    assert!(lambda < mu, "unstable queue: need lambda < mu (rho < 1)");
    assert!(dt > 0.0 && num_ticks > 0, "need dt > 0 and num_ticks > 0");

    let dt_dec = bgn(dt);
    // The global SimClock is faithful to the canonical driver; our statistics do
    // not depend on it (the stations use the Δt passed to `do_time_step`).
    reset_global_clock();
    set_step_size(dt_dec);

    let source = Rc::new(RefCell::new(EntitySource::new(
        "src".to_string(),
        exp_rv(lambda, dt_dec, seed),
        -1,
    )));
    let server = Rc::new(RefCell::new(EntityProcessor::new(
        "srv".to_string(),
        exp_rv(mu, dt_dec, seed.wrapping_add(1009)),
        OutputRoutingPolicy::default(),
    )));
    server.borrow_mut().concurrency = 1; // single server → M/M/1
    let sink = Rc::new(RefCell::new(EntitySink::new("sink".to_string())));

    source
        .borrow_mut()
        .add_out_connection(server.clone() as Rc<RefCell<dyn HasInput>>);
    server
        .borrow_mut()
        .add_out_connection(sink.clone() as Rc<RefCell<dyn HasInput>>);

    let entities: Vec<Rc<RefCell<dyn Entity>>> = vec![
        source.clone() as Rc<RefCell<dyn Entity>>,
        server.clone() as Rc<RefCell<dyn Entity>>,
        sink.clone() as Rc<RefCell<dyn Entity>>,
    ];

    // Shuffle station order each tick (canonical driver behaviour) to avoid a
    // fixed source-before-server bias.
    let mut order: Vec<usize> = (0..entities.len()).collect();
    let mut shuffle_rng = SeededRandom::new(seed.wrapping_add(31));
    for _ in 0..num_ticks {
        bump_time_accrued_by_time_step(dt_dec);
        fisher_yates_shuffle(&mut order, &mut shuffle_rng);
        for &i in &order {
            entities[i].borrow_mut().do_time_step(dt_dec);
        }
    }

    let total_time = dt * num_ticks as f64;
    let srv = server.borrow();
    let lq = histogram_time_average(&srv.input_queue_histogram);
    let mean_in_service = histogram_time_average(&srv.processing_queue_histogram);
    let l = lq + mean_in_service;
    let rho = to_f64(srv.get_server_utilization());
    drop(srv);

    let num_served = sink.borrow().destroyed_count;
    let throughput = num_served as f64 / total_time;
    let (w, wq) = if throughput > 0.0 {
        (l / throughput, lq / throughput)
    } else {
        (0.0, 0.0)
    };

    TimeSteppedMm1Result {
        lambda,
        mu,
        dt,
        num_ticks,
        seed,
        rho,
        l,
        lq,
        w,
        wq,
        throughput,
        num_served,
        ticks: num_ticks,
        station_updates: num_ticks * entities.len() as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_stepped_recovers_mm1_at_small_dt() {
        // λ=0.5, μ=1.0 → ρ=0.5. With a small Δt the discretized queue should be
        // close to the exact M/M/1.
        let r = run_time_stepped_mm1(0.5, 1.0, 0.05, 120_000, 2024);
        assert!((r.rho - 0.5).abs() < 0.06, "rho={}", r.rho);
        assert!((r.throughput - 0.5).abs() < 0.06, "X={}", r.throughput);
        // Lq for M/M/1 at ρ=0.5 is 0.5; allow generous stochastic slack.
        assert!(r.lq > 0.2 && r.lq < 1.0, "Lq={}", r.lq);
        assert_eq!(r.station_updates, r.ticks * 3);
    }

    #[test]
    fn coarse_dt_still_produces_valid_stats() {
        // A coarse Δt (comparable to the mean service time 1/μ) still yields a
        // valid, finite, stable estimate — just a less accurate one. We only
        // assert validity here; the accuracy-vs-Δt trend is reported (not
        // asserted) by the comparison harness, since it is stochastic.
        let r = run_time_stepped_mm1(0.5, 1.0, 1.0, 6_000, 99);
        assert!(r.rho > 0.0 && r.rho < 1.0, "rho={}", r.rho);
        assert!(r.throughput.is_finite() && r.throughput > 0.0);
        assert!(r.lq.is_finite() && r.lq >= 0.0);
        assert_eq!(r.station_updates, r.ticks * 3);
    }
}
