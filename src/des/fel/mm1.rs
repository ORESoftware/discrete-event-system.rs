//! An exact, event-driven **M/M/1 queue** on the [`Engine`] FEL.
//!
//! Two event types — arrival and departure — fully describe a single-server
//! FIFO queue with Poisson(λ) arrivals and Exp(μ) service:
//!
//! * **arrival**: a customer enters; schedule the next arrival at `Exp(λ)`. If
//!   the server is idle, begin service immediately (schedule its departure at
//!   `Exp(μ)`); otherwise the customer waits in line.
//! * **departure**: a customer leaves; if anyone is waiting, begin serving the
//!   head of the line; otherwise the server goes idle.
//!
//! Because the clock jumps event-to-event, time-average statistics are exact:
//! we accumulate the areas ∫N(t)dt, ∫Q(t)dt and busy time over each inter-event
//! interval. Per-customer waiting time is tracked directly (no discretization).
//!
//! The same seeded RNG family as the time-stepped engine is reused
//! ([`SeededRandom`] + [`sample_exponential`]) so the comparison is fair.

use std::collections::VecDeque;

use serde::Serialize;

use crate::des::general::random_variables::sample_exponential;
use crate::des::shared::capabilities::SeededRandom;

use super::engine::Engine;

/// Mutable M/M/1 state plus time-weighted accumulators.
pub struct Mm1World {
    rng: SeededRandom,
    lambda: f64,
    mu: f64,
    /// Customers currently in the system (waiting + in service).
    n: u64,
    /// Whether the single server is busy.
    busy: bool,
    /// Arrival timestamps of customers waiting in line (FIFO).
    waiting: VecDeque<f64>,
    /// Time of the last accumulator update.
    last_t: f64,
    /// ∫ N(t) dt — area under number-in-system.
    area_n: f64,
    /// ∫ Q(t) dt — area under number-waiting (excludes the one in service).
    area_q: f64,
    /// ∫ 1{busy} dt — total server-busy time.
    busy_time: f64,
    num_arrivals: u64,
    num_served: u64,
    /// Customers whose service has started (denominator for direct mean wait).
    num_started: u64,
    /// Σ (service-start − arrival) over all customers that started service.
    total_wait: f64,
}

impl Mm1World {
    fn new(seed: u32, lambda: f64, mu: f64) -> Self {
        Mm1World {
            rng: SeededRandom::new(seed),
            lambda,
            mu,
            n: 0,
            busy: false,
            waiting: VecDeque::new(),
            last_t: 0.0,
            area_n: 0.0,
            area_q: 0.0,
            busy_time: 0.0,
            num_arrivals: 0,
            num_served: 0,
            num_started: 0,
            total_wait: 0.0,
        }
    }

    /// Accumulate time-weighted areas over the interval `[last_t, t]`.
    fn advance_area(&mut self, t: f64) {
        let dt = t - self.last_t;
        if dt > 0.0 {
            self.area_n += self.n as f64 * dt;
            let q = self.n.saturating_sub(if self.busy { 1 } else { 0 });
            self.area_q += q as f64 * dt;
            if self.busy {
                self.busy_time += dt;
            }
        }
        self.last_t = t;
    }
}

fn arrival(eng: &mut Engine<Mm1World>) {
    let t = eng.now();
    eng.world.advance_area(t);
    eng.world.n += 1;
    eng.world.num_arrivals += 1;

    // Schedule the next arrival.
    let lambda = eng.world.lambda;
    let interarrival = sample_exponential(&mut eng.world.rng, lambda);
    eng.schedule_after(interarrival, arrival);

    if !eng.world.busy {
        // Server was idle: this customer starts service immediately (wait = 0).
        eng.world.busy = true;
        eng.world.num_started += 1;
        let mu = eng.world.mu;
        let service = sample_exponential(&mut eng.world.rng, mu);
        eng.schedule_after(service, departure);
    } else {
        eng.world.waiting.push_back(t);
    }
}

fn departure(eng: &mut Engine<Mm1World>) {
    let t = eng.now();
    eng.world.advance_area(t);
    eng.world.n -= 1;
    eng.world.num_served += 1;

    if let Some(arrived_at) = eng.world.waiting.pop_front() {
        // Start serving the head of the line.
        eng.world.total_wait += t - arrived_at;
        eng.world.num_started += 1;
        let mu = eng.world.mu;
        let service = sample_exponential(&mut eng.world.rng, mu);
        eng.schedule_after(service, departure);
    } else {
        eng.world.busy = false;
    }
}

/// Steady-state estimates produced by an FEL M/M/1 run.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FelMm1Result {
    pub lambda: f64,
    pub mu: f64,
    pub horizon: f64,
    pub seed: u32,
    /// Server utilization (busy fraction).
    pub rho: f64,
    /// Time-average number in system.
    pub l: f64,
    /// Time-average number waiting in queue.
    pub lq: f64,
    /// Mean time in system (Little: L / throughput).
    pub w: f64,
    /// Mean wait in queue (Little: Lq / throughput).
    pub wq: f64,
    /// Mean wait in queue measured directly per customer (cross-check).
    pub wq_direct: f64,
    pub throughput: f64,
    pub num_arrivals: u64,
    pub num_served: u64,
    /// Total FEL events processed (≈ 2 per customer).
    pub events: u64,
}

/// Run an exact FEL M/M/1 to time `horizon`. Requires `lambda < mu` for a
/// stable queue (panics otherwise — an unstable queue has no steady state).
pub fn run_fel_mm1(lambda: f64, mu: f64, horizon: f64, seed: u32) -> FelMm1Result {
    assert!(lambda > 0.0 && mu > 0.0, "rates must be positive");
    assert!(lambda < mu, "unstable queue: need lambda < mu (rho < 1)");
    assert!(horizon > 0.0, "horizon must be positive");

    let mut eng = Engine::new(Mm1World::new(seed, lambda, mu));
    // Seed the FEL with the first arrival.
    let first = sample_exponential(&mut eng.world.rng, lambda);
    eng.schedule_after(first, arrival);
    eng.run_until(horizon);
    // Close the final interval up to the horizon (clock is parked at `horizon`).
    eng.world.advance_area(horizon);

    let w = &eng.world;
    let t = horizon;
    let throughput = w.num_served as f64 / t;
    let l = w.area_n / t;
    let lq = w.area_q / t;
    let (w_sys, wq) = if throughput > 0.0 {
        (l / throughput, lq / throughput)
    } else {
        (0.0, 0.0)
    };
    let wq_direct = if w.num_started > 0 {
        w.total_wait / w.num_started as f64
    } else {
        0.0
    };

    FelMm1Result {
        lambda,
        mu,
        horizon,
        seed,
        rho: w.busy_time / t,
        l,
        lq,
        w: w_sys,
        wq,
        wq_direct,
        throughput,
        num_arrivals: w.num_arrivals,
        num_served: w.num_served,
        events: eng.events_processed(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_analytical_mm1_at_rho_half() {
        // λ=0.5, μ=1.0 → ρ=0.5, L=1, Lq=0.5, W=2, Wq=1.
        let r = run_fel_mm1(0.5, 1.0, 60_000.0, 12345);
        assert!((r.rho - 0.5).abs() < 0.03, "rho={}", r.rho);
        assert!((r.l - 1.0).abs() < 0.12, "L={}", r.l);
        assert!((r.lq - 0.5).abs() < 0.1, "Lq={}", r.lq);
        assert!((r.throughput - 0.5).abs() < 0.03, "X={}", r.throughput);
        // Little's law derived wait vs directly measured wait should agree.
        assert!(
            (r.wq - r.wq_direct).abs() < 0.1,
            "wq={} wq_direct={}",
            r.wq,
            r.wq_direct
        );
    }

    #[test]
    fn events_are_about_two_per_customer() {
        let r = run_fel_mm1(0.5, 1.0, 20_000.0, 7);
        // Each customer generates one arrival + one departure event.
        let ratio = r.events as f64 / r.num_served as f64;
        assert!(ratio > 1.8 && ratio < 2.3, "events/customer = {ratio}");
    }

    #[test]
    #[should_panic(expected = "unstable queue")]
    fn rejects_unstable_queue() {
        let _ = run_fel_mm1(1.5, 1.0, 1000.0, 1);
    }
}
