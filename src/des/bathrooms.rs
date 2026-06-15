//! Household bathroom occupancy as a discrete-event, blocking-loss simulation.
//!
//! Question being answered: in a house of `people` people sharing `bathrooms`
//! bathrooms, where each person visits a bathroom `visits_per_day` times a day
//! at uniformly-random times for `service_min` minutes each, what fraction of
//! the time is **no** bathroom occupied, **one** occupied, and **both**
//! occupied? A bathroom in use cannot admit anyone — an arrival that finds
//! every bathroom busy is *rejected* (lost demand), and the same person never
//! occupies two bathrooms at once.
//!
//! This is the classic finite-source loss system (an Engset/Erlang-loss cousin).
//! With `p = visits_per_day * service_min / 1440` the per-person probability of
//! being in a bathroom at a random instant, the number currently in bathrooms is
//! approximately `Binomial(people, p)` and the number of *occupied* bathrooms is
//! `min(that, bathrooms)`. We do not hard-code that — we run a Monte-Carlo DES
//! and let the time-weighted occupancy fractions emerge, then compare against the
//! closed-form binomial as a sanity check.
//!
//! Two products come out of one model:
//!   * [`run_bathroom_montecarlo`] — many independent replications, warm-up
//!     discarded, returning time-weighted P(0)/P(1)/P(2) plus a Monte-Carlo
//!     convergence trace.
//!   * a recorded *sample timeline* (a handful of fully-detailed days) that the
//!     [`render_bathrooms_html`] animation plays back.

use crate::des::shared::capabilities::{RandomSource, SeededRandom};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

const MINUTES_PER_DAY: f64 = 1440.0;

/// Knobs for the household bathroom study. Defaults encode the posed problem:
/// 8 people, 2 bathrooms, 3 visits/day, 20 minutes each.
#[derive(Clone, Debug)]
pub struct BathroomConfig {
    pub people: usize,
    pub bathrooms: usize,
    pub visits_per_day: usize,
    pub service_min: f64,
    /// Independent Monte-Carlo replications (each gets its own seed). Averaging
    /// across many short replications, rather than one long run, gives a clean
    /// convergence curve and washes out start/stop edges.
    pub replications: usize,
    /// Days simulated per replication (the measured horizon, after warm-up).
    pub days_per_rep: usize,
    /// Leading days discarded from each replication to remove the cold-start
    /// edge (no one is mid-visit at t=0).
    pub warmup_days: usize,
    /// Days captured at full per-event detail for the animation playback.
    pub sample_days: usize,
    pub seed: u32,
}

impl Default for BathroomConfig {
    fn default() -> Self {
        return BathroomConfig {
            people: 8,
            bathrooms: 2,
            visits_per_day: 3,
            service_min: 20.0,
            replications: 400,
            days_per_rep: 50,
            warmup_days: 1,
            sample_days: 4,
            seed: 0x0BA7_4007,
        };
    }
}

/// Closed-form binomial reference the Monte-Carlo result is checked against.
#[derive(Clone, Debug)]
pub struct AnalyticReference {
    /// Per-person probability of being in a bathroom at a random instant.
    pub p_person: f64,
    /// P(exactly 0 / exactly 1 / 2-or-more in bathrooms) under Binomial.
    pub p0: f64,
    pub p1: f64,
    pub p2_plus: f64,
    /// E[number of bathrooms occupied] = E[min(X, bathrooms)].
    pub expected_occupied: f64,
    /// P(both occupied | at least one occupied).
    pub p_both_given_any: f64,
}

/// One recorded event in the animation sample timeline.
#[derive(Clone, Debug)]
pub struct SampleEvent {
    /// Minutes since the start of the sample window.
    pub t: f64,
    /// `"enter"`, `"leave"`, or `"reject"`.
    pub kind: &'static str,
    pub person: usize,
    /// Bathroom slot used (enter/leave); `usize::MAX` for a rejection.
    pub bathroom: usize,
    /// Number of bathrooms occupied immediately after this event.
    pub occupied_after: usize,
}

/// A single served (or rejected) visit in the sample window, for the gantt.
#[derive(Clone, Debug)]
pub struct SampleVisit {
    pub person: usize,
    pub start: f64,
    pub end: f64,
    pub bathroom: usize,
    pub rejected: bool,
}

/// Everything the report and animation need.
#[derive(Clone, Debug)]
pub struct BathroomReport {
    pub config: BathroomConfig,
    /// Time-weighted fraction of the measured horizon with exactly k bathrooms
    /// occupied, indexed by k (length `bathrooms + 1`).
    pub occupancy_fraction: Vec<f64>,
    /// Mean number of bathrooms occupied (time-weighted).
    pub mean_occupied: f64,
    /// Fraction of all bathroom *requests* that were rejected (both busy).
    pub blocked_fraction: f64,
    /// Total requests generated across the measured horizon (all replications).
    pub total_requests: u64,
    /// Cumulative Monte-Carlo estimate of `occupancy_fraction` after each
    /// replication — the convergence trace. Outer index = replication.
    pub convergence: Vec<Vec<f64>>,
    pub analytic: AnalyticReference,
    /// Fully detailed sample window for animation playback.
    pub sample_events: Vec<SampleEvent>,
    pub sample_visits: Vec<SampleVisit>,
    pub sample_minutes: f64,
}

/// A scheduled future event in the DES future-event list.
#[derive(Clone, Copy, Debug)]
struct Event {
    time: f64,
    seq: u64,
    kind: EventKind,
}

#[derive(Clone, Copy, Debug)]
enum EventKind {
    Arrival { person: usize },
    Departure { bathroom: usize, person: usize },
}

// Min-heap on time (BinaryHeap is a max-heap, so the ordering is reversed). Ties
// broken by insertion sequence for deterministic, FIFO-stable replay.
impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        return self.time == other.time && self.seq == other.seq;
    }
}
impl Eq for Event {}
impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        return other
            .time
            .partial_cmp(&self.time)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.seq.cmp(&self.seq));
    }
}
impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        return Some(self.cmp(other));
    }
}

/// Per-person sorted, self-non-overlapping arrival times over `days` days.
///
/// Each person draws `visits_per_day` uniform times within each day. A draw that
/// would land while the same person is still in a bathroom is dropped — a person
/// cannot be in two places at once, and self-overlap would double-count their
/// occupancy. (With 3 draws in 1440 minutes and a 20-minute service this is a
/// rare correction.)
fn generate_person_arrivals(
    rng: &mut SeededRandom,
    person: usize,
    days: usize,
    visits_per_day: usize,
    service_min: f64,
) -> Vec<(f64, usize)> {
    let mut starts: Vec<f64> = Vec::with_capacity(days * visits_per_day);
    for d in 0..days {
        let day_base = d as f64 * MINUTES_PER_DAY;
        for _ in 0..visits_per_day {
            starts.push(day_base + rng.next_float() * MINUTES_PER_DAY);
        }
    }
    starts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let mut kept: Vec<(f64, usize)> = Vec::with_capacity(starts.len());
    let mut busy_until = f64::NEG_INFINITY;
    for s in starts {
        if s >= busy_until {
            kept.push((s, person));
            busy_until = s + service_min;
        }
    }
    return kept;
}

/// Run one replication; integrate occupancy over `[warmup, horizon)` and, when
/// `record` is set, capture every event and visit for animation.
struct RepResult {
    /// Minutes spent with exactly k bathrooms busy, indexed by k.
    occupied_minutes: Vec<f64>,
    measured_minutes: f64,
    requests: u64,
    blocked: u64,
    events: Vec<SampleEvent>,
    visits: Vec<SampleVisit>,
}

fn simulate_replication(cfg: &BathroomConfig, seed: u32, record: bool) -> RepResult {
    let mut rng = SeededRandom::new(seed);
    let horizon = cfg.days_per_rep as f64 * MINUTES_PER_DAY;
    let warmup = cfg.warmup_days as f64 * MINUTES_PER_DAY;
    // When recording the animation we only want the first `sample_days`.
    let record_window = cfg.sample_days as f64 * MINUTES_PER_DAY;

    // Build the full arrival stream up front (interleaved across people via the
    // heap), so the RNG draw order is per-person and reproducible.
    let mut heap: BinaryHeap<Event> = BinaryHeap::new();
    let mut seq: u64 = 0;
    for person in 0..cfg.people {
        let arrivals = generate_person_arrivals(
            &mut rng,
            person,
            cfg.days_per_rep,
            cfg.visits_per_day,
            cfg.service_min,
        );
        for (time, person) in arrivals {
            heap.push(Event {
                time,
                seq,
                kind: EventKind::Arrival { person },
            });
            seq += 1;
        }
    }

    let mut occupied_minutes = vec![0.0_f64; cfg.bathrooms + 1];
    let mut busy = 0_usize; // number of occupied bathrooms right now
    let mut slot_free: Vec<bool> = vec![true; cfg.bathrooms];
    let mut clock = 0.0_f64;
    let mut requests = 0_u64;
    let mut blocked = 0_u64;
    let mut events: Vec<SampleEvent> = Vec::new();
    let mut visits: Vec<SampleVisit> = Vec::new();

    // Accumulate the [last, now) slice into the measured (post-warm-up) buckets.
    let accumulate = |occupied_minutes: &mut Vec<f64>, from: f64, to: f64, busy: usize| {
        let lo = from.max(warmup);
        let hi = to.min(horizon);
        if hi > lo {
            occupied_minutes[busy] += hi - lo;
        }
    };

    while let Some(ev) = heap.pop() {
        if ev.time >= horizon {
            break;
        }
        accumulate(&mut occupied_minutes, clock, ev.time, busy);
        clock = ev.time;

        match ev.kind {
            EventKind::Arrival { person } => {
                requests += 1;
                if busy < cfg.bathrooms {
                    let slot = slot_free.iter().position(|free| *free).unwrap();
                    slot_free[slot] = false;
                    busy += 1;
                    let depart = ev.time + cfg.service_min;
                    heap.push(Event {
                        time: depart,
                        seq,
                        kind: EventKind::Departure {
                            bathroom: slot,
                            person,
                        },
                    });
                    seq += 1;
                    if record && ev.time < record_window {
                        events.push(SampleEvent {
                            t: ev.time,
                            kind: "enter",
                            person,
                            bathroom: slot,
                            occupied_after: busy,
                        });
                        visits.push(SampleVisit {
                            person,
                            start: ev.time,
                            end: depart,
                            bathroom: slot,
                            rejected: false,
                        });
                    }
                } else {
                    blocked += 1;
                    if record && ev.time < record_window {
                        events.push(SampleEvent {
                            t: ev.time,
                            kind: "reject",
                            person,
                            bathroom: usize::MAX,
                            occupied_after: busy,
                        });
                        visits.push(SampleVisit {
                            person,
                            start: ev.time,
                            end: ev.time + cfg.service_min,
                            bathroom: usize::MAX,
                            rejected: true,
                        });
                    }
                }
            }
            EventKind::Departure { bathroom, person } => {
                slot_free[bathroom] = true;
                busy -= 1;
                if record && ev.time < record_window {
                    events.push(SampleEvent {
                        t: ev.time,
                        kind: "leave",
                        person,
                        bathroom,
                        occupied_after: busy,
                    });
                }
            }
        }
    }
    // Tail slice from the last event to the horizon.
    accumulate(&mut occupied_minutes, clock, horizon, busy);

    let measured_minutes = (horizon - warmup).max(0.0);
    return RepResult {
        occupied_minutes,
        measured_minutes,
        requests,
        blocked,
        events,
        visits,
    };
}

/// Closed-form Binomial(`people`, `p`) reference with occupancy capped at
/// `bathrooms`.
fn analytic_reference(cfg: &BathroomConfig) -> AnalyticReference {
    let p = (cfg.visits_per_day as f64 * cfg.service_min) / MINUTES_PER_DAY;
    let n = cfg.people;
    // Binomial pmf P(X = k).
    let pmf = |k: usize| -> f64 {
        let mut c = 1.0_f64;
        for i in 0..k {
            c *= (n - i) as f64 / (i + 1) as f64;
        }
        return c * p.powi(k as i32) * (1.0 - p).powi((n - k) as i32);
    };
    let p0 = pmf(0);
    let p1 = pmf(1);
    let p2_plus = (1.0 - p0 - p1).max(0.0);

    let mut expected_occupied = 0.0_f64;
    for k in 0..=n {
        let occ = k.min(cfg.bathrooms) as f64;
        expected_occupied += occ * pmf(k);
    }
    let p_both_given_any = if (1.0 - p0) > 0.0 {
        p2_plus / (1.0 - p0)
    } else {
        0.0
    };

    return AnalyticReference {
        p_person: p,
        p0,
        p1,
        p2_plus,
        expected_occupied,
        p_both_given_any,
    };
}

/// Run the full Monte-Carlo study and capture one detailed sample window.
pub fn run_bathroom_montecarlo(cfg: &BathroomConfig) -> BathroomReport {
    let buckets = cfg.bathrooms + 1;
    let mut cum_minutes = vec![0.0_f64; buckets];
    let mut cum_measured = 0.0_f64;
    let mut total_requests = 0_u64;
    let mut total_blocked = 0_u64;
    let mut convergence: Vec<Vec<f64>> = Vec::with_capacity(cfg.replications);

    for rep in 0..cfg.replications {
        // Distinct, well-spread seed per replication.
        let seed = cfg
            .seed
            .wrapping_add((rep as u32).wrapping_mul(0x9E37_79B9))
            .wrapping_add(1);
        let r = simulate_replication(cfg, seed, false);
        for k in 0..buckets {
            cum_minutes[k] += r.occupied_minutes[k];
        }
        cum_measured += r.measured_minutes;
        total_requests += r.requests;
        total_blocked += r.blocked;

        let snapshot: Vec<f64> = if cum_measured > 0.0 {
            cum_minutes.iter().map(|m| m / cum_measured).collect()
        } else {
            vec![0.0; buckets]
        };
        convergence.push(snapshot);
    }

    let occupancy_fraction: Vec<f64> = if cum_measured > 0.0 {
        cum_minutes.iter().map(|m| m / cum_measured).collect()
    } else {
        vec![0.0; buckets]
    };
    let mean_occupied: f64 = occupancy_fraction
        .iter()
        .enumerate()
        .map(|(k, f)| k as f64 * f)
        .sum();
    let blocked_fraction = if total_requests > 0 {
        total_blocked as f64 / total_requests as f64
    } else {
        0.0
    };

    // Capture one detailed sample run for the animation (its own fixed seed so
    // the playback is reproducible and independent of the stats sweep).
    let sample = simulate_replication(cfg, cfg.seed ^ 0x5DEE_CE66, true);

    return BathroomReport {
        config: cfg.clone(),
        occupancy_fraction,
        mean_occupied,
        blocked_fraction,
        total_requests,
        convergence,
        analytic: analytic_reference(cfg),
        sample_events: sample.events,
        sample_visits: sample.visits,
        sample_minutes: cfg.sample_days as f64 * MINUTES_PER_DAY,
    };
}

/// Serialize the report into the JSON payload the animation consumes.
fn report_to_json(report: &BathroomReport) -> serde_json::Value {
    use serde_json::json;
    let cfg = &report.config;
    let occ = &report.occupancy_fraction;
    let a = &report.analytic;

    let convergence: Vec<serde_json::Value> = report
        .convergence
        .iter()
        .map(|snap| serde_json::Value::from(snap.clone()))
        .collect();
    let events: Vec<serde_json::Value> = report
        .sample_events
        .iter()
        .map(|e| {
            json!({
                "t": e.t,
                "kind": e.kind,
                "person": e.person,
                "bathroom": if e.bathroom == usize::MAX { -1 } else { e.bathroom as i64 },
                "occ": e.occupied_after,
            })
        })
        .collect();
    let visits: Vec<serde_json::Value> = report
        .sample_visits
        .iter()
        .map(|v| {
            json!({
                "person": v.person,
                "start": v.start,
                "end": v.end,
                "bathroom": if v.bathroom == usize::MAX { -1 } else { v.bathroom as i64 },
                "rejected": v.rejected,
            })
        })
        .collect();

    return json!({
        "config": {
            "people": cfg.people,
            "bathrooms": cfg.bathrooms,
            "visitsPerDay": cfg.visits_per_day,
            "serviceMin": cfg.service_min,
            "replications": cfg.replications,
            "daysPerRep": cfg.days_per_rep,
            "sampleDays": cfg.sample_days,
        },
        "sim": {
            "p": occ,
            "mean": report.mean_occupied,
            "blocked": report.blocked_fraction,
            "requests": report.total_requests,
        },
        "analytic": {
            "pPerson": a.p_person,
            "p0": a.p0,
            "p1": a.p1,
            "p2": a.p2_plus,
            "mean": a.expected_occupied,
            "bothGivenAny": a.p_both_given_any,
        },
        "convergence": convergence,
        "sampleMinutes": report.sample_minutes,
        "events": events,
        "visits": visits,
    });
}

/// Render the full self-contained HTML animation + report for a finished study.
pub fn render_bathrooms_html(report: &BathroomReport) -> String {
    let data = report_to_json(report).to_string();
    return BATHROOMS_HTML_TEMPLATE.replace("/*__BATH_DATA__*/0", &data);
}

/// Convenience: run the default study and render it in one call.
pub fn render_default_bathrooms_html() -> String {
    return render_bathrooms_html(&run_bathroom_montecarlo(&BathroomConfig::default()));
}

const BATHROOMS_HTML_TEMPLATE: &str = include_str!("bathrooms.html");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn occupancy_matches_binomial_reference() {
        let cfg = BathroomConfig {
            replications: 200,
            days_per_rep: 40,
            ..BathroomConfig::default()
        };
        let report = run_bathroom_montecarlo(&cfg);
        let a = &report.analytic;
        // The Monte-Carlo time-weighted fractions should track the closed-form
        // binomial within a couple of percentage points.
        assert!(
            (report.occupancy_fraction[0] - a.p0).abs() < 0.02,
            "P0 sim {} vs analytic {}",
            report.occupancy_fraction[0],
            a.p0
        );
        assert!(
            (report.occupancy_fraction[1] - a.p1).abs() < 0.02,
            "P1 sim {} vs analytic {}",
            report.occupancy_fraction[1],
            a.p1
        );
        assert!(
            (report.occupancy_fraction[2] - a.p2_plus).abs() < 0.02,
            "P2 sim {} vs analytic {}",
            report.occupancy_fraction[2],
            a.p2_plus
        );
        // Fractions form a distribution.
        let total: f64 = report.occupancy_fraction.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "fractions sum to {total}");
    }

    #[test]
    fn analytic_reference_is_in_expected_ballpark() {
        let a = analytic_reference(&BathroomConfig::default());
        assert!((a.p_person - 1.0 / 24.0).abs() < 1e-9);
        assert!(a.p0 > 0.68 && a.p0 < 0.74, "p0 = {}", a.p0);
        assert!(a.p1 > 0.22 && a.p1 < 0.27, "p1 = {}", a.p1);
        assert!(a.p2_plus > 0.03 && a.p2_plus < 0.06, "p2 = {}", a.p2_plus);
    }

    #[test]
    fn sample_window_records_events() {
        let report = run_bathroom_montecarlo(&BathroomConfig::default());
        assert!(!report.sample_events.is_empty());
        // Every recorded event lies within the sample window.
        assert!(report
            .sample_events
            .iter()
            .all(|e| e.t >= 0.0 && e.t < report.sample_minutes + report.config.service_min));
    }
}
