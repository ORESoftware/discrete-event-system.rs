//! Port of `src/des/main-two-disease.ts`.
//!
//! Two-disease SIR with co-infection interaction (six compartments: S, A, B,
//! AB, R, D). Each compartment is a tick-driven station holding a population of
//! `Person` tokens; a `WorldCensus` station freezes the global counts at the
//! start of each tick so transitions are order-independent.
//!
//! Conversion notes:
//!   - the cyclic compartment graph and the shared people are modelled with
//!     `Rc<RefCell<…>>` (TS object aliasing).
//!   - mass-action incidence sampling routes through `SeededRandom`; the tick
//!     shuffle uses the `with_seed` stream, per-person draws the `mulberry32`
//!     stream — two independent streams both seeded with `seed`, exactly as the
//!     TS uses the ambient `withSeed` RNG vs. the local `mulberry32(seed)`.
//!   - `competingRisks` / `poissonBinomialPMF` / `binomialPMF` are not in the
//!     Rust `random_variables` module yet, so they are ported as local helpers
//!     (see PORT NOTE below).
//!   - `async main` → [`run`]; `JSON.stringify` → hand-built JSON strings.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

use crate::des::general::general::fisher_yates_shuffle;
use crate::des::general::prng::{mulberry32, with_seed};
use crate::des::general::random_variables::{mean_from_pmf, sample_categorical, variance_from_pmf};
use crate::des::general::time_stepped_station::TimeSteppedStation;
use crate::des::shared::capabilities::{RandomSource, SeededRandom};

// -----------------------------------------------------------------------------
// PORT NOTE: `competingRisks`, `poissonBinomialPMF` and `binomialPMF` from the
// TS `general/random-variables` are not yet present in the Rust
// `general::random_variables` module (which only exposes
// `sample_categorical` / `mean_from_pmf` / `variance_from_pmf`). They are
// faithful local ports; wire to a shared module once it grows them.
// -----------------------------------------------------------------------------

/// Exact discrete-time competing-risks first-event probabilities
/// `[p_no, p_1, …, p_K]` for continuous rates `λ_k` over a step `dt`.
fn competing_risks(rates: &[f64], dt: f64) -> Vec<f64> {
    if dt < 0.0 {
        panic!("bad dt {dt}");
    }
    let mut total = 0.0;
    for (i, &r) in rates.iter().enumerate() {
        if r < 0.0 {
            panic!("bad rate[{i}] {r}");
        }
        total += r;
    }
    if total == 0.0 {
        let mut out = vec![0.0; rates.len() + 1];
        out[0] = 1.0;
        return out;
    }
    let p_no = (-total * dt).exp();
    let p_any = 1.0 - p_no;
    let mut out = vec![0.0; rates.len() + 1];
    out[0] = p_no;
    for (i, &r) in rates.iter().enumerate() {
        out[i + 1] = (r / total) * p_any;
    }
    out
}

/// Binomial PMF `P(X = k)`, `k = 0..=n`, `X ~ Binomial(n, p)`.
fn binomial_pmf(n: usize, p: f64) -> Vec<f64> {
    if !(0.0..=1.0).contains(&p) {
        panic!("bad p {p}");
    }
    if n == 0 {
        return vec![1.0];
    }
    let mut out = vec![0.0; n + 1];
    if p == 0.0 {
        out[0] = 1.0;
        return out;
    }
    if p == 1.0 {
        out[n] = 1.0;
        return out;
    }
    out[0] = (1.0 - p).powi(n as i32);
    let r = p / (1.0 - p);
    for k in 0..n {
        out[k + 1] = out[k] * (n - k) as f64 * r / (k + 1) as f64;
    }
    out
}

/// Poisson-binomial PMF `P(Σ B_i = k)` for independent `B_i ~ Bernoulli(p_i)`.
fn poisson_binomial_pmf(probs: &[f64]) -> Vec<f64> {
    if probs.is_empty() {
        return vec![1.0];
    }
    let all_equal = probs.iter().all(|&p| (p - probs[0]).abs() <= 1e-15);
    if all_equal {
        return binomial_pmf(probs.len(), probs[0]);
    }
    let mut pmf = vec![1.0];
    for (i, &p) in probs.iter().enumerate() {
        if !(0.0..=1.0).contains(&p) {
            panic!("bad p[{i}] {p}");
        }
        let mut next = vec![0.0; pmf.len() + 1];
        for k in 0..pmf.len() {
            next[k] += pmf[k] * (1.0 - p);
            next[k + 1] += pmf[k] * p;
        }
        pmf = next;
    }
    pmf
}

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompartmentId {
    S,
    A,
    B,
    AB,
    R,
    D,
}

#[derive(Clone, Copy, Debug)]
pub struct TwoDiseaseParams {
    pub n: usize,
    pub initial_a: usize,
    pub initial_b: usize,
    pub initial_ab: usize,
    pub beta_a: f64,
    pub beta_b: f64,
    pub gamma_a: f64,
    pub gamma_b: f64,
    pub gamma_ab: f64,
    pub p_death_a: f64,
    pub p_death_b: f64,
    pub p_death_ab: f64,
    pub sim_t: f64,
    pub step_size: f64,
    pub seed: u32,
}

struct Person {
    id: usize,
    state: CompartmentId,
    history: Vec<(CompartmentId, f64)>,
}

impl Person {
    fn new(id: usize) -> Self {
        Person { id, state: CompartmentId::S, history: vec![(CompartmentId::S, 0.0)] }
    }
    fn transition(&mut self, to: CompartmentId, time: f64) {
        self.state = to;
        self.history.push((to, time));
    }
}

type PersonRef = Rc<RefCell<Person>>;
type CompartmentRef = Rc<RefCell<Compartment>>;
type CensusRef = Rc<RefCell<WorldCensus>>;
type RngRef = Rc<RefCell<SeededRandom>>;

#[derive(Clone, Copy, Debug, Default)]
struct GlobalCounts {
    s: f64,
    a: f64,
    b: f64,
    ab: f64,
    r: f64,
    d: f64,
    /// total alive (S + A + B + AB + R) — used as the incidence denominator.
    n: f64,
    /// S + A + B + AB + R + D, an invariant.
    total: f64,
}

// -----------------------------------------------------------------------------
// Stations
// -----------------------------------------------------------------------------

struct WorldCensus {
    id: String,
    s: CompartmentRef,
    a: CompartmentRef,
    b: CompartmentRef,
    ab: CompartmentRef,
    r: CompartmentRef,
    d: CompartmentRef,
    counts: GlobalCounts,
}

impl WorldCensus {
    fn new(
        id: impl Into<String>,
        s: CompartmentRef,
        a: CompartmentRef,
        b: CompartmentRef,
        ab: CompartmentRef,
        r: CompartmentRef,
        d: CompartmentRef,
    ) -> Self {
        WorldCensus { id: id.into(), s, a, b, ab, r, d, counts: GlobalCounts::default() }
    }
}

impl TimeSteppedStation for WorldCensus {
    fn id(&self) -> &str {
        &self.id
    }
    fn run_time_step(&mut self, _step_size: f64, _t: f64) {
        self.counts.s = self.s.borrow().people.len() as f64;
        self.counts.a = self.a.borrow().people.len() as f64;
        self.counts.b = self.b.borrow().people.len() as f64;
        self.counts.ab = self.ab.borrow().people.len() as f64;
        self.counts.r = self.r.borrow().people.len() as f64;
        self.counts.d = self.d.borrow().people.len() as f64;
        self.counts.n =
            self.counts.s + self.counts.a + self.counts.b + self.counts.ab + self.counts.r;
        self.counts.total = self.counts.n + self.counts.d;
    }
}

struct Compartment {
    id: String,
    kind: CompartmentId,
    params: TwoDiseaseParams,
    rng: RngRef,
    census: Option<CensusRef>,
    people: Vec<PersonRef>,
    pending: Vec<PersonRef>,
    dest_a: Option<CompartmentRef>,
    dest_b: Option<CompartmentRef>,
    dest_ab: Option<CompartmentRef>,
    dest_r: Option<CompartmentRef>,
    dest_d: Option<CompartmentRef>,
}

impl Compartment {
    fn new(id: impl Into<String>, kind: CompartmentId, params: TwoDiseaseParams, rng: RngRef) -> Self {
        Compartment {
            id: id.into(),
            kind,
            params,
            rng,
            census: None,
            people: Vec::new(),
            pending: Vec::new(),
            dest_a: None,
            dest_b: None,
            dest_ab: None,
            dest_r: None,
            dest_d: None,
        }
    }

    /// Called by another compartment to push a person into this one.
    fn take_item(&mut self, p: PersonRef) {
        self.pending.push(p);
    }

    /// End-of-tick commit: pending becomes part of the population.
    fn commit(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        for p in std::mem::take(&mut self.pending) {
            self.people.push(p);
        }
    }

    /// The S compartment: dual-exposure resolution (A, B, or AB).
    fn run_s(&self, people: Vec<PersonRef>, counts: &GlobalCounts, n: f64, dt: f64, time: f64) -> Vec<PersonRef> {
        let lambda_a = self.params.beta_a * (counts.a + counts.ab) / n;
        let lambda_b = self.params.beta_b * (counts.b + counts.ab) / n;
        let pmf = competing_risks(&[lambda_a, lambda_b], dt);
        let dest_a = self.dest_a.clone().unwrap();
        let dest_b = self.dest_b.clone().unwrap();
        let dest_ab = self.dest_ab.clone().unwrap();
        let mut survivors = Vec::new();
        let mut rng = self.rng.borrow_mut();
        for p in people {
            let idx = sample_categorical(&mut *rng, &pmf);
            if idx == 0 {
                survivors.push(p);
                continue;
            }
            let mut final_state = if idx == 1 { CompartmentId::A } else { CompartmentId::B };
            if idx == 1 {
                let p_b = 1.0 - (-lambda_b * dt).exp();
                if rng.next_float() < p_b {
                    final_state = CompartmentId::AB;
                }
            } else {
                let p_a = 1.0 - (-lambda_a * dt).exp();
                if rng.next_float() < p_a {
                    final_state = CompartmentId::AB;
                }
            }
            p.borrow_mut().transition(final_state, time);
            match final_state {
                CompartmentId::A => dest_a.borrow_mut().take_item(p),
                CompartmentId::B => dest_b.borrow_mut().take_item(p),
                _ => dest_ab.borrow_mut().take_item(p),
            }
        }
        survivors
    }

    /// Non-S compartments: `outcome_pmf[0]` is "stay", `k > 0` → `dests[k-1]`.
    fn apply_outcomes(
        &self,
        outcome_pmf: &[f64],
        dests: &[CompartmentRef],
        kinds: &[CompartmentId],
        time: f64,
        people: Vec<PersonRef>,
    ) -> Vec<PersonRef> {
        let mut survivors = Vec::new();
        let mut rng = self.rng.borrow_mut();
        for p in people {
            let idx = sample_categorical(&mut *rng, outcome_pmf);
            if idx == 0 {
                survivors.push(p);
                continue;
            }
            let k = idx - 1;
            p.borrow_mut().transition(kinds[k], time);
            dests[k].borrow_mut().take_item(p);
        }
        survivors
    }
}

impl TimeSteppedStation for Compartment {
    fn id(&self) -> &str {
        &self.id
    }
    fn run_time_step(&mut self, step_size: f64, t: f64) {
        if self.people.is_empty() {
            return;
        }
        let counts = self.census.as_ref().unwrap().borrow().counts;
        let n = counts.n.max(1.0);
        let time = t * step_size;
        let dt = step_size;
        match self.kind {
            CompartmentId::S => {
                let people = std::mem::take(&mut self.people);
                self.people = self.run_s(people, &counts, n, dt, time);
            }
            CompartmentId::A => {
                let lambda_ab = self.params.beta_b * (counts.b + counts.ab) / n;
                let lambda_r = self.params.gamma_a * (1.0 - self.params.p_death_a);
                let lambda_d = self.params.gamma_a * self.params.p_death_a;
                let pmf = competing_risks(&[lambda_ab, lambda_r, lambda_d], dt);
                let dests = [
                    self.dest_ab.clone().unwrap(),
                    self.dest_r.clone().unwrap(),
                    self.dest_d.clone().unwrap(),
                ];
                let people = std::mem::take(&mut self.people);
                self.people = self.apply_outcomes(
                    &pmf,
                    &dests,
                    &[CompartmentId::AB, CompartmentId::R, CompartmentId::D],
                    time,
                    people,
                );
            }
            CompartmentId::B => {
                let lambda_ab = self.params.beta_a * (counts.a + counts.ab) / n;
                let lambda_r = self.params.gamma_b * (1.0 - self.params.p_death_b);
                let lambda_d = self.params.gamma_b * self.params.p_death_b;
                let pmf = competing_risks(&[lambda_ab, lambda_r, lambda_d], dt);
                let dests = [
                    self.dest_ab.clone().unwrap(),
                    self.dest_r.clone().unwrap(),
                    self.dest_d.clone().unwrap(),
                ];
                let people = std::mem::take(&mut self.people);
                self.people = self.apply_outcomes(
                    &pmf,
                    &dests,
                    &[CompartmentId::AB, CompartmentId::R, CompartmentId::D],
                    time,
                    people,
                );
            }
            CompartmentId::AB => {
                let lambda_r = self.params.gamma_ab * (1.0 - self.params.p_death_ab);
                let lambda_d = self.params.gamma_ab * self.params.p_death_ab;
                let pmf = competing_risks(&[lambda_r, lambda_d], dt);
                let dests = [self.dest_r.clone().unwrap(), self.dest_d.clone().unwrap()];
                let people = std::mem::take(&mut self.people);
                self.people = self.apply_outcomes(
                    &pmf,
                    &dests,
                    &[CompartmentId::R, CompartmentId::D],
                    time,
                    people,
                );
            }
            CompartmentId::R | CompartmentId::D => {
                // Absorbing: no transitions.
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct TwoDiseaseTrace {
    pub t: Vec<f64>,
    pub s: Vec<f64>,
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub ab: Vec<f64>,
    pub r: Vec<f64>,
    pub d: Vec<f64>,
}

#[derive(Clone, Copy, Debug)]
pub struct FinalCounts {
    pub s: usize,
    pub a: usize,
    pub b: usize,
    pub ab: usize,
    pub r: usize,
    pub d: usize,
}

pub struct PerPerson {
    pub id: usize,
    pub final_state: CompartmentId,
    pub ever_a: bool,
    pub ever_b: bool,
    pub ever_ab: bool,
}

pub struct TwoDiseaseResult {
    pub params: TwoDiseaseParams,
    pub trace: TwoDiseaseTrace,
    pub final_counts: FinalCounts,
    pub per_person: Vec<PerPerson>,
}

/// Build the simulation graph and run it.
pub fn run_two_disease(params: &TwoDiseaseParams) -> TwoDiseaseResult {
    with_seed(params.seed, |global_rng| {
        let rng: RngRef = Rc::new(RefCell::new(mulberry32(params.seed)));

        let s: CompartmentRef =
            Rc::new(RefCell::new(Compartment::new("S", CompartmentId::S, *params, rng.clone())));
        let a: CompartmentRef =
            Rc::new(RefCell::new(Compartment::new("A", CompartmentId::A, *params, rng.clone())));
        let b: CompartmentRef =
            Rc::new(RefCell::new(Compartment::new("B", CompartmentId::B, *params, rng.clone())));
        let ab: CompartmentRef =
            Rc::new(RefCell::new(Compartment::new("AB", CompartmentId::AB, *params, rng.clone())));
        let r: CompartmentRef =
            Rc::new(RefCell::new(Compartment::new("R", CompartmentId::R, *params, rng.clone())));
        let d: CompartmentRef =
            Rc::new(RefCell::new(Compartment::new("D", CompartmentId::D, *params, rng.clone())));

        let census: CensusRef = Rc::new(RefCell::new(WorldCensus::new(
            "census",
            s.clone(),
            a.clone(),
            b.clone(),
            ab.clone(),
            r.clone(),
            d.clone(),
        )));

        for c in [&s, &a, &b, &ab, &r, &d] {
            let mut cm = c.borrow_mut();
            cm.census = Some(census.clone());
            cm.dest_a = Some(a.clone());
            cm.dest_b = Some(b.clone());
            cm.dest_ab = Some(ab.clone());
            cm.dest_r = Some(r.clone());
            cm.dest_d = Some(d.clone());
        }

        // Seed populations.
        let mut next_id = 0usize;
        let init_s = params.n as i64 - params.initial_a as i64 - params.initial_b as i64 - params.initial_ab as i64;
        if init_s < 0 {
            panic!("initial A + B + AB exceed N");
        }
        let init_s = init_s as usize;
        let mut all_people: Vec<PersonRef> = Vec::new();
        for _ in 0..init_s {
            let p: PersonRef = Rc::new(RefCell::new(Person::new(next_id)));
            next_id += 1;
            s.borrow_mut().people.push(p.clone());
            all_people.push(p);
        }
        for _ in 0..params.initial_a {
            let p: PersonRef = Rc::new(RefCell::new(Person::new(next_id)));
            next_id += 1;
            p.borrow_mut().transition(CompartmentId::A, 0.0);
            a.borrow_mut().people.push(p.clone());
            all_people.push(p);
        }
        for _ in 0..params.initial_b {
            let p: PersonRef = Rc::new(RefCell::new(Person::new(next_id)));
            next_id += 1;
            p.borrow_mut().transition(CompartmentId::B, 0.0);
            b.borrow_mut().people.push(p.clone());
            all_people.push(p);
        }
        for _ in 0..params.initial_ab {
            let p: PersonRef = Rc::new(RefCell::new(Person::new(next_id)));
            next_id += 1;
            p.borrow_mut().transition(CompartmentId::AB, 0.0);
            ab.borrow_mut().people.push(p.clone());
            all_people.push(p);
        }

        let compartments = vec![s.clone(), a.clone(), b.clone(), ab.clone(), r.clone(), d.clone()];
        let mut trace = TwoDiseaseTrace::default();

        let n_steps = (params.sim_t / params.step_size).round() as usize;
        for t in 0..n_steps {
            // 1. Census reads frozen counts.
            census.borrow_mut().run_time_step(params.step_size, t as f64);
            // 2. Compartments process in shuffled order using frozen counts.
            let mut order = compartments.clone();
            fisher_yates_shuffle(&mut order, global_rng);
            for c in &order {
                c.borrow_mut().run_time_step(params.step_size, t as f64);
            }
            // 3. Commit pending.
            for c in &compartments {
                c.borrow_mut().commit();
            }
            // 4. Record trace at integer time steps.
            let time = (t as f64 + 1.0) * params.step_size;
            trace.t.push(time);
            trace.s.push(s.borrow().people.len() as f64);
            trace.a.push(a.borrow().people.len() as f64);
            trace.b.push(b.borrow().people.len() as f64);
            trace.ab.push(ab.borrow().people.len() as f64);
            trace.r.push(r.borrow().people.len() as f64);
            trace.d.push(d.borrow().people.len() as f64);
        }

        let final_counts = FinalCounts {
            s: s.borrow().people.len(),
            a: a.borrow().people.len(),
            b: b.borrow().people.len(),
            ab: ab.borrow().people.len(),
            r: r.borrow().people.len(),
            d: d.borrow().people.len(),
        };

        let per_person = all_people
            .iter()
            .map(|p| {
                let pp = p.borrow();
                let ever_a = pp
                    .history
                    .iter()
                    .any(|(st, _)| *st == CompartmentId::A || *st == CompartmentId::AB);
                let ever_b = pp
                    .history
                    .iter()
                    .any(|(st, _)| *st == CompartmentId::B || *st == CompartmentId::AB);
                let ever_ab = pp.history.iter().any(|(st, _)| *st == CompartmentId::AB);
                PerPerson { id: pp.id, final_state: pp.state, ever_a, ever_b, ever_ab }
            })
            .collect();

        TwoDiseaseResult { params: *params, trace, final_counts, per_person }
    })
}

// -----------------------------------------------------------------------------
// CLI helpers
// -----------------------------------------------------------------------------

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn std_dev(xs: &[f64]) -> f64 {
    let m = mean(xs);
    let denom = if xs.len() > 1 { xs.len() - 1 } else { 1 };
    (xs.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / denom as f64).sqrt()
}

/// `JSON.stringify` of a finite number: integers without trailing `.0`.
fn jn(x: f64) -> String {
    if x.is_finite() && x.fract() == 0.0 {
        format!("{}", x as i64)
    } else {
        format!("{}", x)
    }
}

fn arr(xs: &[f64]) -> String {
    xs.iter().map(|v| jn(*v)).collect::<Vec<_>>().join(",")
}

fn trace_json(tr: &TwoDiseaseTrace) -> String {
    format!(
        "{{\"t\":[{}],\"S\":[{}],\"A\":[{}],\"B\":[{}],\"AB\":[{}],\"R\":[{}],\"D\":[{}]}}",
        arr(&tr.t),
        arr(&tr.s),
        arr(&tr.a),
        arr(&tr.b),
        arr(&tr.ab),
        arr(&tr.r),
        arr(&tr.d)
    )
}

/// Entry point (`main()` in the TS source).
pub fn run() {
    let params = TwoDiseaseParams {
        n: env_usize("N", 1000),
        initial_a: env_usize("INIT_A", 5),
        initial_b: env_usize("INIT_B", 5),
        initial_ab: env_usize("INIT_AB", 0),
        beta_a: env_f64("BETA_A", 0.5),
        beta_b: env_f64("BETA_B", 0.4),
        gamma_a: env_f64("GAMMA_A", 1.0 / 7.0),
        gamma_b: env_f64("GAMMA_B", 1.0 / 10.0),
        gamma_ab: env_f64("GAMMA_AB", 1.0 / 8.0),
        p_death_a: env_f64("P_D_A", 0.40),
        p_death_b: env_f64("P_D_B", 0.60),
        p_death_ab: env_f64("P_D_AB", 0.50),
        sim_t: env_f64("SIM_T", 200.0),
        step_size: env_f64("STEPSIZE", 0.1),
        seed: env_usize("SEED", 1) as u32,
    };
    println!("# Two-disease epidemic");
    println!(
        "#   N={} initial A={} B={} AB={}",
        params.n, params.initial_a, params.initial_b, params.initial_ab
    );
    println!("#   β_A={} β_B={}", jn(params.beta_a), jn(params.beta_b));
    println!(
        "#   γ_A={:.4} γ_B={:.4} γ_AB={:.4}",
        params.gamma_a, params.gamma_b, params.gamma_ab
    );
    println!(
        "#   p_death A={} B={} AB={}",
        jn(params.p_death_a),
        jn(params.p_death_b),
        jn(params.p_death_ab)
    );
    println!(
        "#   simT={} dt={} seed={}",
        jn(params.sim_t),
        jn(params.step_size),
        params.seed
    );

    // Multiple seeds for ensemble.
    let reps = env_usize("REPS", 30);
    let mut traces: Vec<TwoDiseaseTrace> = Vec::new();
    let mut final_deaths: Vec<f64> = Vec::new();
    let mut final_recovered: Vec<f64> = Vec::new();
    let mut fraction_ever_ab: Vec<f64> = Vec::new();
    let mut per_person_death_flags: Vec<Vec<f64>> = Vec::new();
    let t0 = Instant::now();
    for rep in 0..reps {
        let mut cfg = params;
        cfg.seed = params.seed + rep as u32;
        let result = run_two_disease(&cfg);
        final_deaths.push(result.final_counts.d as f64);
        final_recovered.push(result.final_counts.r as f64);
        let ever_ab = result.per_person.iter().filter(|p| p.ever_ab).count();
        fraction_ever_ab.push(ever_ab as f64 / params.n as f64);
        per_person_death_flags.push(
            result
                .per_person
                .iter()
                .map(|p| if p.final_state == CompartmentId::D { 1.0 } else { 0.0 })
                .collect(),
        );
        traces.push(result.trace);
    }
    let ms = t0.elapsed().as_millis();
    println!();
    println!("# {} replications, {} ms total", reps, ms);
    println!(
        "#   final D : mean={:.2}  std={:.2}",
        mean(&final_deaths),
        std_dev(&final_deaths)
    );
    println!(
        "#   final R : mean={:.2}  std={:.2}",
        mean(&final_recovered),
        std_dev(&final_recovered)
    );
    println!(
        "#   ever AB : mean={:.2}%  std={:.2}pp",
        mean(&fraction_ever_ab) * 100.0,
        std_dev(&fraction_ever_ab) * 100.0
    );

    // Conservation check.
    let r0 = run_two_disease(&params);
    let f = r0.final_counts;
    println!(
        "#   conservation: S+A+B+AB+R+D = {}, N = {}",
        f.s + f.a + f.b + f.ab + f.r + f.d,
        params.n
    );

    // Poisson-binomial cross-check.
    if reps >= 5 {
        let mut per_person_probs = vec![0.0; params.n];
        for flags in &per_person_death_flags {
            for i in 0..params.n {
                per_person_probs[i] += flags[i] / reps as f64;
            }
        }
        let pb = poisson_binomial_pmf(&per_person_probs);
        let pb_mean = mean_from_pmf(&pb);
        let pb_std = variance_from_pmf(&pb).sqrt();
        println!();
        println!("#   Poisson-binomial cross-check (assumes per-person deaths are ~independent):");
        println!(
            "#     simulation:  E[D] = {:.2}  std = {:.2}",
            mean(&final_deaths),
            std_dev(&final_deaths)
        );
        println!("#     PB model  :  E[D] = {:.2}  std = {:.2}", pb_mean, pb_std);
    }

    // Dump for downstream analysis.
    let out_dir = std::path::Path::new("out");
    let _ = std::fs::create_dir_all(out_dir);
    let t_len = traces[0].t.len();
    let mut mean_trace = TwoDiseaseTrace { t: traces[0].t.clone(), ..Default::default() };
    for i in 0..t_len {
        mean_trace.s.push(mean(&traces.iter().map(|tr| tr.s[i]).collect::<Vec<_>>()));
        mean_trace.a.push(mean(&traces.iter().map(|tr| tr.a[i]).collect::<Vec<_>>()));
        mean_trace.b.push(mean(&traces.iter().map(|tr| tr.b[i]).collect::<Vec<_>>()));
        mean_trace.ab.push(mean(&traces.iter().map(|tr| tr.ab[i]).collect::<Vec<_>>()));
        mean_trace.r.push(mean(&traces.iter().map(|tr| tr.r[i]).collect::<Vec<_>>()));
        mean_trace.d.push(mean(&traces.iter().map(|tr| tr.d[i]).collect::<Vec<_>>()));
    }
    let out_path = out_dir.join("two-disease-framework.json");
    let params_json = format!(
        concat!(
            "{{\"N\":{},\"initialA\":{},\"initialB\":{},\"initialAB\":{},",
            "\"beta_A\":{},\"beta_B\":{},\"gamma_A\":{},\"gamma_B\":{},\"gamma_AB\":{},",
            "\"p_death_A\":{},\"p_death_B\":{},\"p_death_AB\":{},",
            "\"simT\":{},\"stepSize\":{},\"seed\":{}}}"
        ),
        params.n,
        params.initial_a,
        params.initial_b,
        params.initial_ab,
        jn(params.beta_a),
        jn(params.beta_b),
        jn(params.gamma_a),
        jn(params.gamma_b),
        jn(params.gamma_ab),
        jn(params.p_death_a),
        jn(params.p_death_b),
        jn(params.p_death_ab),
        jn(params.sim_t),
        jn(params.step_size),
        params.seed
    );
    let traces_json = traces.iter().map(trace_json).collect::<Vec<_>>().join(",");
    let json = format!(
        concat!(
            "{{\"params\":{},\"reps\":{},\"finalDeaths\":[{}],\"finalRecovered\":[{}],",
            "\"fractionEverAB\":[{}],\"meanTrace\":{},\"traces\":[{}]}}"
        ),
        params_json,
        reps,
        arr(&final_deaths),
        arr(&final_recovered),
        arr(&fraction_ever_ab),
        trace_json(&mean_trace),
        traces_json
    );
    let _ = std::fs::write(&out_path, json);
    println!("# wrote {}", out_path.display());

    // ----- Optional animation -------------------------------------------------
    if std::env::var("ANIMATE").as_deref() == Ok("1") {
        use crate::des::animation::frame_recorder::{FrameRecorder, FrameRecorderOpts};
        use crate::des::animation::scenes::two_disease_scene as scene;

        let which = (reps - 1).min(env_usize("ANIMATE_REP", 0));
        let trace = &traces[which];
        let frames_path = out_dir.join("two-disease.frames.jsonl");
        let html_path = out_dir.join("two-disease.html");
        let record_every = (trace.t.len() / 600).max(1) as f64;
        let mut rec = FrameRecorder::new(FrameRecorderOpts {
            frames_path: frames_path.to_string_lossy().into_owned(),
            html_path: Some(html_path.to_string_lossy().into_owned()),
            width: scene::STAGE_W,
            height: scene::STAGE_H,
            fps: Some(30.0),
            title: Some("Two-disease epidemic — framework simulation".to_string()),
            subtitle: Some(format!(
                "N={}  β_A={}  β_B={}  γ_AB={}  p_d_AB={}  dt={}  rep={}",
                params.n,
                jn(params.beta_a),
                jn(params.beta_b),
                jn(params.gamma_ab),
                jn(params.p_death_ab),
                jn(params.step_size),
                which
            )),
            live_tick_line: Some(true),
            record_every_ticks: Some(record_every),
            ..Default::default()
        })
        .expect("create frame recorder");

        let n_f = params.n as f64;
        for i in 0..trace.t.len() {
            let counts = scene::CompartmentCounts {
                s: trace.s[i],
                a: trace.a[i],
                b: trace.b[i],
                ab: trace.ab[i],
                r: trace.r[i],
                d: trace.d[i],
            };
            let t_i = trace.t[i];
            let i_f = i as f64;
            rec.frame(t_i, i_f, || scene::build_frame(t_i, i_f, &counts, n_f));
        }
        let scene_trace = scene::TwoDiseaseTrace {
            t: trace.t.clone(),
            s: trace.s.clone(),
            a: trace.a.clone(),
            b: trace.b.clone(),
            ab: trace.ab.clone(),
            r: trace.r.clone(),
            d: trace.d.clone(),
        };
        rec.set_charts(vec![scene::build_compartment_chart(&scene_trace, n_f)]);
        rec.finish().expect("finish recorder");
        println!("# wrote {} ({} frames)", html_path.display(), rec.get_frame_count());
    }
}
