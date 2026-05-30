//! Port of `src/des/main-contact-seir.ts`.
//!
//! Contact-based SEIR: explicit pairwise / triplet interactions vs mass-action
//! mean field. Defines the model AND runs it.
//!
//! `mulberry32`/`withSeed` → `crate::des::general::prng`; `sampleGamma`/
//! `samplePoisson` → `crate::des::general::random_variables`; `process.env.*` →
//! `std::env::var`; `fs` → `std::fs`. The JSON artifact is built with
//! `crate::des::observability::logger::JsonValue` (no `serde`).
//!
//! PORT NOTE: the optional `ANIMATE=1` branch needs
//! `animation/scenes/contact-seir-scene`, which is not ported; the animation is
//! stubbed with a console note.

#![allow(dead_code)]

use crate::des::general::prng::{mulberry32, with_seed};
use crate::des::general::random_variables::{sample_gamma, sample_poisson};
use crate::des::observability::logger::JsonValue;
use crate::des::shared::capabilities::{RandomSource, SeededRandom};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    MassAction,
    Pairwise,
    Triplet,
}

impl Kernel {
    fn label(self) -> &'static str {
        match self {
            Kernel::MassAction => "mass-action",
            Kernel::Pairwise => "pairwise",
            Kernel::Triplet => "triplet",
        }
    }
    fn parse(s: &str) -> Kernel {
        match s {
            "mass-action" => Kernel::MassAction,
            "triplet" => Kernel::Triplet,
            _ => Kernel::Pairwise,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SeirState {
    S,
    E,
    I,
    R,
}

#[derive(Clone)]
pub struct ContactSEIRParams {
    pub n: usize,
    pub initial_i: usize,
    pub contact_rate: f64,
    pub contact_rate_cv: f64,
    pub p_transmit: f64,
    pub sigma: f64,
    pub gamma: f64,
    pub sim_t: f64,
    pub step_size: f64,
    pub seed: u32,
    pub kernel: Kernel,
}

#[derive(Clone)]
pub struct Person {
    pub id: i64,
    pub state: SeirState,
    pub c: f64,
    pub t_exposed: f64,
    pub t_infectious: f64,
    pub t_recovered: f64,
    pub offspring: f64,
    pub infected_by: i64,
}

#[derive(Clone, Default)]
pub struct Trace {
    pub t: Vec<f64>,
    pub s: Vec<i64>,
    pub e: Vec<i64>,
    pub i: Vec<i64>,
    pub r: Vec<i64>,
}

#[derive(Clone)]
pub struct PerPerson {
    pub id: i64,
    pub c: f64,
    pub offspring: f64,
    pub ever: bool,
    pub infected_by: i64,
}

pub struct ContactSEIRResult {
    pub params: ContactSEIRParams,
    pub trace: Trace,
    pub total_contacts: i64,
    pub total_transmissions: i64,
    pub final_attack_rate: f64,
    pub r0_empirical: f64,
    pub r0_index_only: f64,
    pub r0_second_gen: f64,
    pub per_person: Vec<PerPerson>,
}

struct Population {
    people: Vec<Person>,
    rng: SeededRandom,
    params: ContactSEIRParams,
    total_contacts: i64,
    total_transmissions: i64,
}

impl Population {
    fn new(params: ContactSEIRParams) -> Self {
        let mut rng = mulberry32(params.seed);
        let mut people = Vec::with_capacity(params.n);
        let cv2 = params.contact_rate_cv * params.contact_rate_cv;
        let (shape, scale) = (1.0 / cv2, params.contact_rate * cv2);
        for i in 0..params.n {
            let c = if params.contact_rate_cv == 0.0 {
                params.contact_rate
            } else {
                sample_gamma(&mut rng, shape, scale)
            };
            people.push(Person {
                id: i as i64,
                state: SeirState::S,
                c,
                t_exposed: f64::NAN,
                t_infectious: f64::NAN,
                t_recovered: f64::NAN,
                offspring: 0.0,
                infected_by: -2,
            });
        }
        for i in 0..params.initial_i {
            let p = &mut people[i];
            p.state = SeirState::I;
            p.t_exposed = 0.0;
            p.t_infectious = 0.0;
            p.infected_by = -1;
        }
        Population {
            people,
            rng,
            params,
            total_contacts: 0,
            total_transmissions: 0,
        }
    }

    fn run_time_step(&mut self, t: i64) {
        let dt = self.params.step_size;
        let state_now: Vec<SeirState> = self.people.iter().map(|p| p.state).collect();
        for i in 0..self.people.len() {
            if state_now[i] == SeirState::E {
                if self.rng.next_float() < 1.0 - (-self.params.sigma * dt).exp() {
                    self.people[i].state = SeirState::I;
                    self.people[i].t_infectious = (t + 1) as f64 * dt;
                }
            } else if state_now[i] == SeirState::I
                && self.rng.next_float() < 1.0 - (-self.params.gamma * dt).exp()
            {
                self.people[i].state = SeirState::R;
                self.people[i].t_recovered = (t + 1) as f64 * dt;
            }
        }

        let is_i: Vec<bool> = state_now.iter().map(|s| *s == SeirState::I).collect();
        match self.params.kernel {
            Kernel::MassAction => self.run_mass_action_kernel(t, &is_i),
            Kernel::Pairwise => self.run_pairwise_kernel(t, &state_now),
            Kernel::Triplet => self.run_triplet_kernel(t, &is_i),
        }
    }

    fn run_mass_action_kernel(&mut self, t: i64, is_i: &[bool]) {
        let n = self.people.len();
        let infectious = is_i.iter().filter(|x| **x).count();
        if infectious == 0 {
            return;
        }
        let beta = self.params.contact_rate * self.params.p_transmit;
        let lambda = beta * infectious as f64 / n as f64;
        let dt = self.params.step_size;
        for i in 0..self.people.len() {
            if self.people[i].state != SeirState::S {
                continue;
            }
            let lambda_i = (self.people[i].c / self.params.contact_rate) * lambda;
            let pi = 1.0 - (-lambda_i * dt).exp();
            if self.rng.next_float() < pi {
                let infector_idx = pick_random_infectious(&mut self.rng, is_i);
                if infector_idx >= 0 {
                    self.people[infector_idx as usize].offspring += 1.0;
                    self.people[i].infected_by = infector_idx;
                }
                self.people[i].state = SeirState::E;
                self.people[i].t_exposed = (t + 1) as f64 * dt;
                self.total_transmissions += 1;
            }
        }
    }

    fn run_pairwise_kernel(&mut self, t: i64, state_snap: &[SeirState]) {
        let n = self.people.len();
        let dt = self.params.step_size;
        for i in 0..self.people.len() {
            let k = sample_poisson(&mut self.rng, self.people[i].c * 0.5 * dt) as i64;
            self.total_contacts += k;
            let my_state = state_snap[i];
            for _ in 0..k {
                let mut other = (self.rng.next_float() * n as f64).floor() as usize;
                if other == i {
                    other = (other + 1) % n;
                }
                let part_state = state_snap[other];
                if my_state == SeirState::S
                    && part_state == SeirState::I
                    && self.rng.next_float() < self.params.p_transmit
                {
                    if self.people[i].state == SeirState::S {
                        self.people[other].offspring += 1.0;
                        self.people[i].state = SeirState::E;
                        self.people[i].t_exposed = (t + 1) as f64 * dt;
                        self.people[i].infected_by = other as i64;
                        self.total_transmissions += 1;
                    }
                } else if my_state == SeirState::I
                    && part_state == SeirState::S
                    && self.rng.next_float() < self.params.p_transmit
                    && self.people[other].state == SeirState::S
                {
                    self.people[i].offspring += 1.0;
                    self.people[other].state = SeirState::E;
                    self.people[other].t_exposed = (t + 1) as f64 * dt;
                    self.people[other].infected_by = i as i64;
                    self.total_transmissions += 1;
                }
            }
        }
    }

    fn run_triplet_kernel(&mut self, t: i64, is_i: &[bool]) {
        let n = self.people.len();
        let dt = self.params.step_size;
        for i in 0..self.people.len() {
            if self.people[i].state != SeirState::S {
                continue;
            }
            let k = sample_poisson(&mut self.rng, self.people[i].c * dt) as i64;
            self.total_contacts += k;
            for _ in 0..k {
                let mut a = (self.rng.next_float() * n as f64).floor() as usize;
                if a == i {
                    a = (a + 1) % n;
                }
                let mut b = (self.rng.next_float() * n as f64).floor() as usize;
                if b == i || b == a {
                    b = (b + 2) % n;
                }
                if is_i[a] && is_i[b] && self.rng.next_float() < self.params.p_transmit {
                    self.people[a].offspring += 0.5;
                    self.people[b].offspring += 0.5;
                    self.people[i].state = SeirState::E;
                    self.people[i].t_exposed = (t + 1) as f64 * dt;
                    self.people[i].infected_by = a as i64;
                    self.total_transmissions += 1;
                    break;
                }
            }
        }
    }
}

fn pick_random_infectious(rng: &mut SeededRandom, is_i: &[bool]) -> i64 {
    let count = is_i.iter().filter(|x| **x).count();
    if count == 0 {
        return -1;
    }
    let mut pick = (rng.next_float() * count as f64).floor() as i64;
    for (i, x) in is_i.iter().enumerate() {
        if *x {
            if pick == 0 {
                return i as i64;
            }
            pick -= 1;
        }
    }
    -1
}

/// `runContactSEIR(params, onTick?)`.
pub fn run_contact_seir(
    params: &ContactSEIRParams,
    mut on_tick: Option<&mut dyn FnMut(&[Person], f64, i64, i64, i64)>,
) -> ContactSEIRResult {
    let params = params.clone();
    with_seed(params.seed, |_global| {
        let mut pop = Population::new(params.clone());
        let n_ticks = (params.sim_t / params.step_size).round() as i64;
        let mut trace = Trace::default();
        for t in 0..n_ticks {
            pop.run_time_step(t);
            if let Some(cb) = on_tick.as_deref_mut() {
                cb(
                    &pop.people,
                    (t + 1) as f64 * params.step_size,
                    t + 1,
                    pop.total_contacts,
                    pop.total_transmissions,
                );
            }
            let (mut s, mut e, mut i, mut r) = (0i64, 0i64, 0i64, 0i64);
            for p in &pop.people {
                match p.state {
                    SeirState::S => s += 1,
                    SeirState::E => e += 1,
                    SeirState::I => i += 1,
                    SeirState::R => r += 1,
                }
            }
            trace.t.push((t + 1) as f64 * params.step_size);
            trace.s.push(s);
            trace.e.push(e);
            trace.i.push(i);
            trace.r.push(r);
        }

        let final_attack_rate =
            (params.n as f64 - *trace.s.last().unwrap_or(&0) as f64) / params.n as f64;

        let ever_infectious: Vec<&Person> = pop
            .people
            .iter()
            .filter(|p| !p.t_infectious.is_nan())
            .collect();
        let r0_empirical = if ever_infectious.is_empty() {
            0.0
        } else {
            ever_infectious.iter().map(|p| p.offspring).sum::<f64>() / ever_infectious.len() as f64
        };

        let index_cases: Vec<&Person> = pop.people.iter().filter(|p| p.infected_by == -1).collect();
        let index_ids: std::collections::HashSet<i64> = index_cases.iter().map(|p| p.id).collect();
        let r0_index_only = if index_cases.is_empty() {
            0.0
        } else {
            index_cases.iter().map(|p| p.offspring).sum::<f64>() / index_cases.len() as f64
        };

        let gen2: Vec<&Person> = pop
            .people
            .iter()
            .filter(|p| index_ids.contains(&p.infected_by))
            .collect();
        let r0_second_gen = if gen2.is_empty() {
            0.0
        } else {
            gen2.iter().map(|p| p.offspring).sum::<f64>() / gen2.len() as f64
        };

        let per_person: Vec<PerPerson> = pop
            .people
            .iter()
            .map(|p| PerPerson {
                id: p.id,
                c: p.c,
                offspring: p.offspring,
                ever: !p.t_exposed.is_nan() || p.infected_by == -1,
                infected_by: p.infected_by,
            })
            .collect();

        ContactSEIRResult {
            params: params.clone(),
            trace,
            total_contacts: pop.total_contacts,
            total_transmissions: pop.total_transmissions,
            final_attack_rate,
            r0_empirical,
            r0_index_only,
            r0_second_gen,
            per_person,
        }
    })
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn std_dev(xs: &[f64]) -> f64 {
    let m = mean(xs);
    let denom = (xs.len().max(2) - 1) as f64;
    (xs.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / denom).sqrt()
}

fn jnum(n: f64) -> JsonValue {
    JsonValue::Number(n)
}
fn jstr(s: &str) -> JsonValue {
    JsonValue::String(s.to_string())
}
fn jarr_f64(xs: &[f64]) -> JsonValue {
    JsonValue::Array(xs.iter().map(|x| jnum(*x)).collect())
}
fn jarr_i64(xs: &[i64]) -> JsonValue {
    JsonValue::Array(xs.iter().map(|x| jnum(*x as f64)).collect())
}
fn jobj(v: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(v.into_iter().map(|(k, val)| (k.to_string(), val)).collect())
}

fn params_json(p: &ContactSEIRParams) -> JsonValue {
    jobj(vec![
        ("N", jnum(p.n as f64)),
        ("initialI", jnum(p.initial_i as f64)),
        ("contactRate", jnum(p.contact_rate)),
        ("contactRateCV", jnum(p.contact_rate_cv)),
        ("pTransmit", jnum(p.p_transmit)),
        ("sigma", jnum(p.sigma)),
        ("gamma", jnum(p.gamma)),
        ("simT", jnum(p.sim_t)),
        ("stepSize", jnum(p.step_size)),
        ("seed", jnum(p.seed as f64)),
        ("kernel", jstr(p.kernel.label())),
    ])
}

fn trace_json(tr: &Trace) -> JsonValue {
    jobj(vec![
        ("t", jarr_f64(&tr.t)),
        ("S", jarr_i64(&tr.s)),
        ("E", jarr_i64(&tr.e)),
        ("I", jarr_i64(&tr.i)),
        ("R", jarr_i64(&tr.r)),
    ])
}

/// Entry point (TS top-level `main`).
pub fn run() {
    let params = ContactSEIRParams {
        n: env_f64("N", 2000.0) as usize,
        initial_i: env_f64("INITIAL_I", 5.0) as usize,
        contact_rate: env_f64("CONTACT_RATE", 6.0),
        contact_rate_cv: env_f64("CONTACT_CV", 0.0),
        p_transmit: env_f64("P_TRANSMIT", 0.05),
        sigma: env_f64("SIGMA", 1.0 / 5.2),
        gamma: env_f64("GAMMA", 1.0 / 7.0),
        sim_t: env_f64("SIM_T", 120.0),
        step_size: env_f64("STEPSIZE", 0.1),
        seed: env_f64("SEED", 1.0) as u32,
        kernel: Kernel::parse(&std::env::var("KERNEL").unwrap_or_else(|_| "pairwise".into())),
    };
    let reps = env_f64("REPS", 1.0) as usize;

    println!(
        "# Contact-SEIR: kernel={}, N={}, c={} (cv={}), p={}, σ={:.3}, γ={:.3}",
        params.kernel.label(),
        params.n,
        params.contact_rate,
        params.contact_rate_cv,
        params.p_transmit,
        params.sigma,
        params.gamma
    );
    println!(
        "#   simT={}, dt={}, reps={}",
        params.sim_t, params.step_size, reps
    );

    let beta_theory = params.contact_rate * params.p_transmit;
    let r0_theory = beta_theory / params.gamma;
    let r0_het_factor = 1.0 + params.contact_rate_cv * params.contact_rate_cv;
    println!(
        "#   β = {:.3},  R₀(homogeneous) ≈ {:.2},  (1+CV²) factor = {:.2},  R₀(heterogeneous) ≈ {:.2}",
        beta_theory,
        r0_theory,
        r0_het_factor,
        r0_theory * r0_het_factor
    );

    let t0 = std::time::Instant::now();
    let mut results: Vec<ContactSEIRResult> = Vec::new();
    for r in 0..reps {
        let mut p = params.clone();
        p.seed = params.seed + r as u32;
        results.push(run_contact_seir(&p, None));
    }
    let ms = t0.elapsed().as_millis();

    let attack_rates: Vec<f64> = results.iter().map(|r| r.final_attack_rate).collect();
    let r0s: Vec<f64> = results.iter().map(|r| r.r0_empirical).collect();
    let r0idx: Vec<f64> = results.iter().map(|r| r.r0_index_only).collect();
    let total_c: Vec<f64> = results.iter().map(|r| r.total_contacts as f64).collect();
    let total_t: Vec<f64> = results
        .iter()
        .map(|r| r.total_transmissions as f64)
        .collect();

    println!();
    println!("# {reps} replication(s) in {ms} ms");
    println!(
        "#   final attack rate : mean={:.2}%   std={:.2}pp",
        mean(&attack_rates) * 100.0,
        std_dev(&attack_rates) * 100.0
    );
    println!(
        "#   R₀ (all infectives): mean={:.2}   std={:.2}",
        mean(&r0s),
        std_dev(&r0s)
    );
    println!(
        "#   R₀ (index cases)  : mean={:.2}   std={:.2}",
        mean(&r0idx),
        std_dev(&r0idx)
    );
    if params.kernel != Kernel::MassAction {
        println!(
            "#   total contacts    : mean={:.0}   std={:.0}",
            mean(&total_c),
            std_dev(&total_c)
        );
        println!(
            "#   total transmissions: mean={:.0}   std={:.0}",
            mean(&total_t),
            std_dev(&total_t)
        );
    }

    let _ = std::fs::create_dir_all("out");
    let out_path = format!("out/contact-seir-{}.json", params.kernel.label());

    // Mean trajectory across replications.
    let cap = results[0].trace.t.len();
    let mut mean_s = vec![0.0f64; cap];
    let mut mean_e = vec![0.0f64; cap];
    let mut mean_i = vec![0.0f64; cap];
    let mut mean_r = vec![0.0f64; cap];
    for res in &results {
        for k in 0..cap {
            mean_s[k] += res.trace.s[k] as f64 / reps as f64;
            mean_e[k] += res.trace.e[k] as f64 / reps as f64;
            mean_i[k] += res.trace.i[k] as f64 / reps as f64;
            mean_r[k] += res.trace.r[k] as f64 / reps as f64;
        }
    }
    let mean_trace = jobj(vec![
        ("t", jarr_f64(&results[0].trace.t)),
        ("S", jarr_f64(&mean_s)),
        ("E", jarr_f64(&mean_e)),
        ("I", jarr_f64(&mean_i)),
        ("R", jarr_f64(&mean_r)),
    ]);

    let per_person_json = JsonValue::Array(
        results[0]
            .per_person
            .iter()
            .map(|p| {
                jobj(vec![
                    ("id", jnum(p.id as f64)),
                    ("c", jnum(p.c)),
                    ("offspring", jnum(p.offspring)),
                    ("ever", JsonValue::Bool(p.ever)),
                    ("infectedBy", jnum(p.infected_by as f64)),
                ])
            })
            .collect(),
    );

    let doc = jobj(vec![
        ("params", params_json(&params)),
        ("reps", jnum(reps as f64)),
        ("meanTrace", mean_trace),
        ("finalAttackRates", jarr_f64(&attack_rates)),
        ("R0_empirical", jarr_f64(&r0s)),
        ("R0_indexOnly", jarr_f64(&r0idx)),
        (
            "traces",
            JsonValue::Array(results.iter().map(|r| trace_json(&r.trace)).collect()),
        ),
        ("perPerson", per_person_json),
    ]);
    let _ = std::fs::write(&out_path, doc.to_string());
    println!("# wrote {out_path}");

    if std::env::var("ANIMATE").as_deref() == Ok("1") {
        // PORT NOTE: contact-seir animation scene not ported; see header.
        println!("# (ANIMATE=1 requested but contact-seir scene not ported — animation skipped)");
    }
}
