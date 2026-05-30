//! Port of `src/des/main.ts` (module `des::main`).
//!
//! Top-level wiring of the queueing-network DES: an `EntitySource` (`A`) feeds a
//! chain of `EntityProcessor`s (`B`→`C`→`D`) into an `EntitySink` (`E`). After a
//! 20 000-tick warm-up the sources are switched off and a second audited loop
//! runs to 20 000, then a finalize pass dumps each node's computed properties.
//!
//! ## Conversion notes (faithful to the TS shape)
//!
//!   * The top-level `run` closure + its invocation → [`run`] (a `pub fn`, NOT
//!     `fn main`; this is a library crate). The `setImmediate` every-100-ticks
//!     yield is a Node I/O trick with no behavioural effect, so the two
//!     mutually-recursive schedulers collapse to plain loops with identical
//!     bounds.
//!   * The heterogeneous `Map<string, Entity<any>>` + uniform `addOutConnection`
//!     is modelled with a [`Node`] enum, exactly as the sister ports
//!     `main_markov` / `main_epidemic`.
//!   * `mathjs.BigNumber` step size (`des.getStepSize()`) → [`Decimal`] via
//!     [`get_step_size`]; `des.bumpTimeAccruedByTimeStep` / `doAudit` reuse the
//!     ported global clock + audit.
//!   * `EntitySink('E', new PoissonRandomVariable(), {})`: the ported
//!     [`EntitySink`] constructor takes only an id, so the rv/opts are dropped.
//!
//! PORT NOTE: `(global as any).turnOffSources = true` has no analogue — only the
//! per-source `turn_off_after_count` guard remains (here `-1`, i.e. the source
//! never auto-stops, matching the TS literal).
//!
//! PORT NOTE: the TS RVs used ambient `Math.random`; per the capability rule
//! each `UniformRandomVariable` here receives a seeded [`SeededRandom`].
//!
//! PORT NOTE: `uuid` / `safe-stringify` / `mathjs` Node imports are dropped.

#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::des::entity_processing::processing::EntityProcessor;
use crate::des::entity_routing::output_routing_policy::OutputRoutingPolicy;
use crate::des::entity_sink::sink::EntitySink;
use crate::des::entity_source::source::EntitySource;
use crate::des::general::do_audit::do_audit;
use crate::des::general::general::fisher_yates_shuffle;
use crate::des::general::time_accrued::{bump_time_accrued_by_time_step, get_step_size};
use crate::des::observers::program_observer::ProgramObserver;
use crate::des::r#abstract::interfaces::{EntityGraphData, HasInput, HasOutput};
use crate::des::r#abstract::r#abstract::{Entity, EntityObserver};
use crate::des::random_variables::rv::{RandomVariable, UniformRandomVariable};
use crate::des::shared::capabilities::{RandomSource, SeededRandom};
use crate::des::shared::precision::{bgn, Decimal};

/// One node in the heterogeneous program graph (`Entity<any>`).
#[derive(Clone)]
enum Node {
    Source(Rc<RefCell<EntitySource>>),
    Proc(Rc<RefCell<EntityProcessor>>),
    Sink(Rc<RefCell<EntitySink>>),
}

impl Node {
    /// `target` view for `addOutConnection`.
    fn as_has_input(&self) -> Rc<RefCell<dyn HasInput>> {
        match self {
            Node::Proc(p) => p.clone(),
            Node::Sink(s) => s.clone(),
            Node::Source(_) => panic!("source has no input connection"),
        }
    }

    fn add_out_connection(&self, target: Rc<RefCell<dyn HasInput>>) {
        match self {
            Node::Source(s) => {
                s.borrow_mut().add_out_connection(target);
            }
            Node::Proc(p) => {
                p.borrow_mut().add_out_connection(target);
            }
            Node::Sink(_) => panic!("sink has no output connection"),
        }
    }

    fn do_time_step(&self, step: Decimal) {
        match self {
            Node::Source(s) => s.borrow_mut().do_time_step(step),
            Node::Proc(p) => p.borrow_mut().do_time_step(step),
            Node::Sink(s) => s.borrow_mut().do_time_step(step),
        }
    }

    fn computed(&self) -> EntityGraphData {
        match self {
            Node::Source(s) => s.borrow().get_with_computed_properties(),
            Node::Proc(p) => p.borrow().get_with_computed_properties(),
            Node::Sink(s) => s.borrow().get_with_computed_properties(),
        }
    }
}

/// Distinct seeds keep each injected RNG reproducible (no `Math.random`).
struct SeedGen(u32);
impl SeedGen {
    fn next_rng(&mut self) -> Box<dyn RandomSource> {
        self.0 = self.0.wrapping_add(1);
        Box::new(SeededRandom::new(self.0))
    }
}

fn uniform(a: f64, b: f64, seeds: &mut SeedGen) -> Box<dyn RandomVariable> {
    Box::new(UniformRandomVariable::new(bgn(a), bgn(b), seeds.next_rng()))
}

/// Entry point (TS top-level `run()` closure + its invocation).
pub fn run() {
    let step_size_millis = get_step_size();

    let obs_concrete = Rc::new(RefCell::new(ProgramObserver::new()));
    let obs: Rc<RefCell<dyn EntityObserver>> = obs_concrete.clone();
    let mut seeds = SeedGen(1);

    // `programEntities` Map literal — insertion order A, B, C, D, E preserved.
    let order = ["A", "B", "C", "D", "E"];
    let mut map: HashMap<&str, Node> = HashMap::new();

    let a = Rc::new(RefCell::new(EntitySource::new(
        "A".to_string(),
        uniform(200.0, 500.0, &mut seeds),
        -1,
    )));
    a.borrow_mut().subscribe(obs.clone());
    map.insert("A", Node::Source(a));

    for id in ["B", "C", "D"] {
        let (lo, hi) = match id {
            "D" => (100.0, 500.0),
            _ => (300.0, 500.0),
        };
        let p = Rc::new(RefCell::new(EntityProcessor::new(
            id.to_string(),
            uniform(lo, hi, &mut seeds),
            OutputRoutingPolicy::default(),
        )));
        p.borrow_mut().subscribe(obs.clone());
        map.insert(id, Node::Proc(p));
    }

    let e = Rc::new(RefCell::new(EntitySink::new("E".to_string())));
    e.borrow_mut().subscribe(obs.clone());
    map.insert("E", Node::Sink(e));

    // Wire A→B→C→D→E (TS only adds OUT connections).
    for (source_id, target_id) in [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")] {
        let source = map.get(source_id).expect("source node").clone();
        let target = map.get(target_id).expect("target node").clone();
        source.add_out_connection(target.as_has_input());
    }

    // `console.log(programEntities.get('A'))`.
    if let Some(node_a) = map.get("A") {
        println!("{:?}", node_a.computed());
    }

    let program_list: Vec<Node> = order
        .iter()
        .map(|id| map.get(id).unwrap().clone())
        .collect();
    let mut shuffled = program_list.clone();
    let mut rng = SeededRandom::new(0xD15EA5E);

    // `finalize()` — dump each node's computed properties + observer population.
    let finalize = |obs: &Rc<RefCell<ProgramObserver>>| {
        let mut i = 0;
        for id in order {
            i += 1;
            println!("{i} {i} {i} {i} {i} {i} {i} {i} {i} **************************************");
            println!("{:?}", map.get(id).unwrap().computed());
        }
        println!(
            "obs.movingEntities.size: {}",
            obs.borrow().moving_entities.len()
        );
    };

    // `runAll(102)` — warm-up loop. Work runs for i = 102 ..= 20001, then the
    // `i > 20000` guard transitions to the sources-off loop.
    let mut i = 102;
    loop {
        bump_time_accrued_by_time_step(step_size_millis);
        fisher_yates_shuffle(&mut shuffled, &mut rng);
        for v in &shuffled {
            println!("{i}");
            v.do_time_step(step_size_millis);
        }
        if i > 20000 {
            break;
        }
        i += 1;
    }

    // PORT NOTE: `(global as any).turnOffSources = true` — see header.

    // `runAfterSourcesOff(102)` — audited loop. Work runs for j = 102 ..= 20000,
    // and `finalize()` fires once after the j == 20000 tick.
    let mut j = 102;
    loop {
        bump_time_accrued_by_time_step(step_size_millis);
        println!("doing the audit:");
        do_audit();
        fisher_yates_shuffle(&mut shuffled, &mut rng);
        for v in &shuffled {
            println!("{j}");
            v.do_time_step(step_size_millis);
        }
        if j == 20000 {
            finalize(&obs_concrete);
            break;
        }
        j += 1;
    }
}
