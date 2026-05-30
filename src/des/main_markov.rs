//! Port of `src/des/main-markov.ts`.
//!
//! Wires a Markov-chain-style entity network of `EntitySource`s feeding a ring
//! of bidirectional `EntityProcessor`s (A↔B↔C↔D↔E↔F plus A→F / F→A chords) and
//! observes its steady-state behaviour. After 1000 warm-up ticks the sources
//! switch off (their per-source `turn_off_after_count` guard) and a 500-tick
//! audit loop asserts the total population trapped in the processor ring stays
//! conserved.
//!
//! Conversion notes (file-specific):
//!   * The heterogeneous `Map<string, Entity<any>>` + uniform `addOutConnection`
//!     / `addInConnection` is modeled with a [`Node`] enum, exactly as the sister
//!     port `main_epidemic`.
//!   * `Set<EntityProcessor>` (`allProcessors`, populated by the TS `addToSet`)
//!     → a `Vec<Rc<RefCell<EntityProcessor>>>` captured by [`AuditSizes`].
//!   * `auditSizes(s)` returns a stateful closure; ported as the [`AuditSizes`]
//!     struct holding `first` / `previous_total`. `makeError(...)` → `panic!`.
//!   * PORT NOTE: `(global as any).turnOffSources = true` has no analog — only
//!     the per-source `turn_off_after_count` guard (300 here) remains, matching
//!     the `entity_source::source` port.
//!   * Node-only imports (`uuid`, `safe-stringify`, `mathjs`) are dropped;
//!     randomness is injected via `SeededRandom`.

#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::des::entity_moving::moving::MovingEntity;
use crate::des::entity_processing::processing::EntityProcessor;
use crate::des::entity_routing::output_routing_policy::OutputRoutingPolicy;
use crate::des::entity_source::source::EntitySource;
use crate::des::general::general::fisher_yates_shuffle;
use crate::des::observers::program_observer::ProgramObserver;
use crate::des::r#abstract::interfaces::{
    EntityGraphData, HasInput, HasManyOutputConnections, HasOutput,
};
use crate::des::r#abstract::r#abstract::{Entity, EntityObserver};
use crate::des::random_variables::rv::{RandomVariable, UniformRandomVariable};
use crate::des::shared::capabilities::{RandomSource, SeededRandom};
use crate::des::shared::precision::{bgn, Decimal};

/// One node in the heterogeneous program graph (`Entity<any>`). Markov only uses
/// sources and processors (no sink), so the enum has just two arms.
#[derive(Clone)]
enum Node {
    Source(Rc<RefCell<EntitySource>>),
    Proc(Rc<RefCell<EntityProcessor>>),
}

impl Node {
    /// `target` view for `addOutConnection`.
    fn as_has_input(&self) -> Rc<RefCell<dyn HasInput>> {
        match self {
            Node::Proc(p) => p.clone(),
            Node::Source(_) => panic!("source has no input connection"),
        }
    }

    /// `source` view for `addInConnection`.
    fn as_has_many_out(&self) -> Rc<RefCell<dyn HasManyOutputConnections>> {
        match self {
            Node::Source(s) => s.clone(),
            Node::Proc(p) => p.clone(),
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
        }
    }

    fn add_in_connection(&self, source: Rc<RefCell<dyn HasManyOutputConnections>>) {
        match self {
            Node::Proc(p) => {
                p.borrow_mut().add_in_connection(source);
            }
            Node::Source(_) => {}
        }
    }

    fn do_time_step(&self, step: Decimal) {
        match self {
            Node::Source(s) => s.borrow_mut().do_time_step(step),
            Node::Proc(p) => p.borrow_mut().do_time_step(step),
        }
    }

    fn computed(&self) -> EntityGraphData {
        match self {
            Node::Source(s) => s.borrow().get_with_computed_properties(),
            Node::Proc(p) => p.borrow().get_with_computed_properties(),
        }
    }
}

/// `auditSizes(allProcessors)` — a stateful auditor over the processor set that
/// throws (`makeError('totals are not equal')`) if the summed internal-queue
/// population ever changes between successive audits.
struct AuditSizes {
    processors: Vec<Rc<RefCell<EntityProcessor>>>,
    first: bool,
    previous_total: usize,
}

impl AuditSizes {
    fn new(processors: Vec<Rc<RefCell<EntityProcessor>>>) -> Self {
        AuditSizes { processors, first: true, previous_total: 0 }
    }

    /// The returned closure body of the TS `auditSizes`.
    fn run(&mut self) {
        let mut total = 0usize;
        for v in &self.processors {
            total += v.borrow().do_audit();
        }
        if !self.first && self.previous_total != total {
            panic!("totals are not equal");
        }
        self.previous_total = total;
        self.first = false;
    }
}

/// Distinct seeds keep each injected RNG reproducible (no `Math.random`).
struct SeedGen(u32);
impl SeedGen {
    fn next(&mut self) -> Box<dyn RandomSource> {
        self.0 = self.0.wrapping_add(1);
        Box::new(SeededRandom::new(self.0))
    }
}

fn uniform(a: f64, b: f64, seeds: &mut SeedGen) -> Box<dyn RandomVariable> {
    Box::new(UniformRandomVariable::new(bgn(a), bgn(b), seeds.next()))
}

fn add_source(
    order: &mut Vec<String>,
    map: &mut HashMap<String, Node>,
    obs: &Rc<RefCell<dyn EntityObserver>>,
    seeds: &mut SeedGen,
    id: &str,
    a: f64,
    b: f64,
) {
    let s = Rc::new(RefCell::new(EntitySource::new(id.to_string(), uniform(a, b, seeds), 300)));
    s.borrow_mut().subscribe(obs.clone());
    order.push(id.to_string());
    map.insert(id.to_string(), Node::Source(s));
}

#[allow(clippy::too_many_arguments)]
fn add_proc(
    order: &mut Vec<String>,
    map: &mut HashMap<String, Node>,
    obs: &Rc<RefCell<dyn EntityObserver>>,
    seeds: &mut SeedGen,
    all_processors: &mut Vec<Rc<RefCell<EntityProcessor>>>,
    id: &str,
    a: f64,
    b: f64,
) {
    let p = Rc::new(RefCell::new(EntityProcessor::new(
        id.to_string(),
        uniform(a, b, seeds),
        OutputRoutingPolicy::default(),
    )));
    p.borrow_mut().subscribe(obs.clone());
    all_processors.push(p.clone());
    order.push(id.to_string());
    map.insert(id.to_string(), Node::Proc(p));
}

/// Entry point (TS top-level `run()` closure + invocation).
pub fn run() {
    let step_size_millis = bgn(500.0);
    let obs_concrete = Rc::new(RefCell::new(ProgramObserver::new()));
    let obs: Rc<RefCell<dyn EntityObserver>> = obs_concrete.clone();
    let mut seeds = SeedGen(1);

    let mut order: Vec<String> = Vec::new();
    let mut map: HashMap<String, Node> = HashMap::new();
    let mut all_processors: Vec<Rc<RefCell<EntityProcessor>>> = Vec::new();

    // `programEntities` Map literal — insertion order preserved exactly.
    add_source(&mut order, &mut map, &obs, &mut seeds, "A-source", 10.0, 20.0);
    add_proc(&mut order, &mut map, &obs, &mut seeds, &mut all_processors, "A", 10.0, 20.0);
    add_source(&mut order, &mut map, &obs, &mut seeds, "B-source", 10.0, 20.0);
    add_proc(&mut order, &mut map, &obs, &mut seeds, &mut all_processors, "B", 10.0, 20.0);
    add_proc(&mut order, &mut map, &obs, &mut seeds, &mut all_processors, "C", 10.0, 20.0);
    add_source(&mut order, &mut map, &obs, &mut seeds, "C-source", 10.0, 20.0);
    add_proc(&mut order, &mut map, &obs, &mut seeds, &mut all_processors, "D", 10.0, 20.0);
    add_proc(&mut order, &mut map, &obs, &mut seeds, &mut all_processors, "E", 10.0, 20.0);
    add_proc(&mut order, &mut map, &obs, &mut seeds, &mut all_processors, "F", 5.0, 10.0);
    add_source(&mut order, &mut map, &obs, &mut seeds, "D-source", 10.0, 20.0);

    // `doAudit = auditSizes(allProcessors)`.
    let mut do_audit = AuditSizes::new(all_processors.clone());

    let edges: [(&str, &str); 16] = [
        ("A-source", "A"),
        ("A", "F"),
        ("A", "B"),
        ("B", "A"),
        ("B-source", "B"),
        ("B", "C"),
        ("C", "B"),
        ("C-source", "C"),
        ("C", "D"),
        ("D", "C"),
        ("D-source", "D"),
        ("D", "E"),
        ("E", "D"),
        ("E", "F"),
        ("F", "E"),
        ("F", "A"),
    ];

    for (source_id, target_id) in edges {
        let source = map.get(source_id).expect("source node").clone();
        let target = map.get(target_id).expect("target node").clone();
        println!("{{ source: {source_id} }}");
        source.add_out_connection(target.as_has_input());
        target.add_in_connection(source.as_has_many_out());
    }

    // `console.log(programEntities.get('A'))`.
    if let Some(a) = map.get("A") {
        println!("{:?}", a.computed());
    }

    let mut program_list: Vec<Node> = order.iter().map(|id| map.get(id).unwrap().clone()).collect();
    let mut rng = SeededRandom::new(0xC0FFEE);

    for i in 0..1000 {
        println!("doing first iteration: {i}");
        fisher_yates_shuffle(&mut program_list, &mut rng);
        for v in &program_list {
            v.do_time_step(step_size_millis);
        }
    }

    // PORT NOTE: `(global as any).turnOffSources = true` — see header.

    for i in 0..500 {
        println!("doing second iteration: {i}");
        do_audit.run();
        fisher_yates_shuffle(&mut program_list, &mut rng);
        for v in &program_list {
            v.do_time_step(bgn(500.0));
        }
    }

    for b in &obs_concrete.borrow().moving_entities {
        println!("{}", b.borrow().id());
    }

    let mut i = 0;
    for id in &order {
        i += 1;
        println!("{i} {i} {i} {i} {i} {i} {i} {i} {i} **************************************");
        println!("{i} {i} {i} {i} {i} {i} {i} {i} {i} **************************************");
        println!("{:?}", map.get(id).unwrap().computed());
    }
}
