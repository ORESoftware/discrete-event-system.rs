//! Port of `src/des/main-epidemic.ts`.
//!
//! Wires an SEIR-style epidemic over the entity graph: a source feeds the
//! Susceptible processor; tokens flow S → E → I-P and fan out through
//! presymptomatic / asymptomatic / symptomatic / hospitalized processors with
//! probability-decision routing to Recovered / Dead, recycling R → S.
//!
//! `bgn(21)` step size → `crate::des::shared::precision::bgn`. The heterogeneous
//! `Map<string, StationaryEntity<any>>` + uniform `addOutConnection` /
//! `addInConnection` / `doSetupAfter*` is modeled with a [`Node`] enum that
//! dispatches to the concrete entity's trait methods.
//!
//! PORT NOTES:
//!   * Node-only imports (`uuid`, `safe-stringify`, `ws-server`, `visual-node`,
//!     `mathjs`) are dropped; randomness is injected via
//!     `crate::des::shared::capabilities::SeededRandom`.
//!   * `(global as any).turnOffSources = true` has no analog — the Rust source
//!     keeps only the per-source `turn_off_after_count` guard (300 here).
//!   * `console.log({source})` / `programEntities.get('A')` debug prints are
//!     reproduced as id-level traces.

#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::des::entity_decision::probability_decision::{Branch, ProbabilityDecisionEntity};
use crate::des::entity_processing::processing::EntityProcessor;
use crate::des::entity_routing::output_routing_policy::OutputRoutingPolicy;
use crate::des::entity_sink::sink::EntitySink;
use crate::des::entity_source::source::EntitySource;
use crate::des::general::do_audit::do_audit;
use crate::des::general::general::fisher_yates_shuffle;
use crate::des::observers::program_observer::ProgramObserver;
use crate::des::r#abstract::interfaces::{EntityGraphData, HasInput, HasManyOutputConnections, HasOutput};
use crate::des::r#abstract::r#abstract::{Entity, EntityObserver};
use crate::des::random_variables::rv::{PoissonRandomVariable, RandomVariable, UniformRandomVariable};
use crate::des::shared::capabilities::{RandomSource, SeededRandom};
use crate::des::shared::precision::{bgn, Decimal};

/// One node in the heterogeneous program graph (`StationaryEntity<any>`).
#[derive(Clone)]
enum Node {
    Source(Rc<RefCell<EntitySource>>),
    Proc(Rc<RefCell<EntityProcessor>>),
    Decision(Rc<RefCell<ProbabilityDecisionEntity>>),
    Sink(Rc<RefCell<EntitySink>>),
}

impl Node {
    /// `target` view for `addOutConnection`.
    fn as_has_input(&self) -> Rc<RefCell<dyn HasInput>> {
        match self {
            Node::Proc(p) => p.clone(),
            Node::Decision(d) => d.clone(),
            Node::Sink(s) => s.clone(),
            Node::Source(_) => panic!("source has no input connection"),
        }
    }

    /// `source` view for `addInConnection`.
    fn as_has_many_out(&self) -> Rc<RefCell<dyn HasManyOutputConnections>> {
        match self {
            Node::Source(s) => s.clone(),
            Node::Proc(p) => p.clone(),
            Node::Decision(d) => d.clone(),
            Node::Sink(_) => panic!("sink has no output connection"),
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
            Node::Decision(d) => {
                d.borrow_mut().add_out_connection(target);
            }
            Node::Sink(_) => {}
        }
    }

    fn add_in_connection(&self, source: Rc<RefCell<dyn HasManyOutputConnections>>) {
        match self {
            Node::Proc(p) => {
                p.borrow_mut().add_in_connection(source);
            }
            Node::Decision(d) => {
                d.borrow_mut().add_in_connection(source);
            }
            Node::Sink(s) => {
                s.borrow_mut().add_in_connection(source);
            }
            Node::Source(_) => {}
        }
    }

    fn setup_after_output_conn(&self) {
        match self {
            Node::Source(s) => {
                HasOutput::do_setup_after_output_conn(&mut *s.borrow_mut());
            }
            Node::Proc(p) => {
                HasOutput::do_setup_after_output_conn(&mut *p.borrow_mut());
            }
            Node::Decision(d) => {
                HasOutput::do_setup_after_output_conn(&mut *d.borrow_mut());
            }
            Node::Sink(s) => {
                HasInput::do_setup_after_output_conn(&mut *s.borrow_mut());
            }
        }
    }

    fn setup_after_input_conn(&self) {
        match self {
            Node::Source(s) => {
                HasOutput::do_setup_after_input_conn(&mut *s.borrow_mut());
            }
            Node::Proc(p) => {
                HasOutput::do_setup_after_input_conn(&mut *p.borrow_mut());
            }
            Node::Decision(d) => {
                HasOutput::do_setup_after_input_conn(&mut *d.borrow_mut());
            }
            Node::Sink(s) => {
                HasInput::do_setup_after_input_conn(&mut *s.borrow_mut());
            }
        }
    }

    fn do_time_step(&self, step: Decimal) {
        match self {
            Node::Source(s) => s.borrow_mut().do_time_step(step),
            Node::Proc(p) => p.borrow_mut().do_time_step(step),
            Node::Decision(d) => d.borrow_mut().do_time_step(step),
            Node::Sink(s) => s.borrow_mut().do_time_step(step),
        }
    }

    fn computed(&self) -> EntityGraphData {
        match self {
            Node::Source(s) => s.borrow().get_with_computed_properties(),
            Node::Proc(p) => p.borrow().get_with_computed_properties(),
            Node::Decision(d) => d.borrow().get_with_computed_properties(),
            Node::Sink(s) => s.borrow().get_with_computed_properties(),
        }
    }
}

/// Distinct seeds keep each injected RNG reproducible.
struct SeedGen(u32);
impl SeedGen {
    fn next(&mut self) -> Box<dyn RandomSource> {
        self.0 = self.0.wrapping_add(1);
        Box::new(SeededRandom::new(self.0))
    }
}

/// `[10, 20]` uniform inter-event RV used throughout the TS spec.
fn uniform_10_20(seeds: &mut SeedGen) -> Box<dyn RandomVariable> {
    Box::new(UniformRandomVariable::new(bgn(10.0), bgn(20.0), seeds.next()))
}

/// Entry point (TS top-level `run()` closure + invocation).
pub fn run() {
    let step_size_millis = bgn(21.0);
    let obs_concrete = Rc::new(RefCell::new(ProgramObserver::new()));
    let obs: Rc<RefCell<dyn EntityObserver>> = obs_concrete.clone();
    let mut seeds = SeedGen(1);

    let mut order: Vec<String> = Vec::new();
    let mut map: HashMap<String, Node> = HashMap::new();

    let add = |order: &mut Vec<String>, map: &mut HashMap<String, Node>, id: &str, node: Node| {
        order.push(id.to_string());
        map.insert(id.to_string(), node);
    };

    // main-source: EntitySource (turnOffAfterCount = 300).
    {
        let s = Rc::new(RefCell::new(EntitySource::new(
            "main-source".to_string(),
            uniform_10_20(&mut seeds),
            300,
        )));
        s.borrow_mut().subscribe(obs.clone());
        add(&mut order, &mut map, "main-source", Node::Source(s));
    }

    // Processors S, E, I-P, I-S, I-A, I-H, R, D.
    for id in ["S", "E", "I-P", "I-S", "I-A", "I-H", "R", "D"] {
        let p = Rc::new(RefCell::new(EntityProcessor::new(
            id.to_string(),
            uniform_10_20(&mut seeds),
            OutputRoutingPolicy::default(),
        )));
        p.borrow_mut().subscribe(obs.clone());
        add(&mut order, &mut map, id, Node::Proc(p));
    }

    // Probability-decision routers (0.4 / 0.6 split each).
    for id in ["I-P-Decision", "I-S-Decision", "I-H-Decision"] {
        let d = Rc::new(RefCell::new(ProbabilityDecisionEntity::new(
            id.to_string(),
            vec![Branch { index: 0, prob: bgn(0.4) }, Branch { index: 1, prob: bgn(0.6) }],
            uniform_10_20(&mut seeds),
            seeds.next(),
        )));
        d.borrow_mut().subscribe(obs.clone());
        add(&mut order, &mut map, id, Node::Decision(d));
    }

    // main-sink: EntitySink.
    {
        let _poisson: Box<dyn RandomVariable> = Box::new(PoissonRandomVariable::new(seeds.next()));
        let s = Rc::new(RefCell::new(EntitySink::new("main-sink".to_string())));
        s.borrow_mut().subscribe(obs.clone());
        add(&mut order, &mut map, "main-sink", Node::Sink(s));
    }

    // Re-order the insertion to match the TS `Map` literal exactly.
    order = vec![
        "main-source".into(),
        "S".into(),
        "E".into(),
        "I-P".into(),
        "I-P-Decision".into(),
        "I-S".into(),
        "I-S-Decision".into(),
        "I-A".into(),
        "I-H".into(),
        "I-H-Decision".into(),
        "R".into(),
        "D".into(),
        "main-sink".into(),
    ];

    let edges: [(&str, &str); 17] = [
        ("main-source", "S"),
        ("S", "E"),
        ("E", "I-P"),
        ("I-P", "I-A"),
        ("I-P", "I-S"),
        ("I-P", "I-P-Decision"),
        ("I-P-Decision", "I-A"),
        ("I-P-Decision", "I-S"),
        ("I-A", "R"),
        ("I-S", "I-S-Decision"),
        ("I-S-Decision", "R"),
        ("I-S-Decision", "I-H"),
        ("I-H", "I-H-Decision"),
        ("I-H-Decision", "R"),
        ("I-H-Decision", "D"),
        ("D", "main-sink"),
        ("R", "S"),
    ];

    for (source_id, target_id) in edges {
        let source = map.get(source_id).expect("source node").clone();
        let target = map.get(target_id).expect("target node").clone();
        println!("{{ source: {source_id} }}");
        source.add_out_connection(target.as_has_input());
        target.add_in_connection(source.as_has_many_out());
    }

    for id in &order {
        let n = map.get(id).unwrap();
        n.setup_after_output_conn();
        n.setup_after_input_conn();
    }

    // `programEntities.get('A')` — there is no 'A'.
    println!("{:?}", map.get("A").map(|_| "<node>"));

    let mut program_list: Vec<Node> = order.iter().map(|id| map.get(id).unwrap().clone()).collect();
    let mut rng = SeededRandom::new(0xC0FFEE);

    let now = std::time::Instant::now();
    for i in 0..1000 {
        println!("doing first iteration: {i}");
        fisher_yates_shuffle(&mut program_list, &mut rng);
        for v in &program_list {
            v.do_time_step(step_size_millis);
        }
    }
    println!("{}", now.elapsed().as_millis());

    // PORT NOTE: `(global as any).turnOffSources = true` — see header.

    for i in 0..500 {
        println!("doing second iteration: {i}");
        do_audit();
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
