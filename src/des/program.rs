//! Port of `src/des/program.ts` (module `des::program`).
//!
//! Despite its shebang the TS file is a LIBRARY: it exports `getEntities`, the
//! builder for the default five-node entity formation (`A` source → `B`/`C`/`D`
//! processors → `E` sink), and has no `require.main` guard. It is therefore a
//! plain module, not a binary.
//!
//! ## Conversion notes (faithful to the TS shape)
//!
//!   * `getEntities(stepSize): Map<string, VisualNode<any>>` → [`get_entities`]
//!     returning a `Vec<(String, VisualNode)>` — the insertion-ordered analogue
//!     of a JS `Map` (the consumer, `http_server`, does `Array.from(map)`).
//!   * `mathjs.BigNumber stepSize` → [`Decimal`] (the engine-wide compound-clock
//!     tier), built with [`bgn`].
//!   * `EntitySink('E', new PoissonRandomVariable(), {})`: the ported
//!     [`EntitySink`] constructor takes only an id (its rv/opts were dropped in
//!     the entity port), so the `PoissonRandomVariable` and `{}` are not passed.
//!
//! PORT NOTE: `VisualNodeArgs::entity` requires `dyn StationaryEntity`, but the
//! existing entity ports (`entity_source` / `entity_processing` / `entity_sink`)
//! do not implement that trait (only `Entity` + the `Has*` traits). This module
//! supplies the smallest possible `impl StationaryEntity` for each, with the
//! `do_setup_after_*_conn` bodies copied verbatim from the TS sources
//! (`EntitySource`/`EntityProcessor` return `true`; `EntitySink` returns
//! `false`), so `get_entities` can wrap them in [`VisualNode`]s.
//!
//! PORT NOTE: the TS relied on ambient `Math.random` inside the RV constructors;
//! per the engine's capability-injection rule each RV here receives a seeded
//! [`SeededRandom`] (deterministic, the Rust side's `RandomSource`).

#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use crate::des::entity_processing::processing::EntityProcessor;
use crate::des::entity_routing::output_routing_policy::OutputRoutingPolicy;
use crate::des::entity_sink::sink::EntitySink;
use crate::des::entity_source::source::EntitySource;
use crate::des::r#abstract::r#abstract::StationaryEntity;
use crate::des::random_variables::rv::{ExponentialRandomVariable, RandomVariable};
use crate::des::shared::capabilities::SeededRandom;
use crate::des::shared::precision::{bgn, Decimal};
use crate::des::visual::visual_node::{VisualNode, VisualNodeArgs};

// -----------------------------------------------------------------------------
// StationaryEntity impls required by `VisualNodeArgs` (see module PORT NOTE).
// -----------------------------------------------------------------------------

impl StationaryEntity for EntitySource {
    fn do_setup_after_input_conn(&mut self) -> bool {
        true
    }
    fn do_setup_after_output_conn(&mut self) -> bool {
        true
    }
}

impl StationaryEntity for EntityProcessor {
    fn do_setup_after_input_conn(&mut self) -> bool {
        true
    }
    fn do_setup_after_output_conn(&mut self) -> bool {
        true
    }
}

impl StationaryEntity for EntitySink {
    fn do_setup_after_input_conn(&mut self) -> bool {
        false
    }
    fn do_setup_after_output_conn(&mut self) -> bool {
        false
    }
}

const ICON_URL: &str = "https://xyz.com";

/// Distinct seeds keep each injected RNG reproducible (no `Math.random`).
struct SeedGen(u32);

impl SeedGen {
    fn next_rng(&mut self) -> Box<SeededRandom> {
        self.0 = self.0.wrapping_add(1);
        Box::new(SeededRandom::new(self.0))
    }
}

/// `new ExponentialRandomVariable({lambda, timeStep})`.
fn exp_rv(lambda: Decimal, time_step: Decimal, seeds: &mut SeedGen) -> Box<dyn RandomVariable> {
    Box::new(ExponentialRandomVariable::new(
        lambda,
        time_step,
        seeds.next_rng(),
    ))
}

fn visual<E: StationaryEntity + 'static>(label: &str, entity: E) -> VisualNode {
    VisualNode::new(VisualNodeArgs {
        label: label.to_string(),
        icon_url: ICON_URL.to_string(),
        entity: Rc::new(RefCell::new(entity)) as Rc<RefCell<dyn StationaryEntity>>,
    })
}

/// `getEntities(stepSize)` — build the default A→B/C/D→E formation. The returned
/// `Vec` preserves the TS `Map` insertion order (`A`, `B`, `C`, `D`, `E`).
pub fn get_entities(step_size: Decimal) -> Vec<(String, VisualNode)> {
    let mut seeds = SeedGen(0);

    // A: source, exponential inter-arrival, never turns off (turnOffAfterCount: -1).
    let a = EntitySource::new(
        "A".to_string(),
        exp_rv(bgn(5.0 / 100.0), step_size, &mut seeds),
        -1,
    );

    // B / C / D: exponential service-time processors (lambda = 1/10).
    let b = EntityProcessor::new(
        "B".to_string(),
        exp_rv(bgn(1.0 / 10.0), step_size, &mut seeds),
        OutputRoutingPolicy::default(),
    );
    let c = EntityProcessor::new(
        "C".to_string(),
        exp_rv(bgn(1.0 / 10.0), step_size, &mut seeds),
        OutputRoutingPolicy::default(),
    );
    let d = EntityProcessor::new(
        "D".to_string(),
        exp_rv(bgn(1.0 / 10.0), step_size, &mut seeds),
        OutputRoutingPolicy::default(),
    );

    // E: sink (the TS PoissonRandomVariable + {} opts are dropped — see notes).
    let e = EntitySink::new("E".to_string());

    vec![
        ("A".to_string(), visual("A", a)),
        ("B".to_string(), visual("B", b)),
        ("C".to_string(), visual("C", c)),
        ("D".to_string(), visual("D", d)),
        ("E".to_string(), visual("E", e)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_five_node_formation_in_order() {
        let entities = get_entities(bgn(500.0));
        let labels: Vec<&str> = entities.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(labels, ["A", "B", "C", "D", "E"]);
        // Every node carries its label and icon url.
        for (k, node) in &entities {
            assert_eq!(&node.label, k);
            assert_eq!(node.icon_url, ICON_URL);
        }
    }
}
