//! Canonical studio demos. Each is a *flat* graph of visual blocks where at
//! least one block packs several Layer-2 elements into a single cell — exactly
//! the two architectural constraints (multi-element blocks, no nesting).

use serde_json::{json, Value};

use super::cell::{Gain, Queue, RuntimeCell, Saturation, Source, SourceKind, Sum, TransportDelay};
use super::graph::{CompiledStudio, NodeRole, StudioError, VisualNode};

/// A built demo ready to run + render.
pub struct StudioDemo {
    pub compiled: CompiledStudio,
    pub steps: usize,
    pub dt: f64,
    pub title: String,
    pub description: String,
    pub blocks: Value,
}

/// A JSON description of each block and the Layer-2 elements it contains.
pub fn blocks_doc(c: &CompiledStudio) -> Value {
    json!(c
        .nodes()
        .iter()
        .map(|nd| json!({
            "id": nd.id,
            "role": nd.role.as_str(),
            "elements": nd.cell.element_names(),
            "inPorts": nd.n_in(),
            "outPorts": nd.n_out(),
        }))
        .collect::<Vec<_>>())
}

/// Source (ramp) → a single block packing TWO Layer-2 ops (gain ▸ saturation) →
/// sink. One visual block, two runtime elements.
pub fn signal_chain() -> Result<StudioDemo, StudioError> {
    let mut g = StudioGraphBuilder();
    let src = g.add(
        VisualNode::new(
            "input",
            NodeRole::Source,
            RuntimeCell::single(Box::new(Source::new(
                "ramp",
                SourceKind::Ramp {
                    slope: 1.0,
                    intercept: 0.0,
                },
            ))),
        )
        .with_label("input (ramp)")
        .at(40.0, 130.0),
    )?;
    let shaper = g.add(
        VisualNode::new(
            "shaper",
            NodeRole::Transform,
            RuntimeCell::new(vec![
                Box::new(Gain::new("gain·0.5", 0.5)),
                Box::new(Saturation::new("sat[-1,2]", -1.0, 2.0)),
            ])?,
        )
        .with_label("shaper")
        .at(250.0, 120.0),
    )?;
    let sink = g.add(
        VisualNode::new(
            "output",
            NodeRole::Sink,
            RuntimeCell::single(Box::new(Gain::new("probe", 1.0))),
        )
        .with_label("output")
        .at(470.0, 130.0),
    )?;
    g.connect(src, 0, shaper, 0)?;
    g.connect(shaper, 0, sink, 0)?;
    let compiled = g.build()?;
    let blocks = blocks_doc(&compiled);
    Ok(StudioDemo {
        compiled,
        steps: 80,
        dt: 0.1,
        title: "Signal Chain".to_string(),
        description: "A flat block graph: a ramp source feeds one block that packs gain ▸ \
                      saturation (two Layer-2 elements), into a sink."
            .to_string(),
        blocks,
    })
}

/// Two sources (sine + constant) → a multi-input block packing sum ▸ gain → sink.
/// Demonstrates a block with several input ports *and* several Layer-2 elements.
pub fn mixer() -> Result<StudioDemo, StudioError> {
    let mut g = StudioGraphBuilder();
    let sine = g.add(
        VisualNode::new(
            "carrier",
            NodeRole::Source,
            RuntimeCell::single(Box::new(Source::new(
                "sine",
                SourceKind::Sine {
                    amp: 1.0,
                    freq: 0.25,
                    bias: 0.0,
                },
            ))),
        )
        .with_label("carrier (sine)")
        .at(40.0, 60.0),
    )?;
    let bias = g.add(
        VisualNode::new(
            "bias",
            NodeRole::Source,
            RuntimeCell::single(Box::new(Source::new("const", SourceKind::Const(0.5)))),
        )
        .with_label("bias (const)")
        .at(40.0, 210.0),
    )?;
    let mix = g.add(
        VisualNode::new(
            "mix",
            NodeRole::Transform,
            RuntimeCell::new(vec![
                Box::new(Sum::new("sum", vec![1.0, 1.0])),
                Box::new(Gain::new("gain·0.5", 0.5)),
            ])?,
        )
        .with_label("mix")
        .at(260.0, 135.0),
    )?;
    let sink = g.add(
        VisualNode::new(
            "output",
            NodeRole::Sink,
            RuntimeCell::single(Box::new(Gain::new("probe", 1.0))),
        )
        .with_label("output")
        .at(480.0, 135.0),
    )?;
    g.connect(sine, 0, mix, 0)?;
    g.connect(bias, 0, mix, 1)?;
    g.connect(mix, 0, sink, 0)?;
    let compiled = g.build()?;
    let blocks = blocks_doc(&compiled);
    Ok(StudioDemo {
        compiled,
        steps: 96,
        dt: 0.1,
        title: "Mixer".to_string(),
        description: "A flat block graph: a sine carrier and a constant bias feed one block \
                      with two input ports packing sum ▸ gain, into a sink."
            .to_string(),
        blocks,
    })
}

/// Arrivals → a single-server queue block (StationEntity semantics) → a conveyor
/// block (Movable-in-transit) → sink. Demonstrates the DES runtime primitives as
/// Layer-2 ops inside the flat studio graph, with `dt = 1` reading as per-tick.
pub fn queue_line() -> Result<StudioDemo, StudioError> {
    let mut g = StudioGraphBuilder();
    let arrivals = g.add(
        VisualNode::new(
            "arrivals",
            NodeRole::Source,
            // 0 arrivals/tick until t=3, then a burst of 8/tick (overloads the server).
            RuntimeCell::single(Box::new(Source::new("step", SourceKind::Step { t0: 3.0, before: 0.0, after: 8.0 }))),
        )
        .with_label("arrivals (step)")
        .at(40.0, 130.0),
    )?;
    let server = g.add(
        VisualNode::new(
            "server",
            NodeRole::Transform,
            RuntimeCell::single(Box::new(Queue::new("queue·5/tick", 5.0))),
        )
        .with_label("server")
        .at(250.0, 120.0),
    )?;
    let belt = g.add(
        VisualNode::new(
            "belt",
            NodeRole::Transform,
            RuntimeCell::single(Box::new(TransportDelay::new("delay·4", 4))),
        )
        .with_label("belt")
        .at(450.0, 120.0),
    )?;
    let sink = g.add(
        VisualNode::new(
            "departures",
            NodeRole::Sink,
            RuntimeCell::single(Box::new(Gain::new("probe", 1.0))),
        )
        .with_label("departures")
        .at(650.0, 130.0),
    )?;
    g.connect(arrivals, 0, server, 0)?;
    g.connect(server, 0, belt, 0)?;
    g.connect(belt, 0, sink, 0)?;
    let compiled = g.build()?;
    let blocks = blocks_doc(&compiled);
    Ok(StudioDemo {
        compiled,
        steps: 24,
        dt: 1.0,
        title: "Queue Line".to_string(),
        description: "A flat block graph using DES runtime primitives as Layer-2 ops: an \
                      overloaded single-server queue (StationEntity) feeding a conveyor delay \
                      (Movable in transit) into a sink."
            .to_string(),
        blocks,
    })
}

/// Tiny constructor alias so demo code reads like a builder.
#[allow(non_snake_case)]
fn StudioGraphBuilder() -> super::graph::StudioGraph {
    super::graph::StudioGraph::new()
}

#[cfg(test)]
mod tests {
    use super::super::run::run;
    use super::*;

    #[test]
    fn signal_chain_block_holds_two_layer2_elements() {
        let demo = signal_chain().unwrap();
        let shaper = demo
            .compiled
            .nodes()
            .iter()
            .find(|n| n.id == "shaper")
            .unwrap();
        assert_eq!(
            shaper.cell.len(),
            2,
            "one block should hold two Layer-2 ops"
        );
    }

    #[test]
    fn signal_chain_saturates_the_ramp() {
        let mut demo = signal_chain().unwrap();
        let run_out = run(&mut demo.compiled, demo.steps, demo.dt);
        // ramp = t; gain 0.5 → 0.5t; saturates at 2 once t ≥ 4 (step 40+).
        let out = run_out.series("output").unwrap();
        assert!((out[10] - 0.5).abs() < 1e-9, "t=1 → 0.5; got {}", out[10]);
        assert!(
            (out[70] - 2.0).abs() < 1e-9,
            "t=7 → saturated 2; got {}",
            out[70]
        );
    }

    #[test]
    fn queue_line_overloads_then_holds_at_service_rate() {
        let mut demo = queue_line().unwrap();
        let run_out = run(&mut demo.compiled, demo.steps, demo.dt);
        // departures = server output (queue·5/tick) delayed 4 ticks by the belt.
        let dep = run_out.series("departures").unwrap();
        // Before the burst (t<3) nothing arrives → nothing departs.
        assert_eq!(dep[0], 0.0);
        // Server output saturates at the service rate (5) once overloaded; after
        // the 4-tick belt delay the sink sees a steady 5.
        let server = run_out.series("server").unwrap();
        assert!((server[10] - 5.0).abs() < 1e-9, "server holds at 5/tick; got {}", server[10]);
        assert!((dep[20] - 5.0).abs() < 1e-9, "departures steady at 5; got {}", dep[20]);
    }

    #[test]
    fn mixer_combines_two_inputs_through_a_two_op_cell() {
        let mut demo = mixer().unwrap();
        let mix = demo
            .compiled
            .nodes()
            .iter()
            .find(|n| n.id == "mix")
            .unwrap();
        assert_eq!(mix.n_in(), 2, "mix block has two input ports");
        assert_eq!(mix.cell.len(), 2, "mix block packs sum ▸ gain");
        let run_out = run(&mut demo.compiled, demo.steps, demo.dt);
        // output = 0.5 * (sine(t) + 0.5). At t=0, sine=0 → 0.25.
        let out = run_out.series("output").unwrap();
        assert!((out[0] - 0.25).abs() < 1e-9, "t=0 → 0.25; got {}", out[0]);
    }
}
