//! Canonical studio demos. Each is a *flat* graph of visual blocks where at
//! least one block packs several Layer-2 elements into a single cell — exactly
//! the two architectural constraints (multi-element blocks, no nesting).

use serde_json::{json, Value};

use super::cell::{Gain, RuntimeCell, Saturation, Source, SourceKind, Sum};
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
