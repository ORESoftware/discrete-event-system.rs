//! Concrete [`Executive`] adapters wrapping the studio dataflow engine and the
//! hybrid signal-flow engine. Each reuses its native engine verbatim and emits a
//! uniform [`RunArtifact`].

use serde_json::{json, Value};

use crate::des::hybrid::{simulate, Compiled, SimOptions};
use crate::des::model::RunArtifact;
use crate::des::plugin::UiControl;
use crate::des::studio::run::run as studio_run;
use crate::des::studio::{CompiledStudio, StudioDemo};

use super::{ExecCapabilities, Executive};

/// Runs a compiled studio (visual-block) graph as acyclic signal dataflow.
pub struct StudioExecutive {
    compiled: CompiledStudio,
    steps: usize,
    dt: f64,
    title: String,
    description: String,
    blocks: Value,
}

impl StudioExecutive {
    pub fn new(
        compiled: CompiledStudio,
        steps: usize,
        dt: f64,
        title: impl Into<String>,
        description: impl Into<String>,
        blocks: Value,
    ) -> Self {
        StudioExecutive {
            compiled,
            steps,
            dt,
            title: title.into(),
            description: description.into(),
            blocks,
        }
    }

    /// Build directly from a studio demo.
    pub fn from_demo(d: StudioDemo) -> Self {
        StudioExecutive::new(d.compiled, d.steps, d.dt, d.title, d.description, d.blocks)
    }
}

impl Executive for StudioExecutive {
    fn kind(&self) -> &'static str {
        "studio"
    }

    fn capabilities(&self) -> ExecCapabilities {
        ExecCapabilities { dataflow: true, discrete: self.compiled.has_state(), ..Default::default() }
    }

    fn run(&mut self) -> RunArtifact {
        let out = studio_run(&mut self.compiled, self.steps, self.dt);
        out.to_artifact("studio", &self.title, &self.description, self.blocks.clone())
    }
}

/// Runs a compiled hybrid block diagram (continuous + discrete + events).
pub struct HybridExecutive {
    compiled: Compiled,
    opts: SimOptions,
    name: String,
    title: String,
    description: String,
}

impl HybridExecutive {
    pub fn new(
        compiled: Compiled,
        opts: SimOptions,
        name: impl Into<String>,
        title: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        HybridExecutive {
            compiled,
            opts,
            name: name.into(),
            title: title.into(),
            description: description.into(),
        }
    }
}

impl Executive for HybridExecutive {
    fn kind(&self) -> &'static str {
        "hybrid"
    }

    fn capabilities(&self) -> ExecCapabilities {
        ExecCapabilities {
            continuous: true,
            discrete: true,
            events: true,
            feedback: true,
            dataflow: true,
            agents: false,
        }
    }

    fn run(&mut self) -> RunArtifact {
        let trace = simulate(&self.compiled, &self.opts);
        let frames = trace.to_jsonl_frames();
        let results = json!({
            "kind": "hybrid",
            "model": self.name,
            "events": trace.events,
            "columns": trace.columns,
            "samples": trace.times.len(),
        });
        let summary = format!(
            "Hybrid `{}` run: {} samples, {} event(s).",
            self.name,
            trace.times.len(),
            trace.events
        );
        RunArtifact::sim(
            "hybrid",
            &self.title,
            &self.description,
            frames,
            results,
            vec![UiControl::range("speed", "Speed (fps)", 1.0, 60.0, 1.0, 20.0)],
            &summary,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::hybrid::demos as hybrid_demos;
    use crate::des::studio::signal_chain;

    #[test]
    fn studio_executive_runs_and_reports_caps() {
        let mut ex = StudioExecutive::from_demo(signal_chain().unwrap());
        assert_eq!(ex.kind(), "studio");
        let caps = ex.capabilities();
        assert!(caps.dataflow);
        let art = ex.run();
        assert_eq!(art.kind, "studio");
        assert!(!art.frames.is_empty());
    }

    #[test]
    fn hybrid_executive_runs_the_bouncing_ball() {
        let (compiled, opts) = hybrid_demos::bouncing_ball().unwrap();
        let mut ex = HybridExecutive::new(
            compiled,
            opts,
            "bouncing-ball",
            "Hybrid Block Diagram",
            "Mixed continuous/discrete/event simulation.",
        );
        assert_eq!(ex.kind(), "hybrid");
        assert!(ex.capabilities().continuous);
        let art = ex.run();
        assert_eq!(art.kind, "hybrid");
        assert!(!art.frames.is_empty());
    }
}
