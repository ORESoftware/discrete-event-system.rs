//! The core block abstraction for the hybrid block-diagram engine.
//!
//! A [`Block`] is a node in a signal-flow [`Diagram`](super::diagram::Diagram):
//! it has typed input/output ports (each a fixed-width vector of `f64`), may own
//! continuous state (integrated by the executive), discrete state (advanced at
//! its own [`SampleTime`]), and may emit zero-crossing functions whose sign
//! changes trigger state events. This is the single block type the hybrid
//! executive understands — the Simulink-style spine the rest of the platform
//! layers onto.
//!
//! ## Execution contract (what the executive calls, and when)
//!
//! * [`Block::outputs`] — pure function of `(t, continuous state, discrete
//!   state, inputs)`. Called whenever the executive needs port values
//!   (every integrator stage, every event probe, every sample hit).
//! * [`Block::derivatives`] — `dx/dt` for continuous state; called at each RK
//!   stage.
//! * [`Block::update`] — advances discrete state in place, called only at this
//!   block's sample hits.
//! * [`Block::zero_crossings`] — values watched for sign changes between steps.
//! * [`Block::on_event`] — fired (with the local zero-crossing index) once the
//!   executive has bisected to the crossing time; may reset state in place.
//!
//! ## Algebraic loops
//!
//! [`Block::feedthrough`] declares whether *any* output depends on the inputs.
//! A direct-feedthrough cycle is an algebraic loop and is rejected at build
//! time. Blocks whose outputs depend only on state (an `Integrator`, a discrete
//! controller holding its last command) return `false` and so *break* loops —
//! exactly as in Simulink.

/// A port value: a fixed-width vector of reals.
pub type Signal = Vec<f64>;

/// Declares a block's port widths: `inputs[i]` / `outputs[j]` is the number of
/// scalar channels on input port `i` / output port `j`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PortSpec {
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
}

impl PortSpec {
    pub fn new(inputs: Vec<usize>, outputs: Vec<usize>) -> Self {
        PortSpec { inputs, outputs }
    }

    /// A source block: no inputs, one output of `width`.
    pub fn source(width: usize) -> Self {
        PortSpec::new(vec![], vec![width])
    }

    /// A sink block: one input of `width`, no outputs.
    pub fn sink(width: usize) -> Self {
        PortSpec::new(vec![width], vec![])
    }

    /// A single-input single-output block.
    pub fn siso(in_w: usize, out_w: usize) -> Self {
        PortSpec::new(vec![in_w], vec![out_w])
    }
}

/// When a block executes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SampleTime {
    /// Continuous-time: state integrated by the ODE solver, outputs evaluated at
    /// every solver stage.
    Continuous,
    /// Discrete: `update` fires at `t = offset + k·period` (k = 0, 1, 2, …).
    Discrete { period: f64, offset: f64 },
    /// Evaluated like a continuous block but holds no state (sources, gains).
    Constant,
}

/// A node in a hybrid block diagram. Implementors override only what they need;
/// the defaults make a stateless, direct-feedthrough block with no events.
pub trait Block {
    /// Human-readable, used for tracing/columns (need not be unique, but it
    /// helps).
    fn name(&self) -> &str;

    /// Port widths.
    fn port_spec(&self) -> PortSpec;

    /// When this block runs. Defaults to continuous.
    fn sample_time(&self) -> SampleTime {
        SampleTime::Continuous
    }

    /// Number of continuous states this block owns.
    fn n_cont(&self) -> usize {
        0
    }

    /// Initial continuous state (length must equal [`Block::n_cont`]).
    fn init_cont(&self) -> Vec<f64> {
        vec![0.0; self.n_cont()]
    }

    /// Number of discrete states this block owns.
    fn n_disc(&self) -> usize {
        0
    }

    /// Initial discrete state (length must equal [`Block::n_disc`]).
    fn init_disc(&self) -> Vec<f64> {
        vec![0.0; self.n_disc()]
    }

    /// Does any output depend on the inputs (direct feedthrough)? Returning
    /// `false` lets the block sit inside a feedback loop without forming an
    /// algebraic loop. Default: `true` (conservative).
    fn feedthrough(&self) -> bool {
        true
    }

    /// Number of zero-crossing functions this block exposes.
    fn n_zero_crossings(&self) -> usize {
        0
    }

    /// Output signals, one per output port. `xc` is *this block's* continuous
    /// state slice, `xd` its discrete state, `u[i]` the signal on input port i.
    fn outputs(&self, t: f64, xc: &[f64], xd: &[f64], u: &[Signal]) -> Vec<Signal>;

    /// Continuous derivatives `dx/dt` (length [`Block::n_cont`]). Default: none.
    fn derivatives(&self, _t: f64, _xc: &[f64], _xd: &[f64], _u: &[Signal]) -> Vec<f64> {
        Vec::new()
    }

    /// Advance discrete state in place at a sample hit. Default: no-op.
    fn update(&self, _t: f64, _xd: &mut Vec<f64>, _u: &[Signal]) {}

    /// Zero-crossing function values (length [`Block::n_zero_crossings`]). A sign
    /// change between solver steps triggers an event. Default: none.
    fn zero_crossings(&self, _t: f64, _xc: &[f64], _xd: &[f64], _u: &[Signal]) -> Vec<f64> {
        Vec::new()
    }

    /// Handle a located event. `zc_index` is the *local* crossing index that
    /// fired; the block may reset its continuous/discrete state in place.
    fn on_event(&self, _t: f64, _xc: &mut [f64], _xd: &mut Vec<f64>, _u: &[Signal], _zc_index: usize) {}
}
