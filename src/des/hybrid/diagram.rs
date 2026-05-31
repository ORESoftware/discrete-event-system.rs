//! Building and compiling a block diagram: wiring with width checks, topological
//! ordering of direct-feedthrough outputs (which detects algebraic loops), and
//! the numeric kernel the executive drives (output propagation, derivative
//! assembly, an RK4 step, zero-crossing evaluation + location, event firing).

use super::block::{Block, SampleTime, Signal};

/// Opaque handle to a block added to a [`Diagram`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockHandle(pub usize);

/// A directed connection from one block's output port to another's input port.
#[derive(Clone, Copy, Debug)]
pub struct Wire {
    pub src: usize,
    pub src_port: usize,
    pub dst: usize,
    pub dst_port: usize,
}

/// Errors raised while wiring or compiling a diagram.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HybridError {
    PortOutOfRange {
        block: String,
        port: usize,
        is_input: bool,
    },
    WidthMismatch {
        from: String,
        to: String,
        src_width: usize,
        dst_width: usize,
    },
    InputAlreadyDriven {
        block: String,
        port: usize,
    },
    /// A direct-feedthrough cycle: these blocks form an algebraic loop. Break it
    /// by inserting a non-feedthrough block (Integrator, UnitDelay, a discrete
    /// controller that holds its output, …).
    AlgebraicLoop(Vec<String>),
}

impl std::fmt::Display for HybridError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HybridError::PortOutOfRange { block, port, is_input } => write!(
                f,
                "{} port {port} out of range on block `{block}`",
                if *is_input { "input" } else { "output" }
            ),
            HybridError::WidthMismatch { from, to, src_width, dst_width } => write!(
                f,
                "width mismatch wiring `{from}` ({src_width}) -> `{to}` ({dst_width})"
            ),
            HybridError::InputAlreadyDriven { block, port } => {
                write!(f, "input port {port} of `{block}` already driven by another wire")
            }
            HybridError::AlgebraicLoop(names) => {
                write!(f, "algebraic loop among direct-feedthrough blocks: {}", names.join(" -> "))
            }
        }
    }
}

impl std::error::Error for HybridError {}

/// A mutable block diagram under construction.
#[derive(Default)]
pub struct Diagram {
    blocks: Vec<Box<dyn Block>>,
    wires: Vec<Wire>,
}

impl Diagram {
    pub fn new() -> Self {
        Diagram::default()
    }

    /// Add a block, returning a handle to wire it.
    pub fn add(&mut self, block: Box<dyn Block>) -> BlockHandle {
        self.blocks.push(block);
        BlockHandle(self.blocks.len() - 1)
    }

    /// Connect `from = (block, output_port)` to `to = (block, input_port)`.
    /// Validates port indices, equal widths, and that the input is not already
    /// driven.
    pub fn connect(
        &mut self,
        from: (BlockHandle, usize),
        to: (BlockHandle, usize),
    ) -> Result<(), HybridError> {
        let (src, src_port) = (from.0 .0, from.1);
        let (dst, dst_port) = (to.0 .0, to.1);
        let src_spec = self.blocks[src].port_spec();
        let dst_spec = self.blocks[dst].port_spec();
        if src_port >= src_spec.outputs.len() {
            return Err(HybridError::PortOutOfRange {
                block: self.blocks[src].name().to_string(),
                port: src_port,
                is_input: false,
            });
        }
        if dst_port >= dst_spec.inputs.len() {
            return Err(HybridError::PortOutOfRange {
                block: self.blocks[dst].name().to_string(),
                port: dst_port,
                is_input: true,
            });
        }
        let sw = src_spec.outputs[src_port];
        let dw = dst_spec.inputs[dst_port];
        if sw != dw {
            return Err(HybridError::WidthMismatch {
                from: self.blocks[src].name().to_string(),
                to: self.blocks[dst].name().to_string(),
                src_width: sw,
                dst_width: dw,
            });
        }
        if self.wires.iter().any(|w| w.dst == dst && w.dst_port == dst_port) {
            return Err(HybridError::InputAlreadyDriven {
                block: self.blocks[dst].name().to_string(),
                port: dst_port,
            });
        }
        self.wires.push(Wire { src, src_port, dst, dst_port });
        Ok(())
    }

    /// Validate and compile into a [`Compiled`] diagram ready to simulate.
    pub fn build(self) -> Result<Compiled, HybridError> {
        let n = self.blocks.len();
        let names: Vec<String> = self.blocks.iter().map(|b| b.name().to_string()).collect();
        let port_specs: Vec<_> = self.blocks.iter().map(|b| b.port_spec()).collect();
        let input_widths: Vec<Vec<usize>> = port_specs.iter().map(|p| p.inputs.clone()).collect();
        let output_widths: Vec<Vec<usize>> = port_specs.iter().map(|p| p.outputs.clone()).collect();

        // Continuous-state layout.
        let cont_n: Vec<usize> = self.blocks.iter().map(|b| b.n_cont()).collect();
        let mut cont_offset = vec![0usize; n];
        let mut n_cont_total = 0;
        for b in 0..n {
            cont_offset[b] = n_cont_total;
            n_cont_total += cont_n[b];
        }

        // Input source map: in_sources[b][port] = Some((src_block, src_port)).
        let mut in_sources: Vec<Vec<Option<(usize, usize)>>> =
            input_widths.iter().map(|ws| vec![None; ws.len()]).collect();
        for w in &self.wires {
            in_sources[w.dst][w.dst_port] = Some((w.src, w.src_port));
        }

        // Feedthrough flags + topological order (Kahn). Edge src->dst only when
        // dst is direct-feedthrough (its outputs need its inputs first).
        let feedthrough: Vec<bool> = self.blocks.iter().map(|b| b.feedthrough()).collect();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut indeg = vec![0usize; n];
        for w in &self.wires {
            if feedthrough[w.dst] && w.src != w.dst {
                adj[w.src].push(w.dst);
                indeg[w.dst] += 1;
            } else if feedthrough[w.dst] && w.src == w.dst {
                // a feedthrough block feeding itself is a trivial algebraic loop.
                return Err(HybridError::AlgebraicLoop(vec![names[w.dst].clone()]));
            }
        }
        let mut queue: Vec<usize> = (0..n).filter(|&b| indeg[b] == 0).collect();
        let mut topo = Vec::with_capacity(n);
        let mut head = 0;
        while head < queue.len() {
            let b = queue[head];
            head += 1;
            topo.push(b);
            for &m in &adj[b] {
                indeg[m] -= 1;
                if indeg[m] == 0 {
                    queue.push(m);
                }
            }
        }
        if topo.len() != n {
            let loop_blocks: Vec<String> = (0..n)
                .filter(|&b| indeg[b] > 0)
                .map(|b| names[b].clone())
                .collect();
            return Err(HybridError::AlgebraicLoop(loop_blocks));
        }

        // Zero-crossing ownership: global index -> (block, local index).
        let mut zc_owner = Vec::new();
        for (b, blk) in self.blocks.iter().enumerate() {
            for local in 0..blk.n_zero_crossings() {
                zc_owner.push((b, local));
            }
        }

        // Discrete schedule descriptors.
        let mut disc = Vec::new();
        for (b, blk) in self.blocks.iter().enumerate() {
            if let SampleTime::Discrete { period, offset } = blk.sample_time() {
                disc.push((b, period, offset));
            }
        }

        Ok(Compiled {
            blocks: self.blocks,
            topo,
            cont_offset,
            cont_n,
            n_cont_total,
            input_widths,
            output_widths,
            in_sources,
            zc_owner,
            disc,
            names,
        })
    }
}

/// A validated diagram with precomputed execution metadata. The numeric kernel
/// (propagate / derivatives / RK4 / zero-crossings / events) lives here; the
/// [`super::executive`] drives it.
pub struct Compiled {
    pub(super) blocks: Vec<Box<dyn Block>>,
    topo: Vec<usize>,
    cont_offset: Vec<usize>,
    cont_n: Vec<usize>,
    pub(super) n_cont_total: usize,
    input_widths: Vec<Vec<usize>>,
    output_widths: Vec<Vec<usize>>,
    in_sources: Vec<Vec<Option<(usize, usize)>>>,
    zc_owner: Vec<(usize, usize)>,
    /// (block index, period, offset) for each discrete block.
    pub(super) disc: Vec<(usize, f64, f64)>,
    pub(super) names: Vec<String>,
}

impl Compiled {
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn output_widths(&self) -> &[Vec<usize>] {
        &self.output_widths
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Total number of zero-crossing functions.
    pub fn n_zero_crossings(&self) -> usize {
        self.zc_owner.len()
    }

    /// Global initial continuous state vector.
    pub fn init_cont_global(&self) -> Vec<f64> {
        let mut x = vec![0.0; self.n_cont_total];
        for (b, blk) in self.blocks.iter().enumerate() {
            if self.cont_n[b] == 0 {
                continue;
            }
            let init = blk.init_cont();
            let off = self.cont_offset[b];
            for i in 0..self.cont_n[b] {
                x[off + i] = init.get(i).copied().unwrap_or(0.0);
            }
        }
        x
    }

    /// Per-block initial discrete state.
    pub fn init_disc_all(&self) -> Vec<Vec<f64>> {
        self.blocks.iter().map(|b| b.init_disc()).collect()
    }

    fn cont_slice<'a>(&self, xc: &'a [f64], b: usize) -> &'a [f64] {
        let n = self.cont_n[b];
        if n == 0 {
            &[]
        } else {
            let off = self.cont_offset[b];
            &xc[off..off + n]
        }
    }

    /// Gather the input signals for block `b` from already-computed `outs`,
    /// substituting a zero vector for any unconnected input port.
    fn gather(&self, b: usize, outs: &[Vec<Signal>]) -> Vec<Signal> {
        let mut u = Vec::with_capacity(self.input_widths[b].len());
        for (port, &w) in self.input_widths[b].iter().enumerate() {
            match self.in_sources[b][port] {
                Some((sb, sp)) => u.push(outs[sb][sp].clone()),
                None => u.push(vec![0.0; w]),
            }
        }
        u
    }

    /// Evaluate every block's outputs at `(t, xc, xd)`, in feedthrough
    /// topological order so each feedthrough block reads finalized inputs.
    pub(super) fn propagate(&self, t: f64, xc: &[f64], xd: &[Vec<f64>]) -> Vec<Vec<Signal>> {
        // Initialize with correctly-sized zero signals so consumers always read
        // a well-formed vector even before a producer has run.
        let mut outs: Vec<Vec<Signal>> = self
            .output_widths
            .iter()
            .map(|ws| ws.iter().map(|&w| vec![0.0; w]).collect())
            .collect();
        for &b in &self.topo {
            let u = self.gather(b, &outs);
            let xc_b = self.cont_slice(xc, b);
            outs[b] = self.blocks[b].outputs(t, xc_b, &xd[b], &u);
        }
        outs
    }

    /// Assemble the global derivative vector `dx/dt` at `(t, xc, xd)`.
    fn assemble_xdot(&self, t: f64, xc: &[f64], xd: &[Vec<f64>]) -> Vec<f64> {
        let outs = self.propagate(t, xc, xd);
        let mut xdot = vec![0.0; self.n_cont_total];
        for b in 0..self.blocks.len() {
            if self.cont_n[b] == 0 {
                continue;
            }
            let u = self.gather(b, &outs);
            let xc_b = self.cont_slice(xc, b);
            let d = self.blocks[b].derivatives(t, xc_b, &xd[b], &u);
            let off = self.cont_offset[b];
            for i in 0..self.cont_n[b] {
                xdot[off + i] = d.get(i).copied().unwrap_or(0.0);
            }
        }
        xdot
    }

    /// One classical RK4 step of the continuous state over `h`, holding discrete
    /// state `xd` constant across the step. Returns the new state (empty if the
    /// diagram has no continuous states).
    pub(super) fn rk4_step(&self, t: f64, xc: &[f64], xd: &[Vec<f64>], h: f64) -> Vec<f64> {
        if self.n_cont_total == 0 {
            return Vec::new();
        }
        let add = |a: &[f64], b: &[f64], s: f64| -> Vec<f64> {
            a.iter().zip(b).map(|(x, y)| x + s * y).collect()
        };
        let k1 = self.assemble_xdot(t, xc, xd);
        let x2 = add(xc, &k1, h * 0.5);
        let k2 = self.assemble_xdot(t + h * 0.5, &x2, xd);
        let x3 = add(xc, &k2, h * 0.5);
        let k3 = self.assemble_xdot(t + h * 0.5, &x3, xd);
        let x4 = add(xc, &k3, h);
        let k4 = self.assemble_xdot(t + h, &x4, xd);
        let mut out = xc.to_vec();
        for i in 0..out.len() {
            out[i] += (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        out
    }

    /// Concatenated zero-crossing values at `(t, xc, xd)`.
    pub(super) fn zc_values(&self, t: f64, xc: &[f64], xd: &[Vec<f64>]) -> Vec<f64> {
        if self.zc_owner.is_empty() {
            return Vec::new();
        }
        let outs = self.propagate(t, xc, xd);
        let mut z = Vec::with_capacity(self.zc_owner.len());
        for b in 0..self.blocks.len() {
            if self.blocks[b].n_zero_crossings() == 0 {
                continue;
            }
            let u = self.gather(b, &outs);
            let xc_b = self.cont_slice(xc, b);
            z.extend(self.blocks[b].zero_crossings(t, xc_b, &xd[b], &u));
        }
        z
    }

    /// Detect the first zero-crossing in `(t, t+h]` given the trial end state.
    /// Returns `(global_zc_index, offset_into_step)` where the sign first flips.
    pub(super) fn find_crossing(
        &self,
        t: f64,
        xc: &[f64],
        x_trial: &[f64],
        xd: &[Vec<f64>],
        h: f64,
        tol: f64,
    ) -> Option<(usize, f64)> {
        if self.zc_owner.is_empty() {
            return None;
        }
        let z0 = self.zc_values(t, xc, xd);
        let z1 = self.zc_values(t + h, x_trial, xd);
        let mut gi = None;
        for i in 0..z0.len() {
            if z0[i].abs() < 1e-12 {
                continue; // already on the surface; don't re-trigger
            }
            if z0[i] * z1[i] < 0.0 {
                gi = Some(i);
                break;
            }
        }
        let gi = gi?;
        let s0 = z0[gi].signum();
        let (mut lo, mut hi) = (0.0f64, h);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            let xm = self.rk4_step(t, xc, xd, mid);
            let zm = self.zc_values(t + mid, &xm, xd)[gi];
            if zm.signum() == s0 {
                lo = mid;
            } else {
                hi = mid;
            }
            if hi - lo < tol {
                break;
            }
        }
        Some((gi, hi))
    }

    /// Fire the event owned by global zero-crossing index `gi` at time `tc`,
    /// letting the owning block reset its state in place.
    pub(super) fn fire_event(&self, gi: usize, tc: f64, xc: &mut [f64], xd: &mut [Vec<f64>]) {
        let (b, local) = self.zc_owner[gi];
        let outs = self.propagate(tc, xc, xd);
        let u = self.gather(b, &outs);
        let n = self.cont_n[b];
        if n > 0 {
            let off = self.cont_offset[b];
            let (left, rest) = xc.split_at_mut(off);
            let _ = left;
            let xc_b = &mut rest[..n];
            self.blocks[b].on_event(tc, xc_b, &mut xd[b], &u, local);
        } else {
            let mut empty: [f64; 0] = [];
            self.blocks[b].on_event(tc, &mut empty, &mut xd[b], &u, local);
        }
    }

    /// Run discrete `update` for block `b` using inputs gathered from `outs`.
    pub(super) fn discrete_update(&self, b: usize, t: f64, xd: &mut [Vec<f64>], outs: &[Vec<Signal>]) {
        let u = self.gather(b, outs);
        let mut state = std::mem::take(&mut xd[b]);
        self.blocks[b].update(t, &mut state, &u);
        xd[b] = state;
    }
}
