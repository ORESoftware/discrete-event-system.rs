//! A small standard block library for the hybrid engine: the Simulink-style
//! primitives needed to build continuous, discrete, and mixed models. All are
//! `f64`-vector blocks implementing [`Block`].

use super::block::{Block, PortSpec, SampleTime, Signal};

fn matvec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter()
        .map(|row| row.iter().zip(v).map(|(a, b)| a * b).sum())
        .collect()
}

// ---- Sources ---------------------------------------------------------------

/// Constant source: emits a fixed vector.
pub struct Constant {
    name: String,
    value: Vec<f64>,
}

impl Constant {
    pub fn new(name: &str, value: Vec<f64>) -> Self {
        Constant {
            name: name.to_string(),
            value,
        }
    }
    pub fn scalar(name: &str, value: f64) -> Self {
        Constant::new(name, vec![value])
    }
}

impl Block for Constant {
    fn name(&self) -> &str {
        &self.name
    }
    fn port_spec(&self) -> PortSpec {
        PortSpec::source(self.value.len())
    }
    fn sample_time(&self) -> SampleTime {
        SampleTime::Constant
    }
    fn feedthrough(&self) -> bool {
        false
    }
    fn outputs(&self, _t: f64, _xc: &[f64], _xd: &[f64], _u: &[Signal]) -> Vec<Signal> {
        vec![self.value.clone()]
    }
}

// ---- Algebraic (feedthrough) blocks ----------------------------------------

/// Scalar gain applied elementwise: `y = k · u`.
pub struct Gain {
    name: String,
    width: usize,
    k: f64,
}

impl Gain {
    pub fn new(name: &str, width: usize, k: f64) -> Self {
        Gain {
            name: name.to_string(),
            width,
            k,
        }
    }
}

impl Block for Gain {
    fn name(&self) -> &str {
        &self.name
    }
    fn port_spec(&self) -> PortSpec {
        PortSpec::siso(self.width, self.width)
    }
    fn outputs(&self, _t: f64, _xc: &[f64], _xd: &[f64], u: &[Signal]) -> Vec<Signal> {
        vec![u[0].iter().map(|x| self.k * x).collect()]
    }
}

/// Weighted sum of N equal-width inputs: `y = Σ signs[i] · u_i`.
pub struct Sum {
    name: String,
    width: usize,
    signs: Vec<f64>,
}

impl Sum {
    pub fn new(name: &str, width: usize, signs: Vec<f64>) -> Self {
        Sum {
            name: name.to_string(),
            width,
            signs,
        }
    }
}

impl Block for Sum {
    fn name(&self) -> &str {
        &self.name
    }
    fn port_spec(&self) -> PortSpec {
        PortSpec::new(vec![self.width; self.signs.len()], vec![self.width])
    }
    fn outputs(&self, _t: f64, _xc: &[f64], _xd: &[f64], u: &[Signal]) -> Vec<Signal> {
        let mut y = vec![0.0; self.width];
        for (k, sign) in self.signs.iter().enumerate() {
            for i in 0..self.width {
                y[i] += sign * u[k][i];
            }
        }
        vec![y]
    }
}

/// Saturation: `y = clamp(u, lo, hi)` (width-1). Exposes zero-crossings at the
/// two limits so the solver lands precisely where the signal enters/leaves
/// saturation.
pub struct Saturation {
    name: String,
    lo: f64,
    hi: f64,
}

impl Saturation {
    pub fn new(name: &str, lo: f64, hi: f64) -> Self {
        Saturation {
            name: name.to_string(),
            lo,
            hi,
        }
    }
}

impl Block for Saturation {
    fn name(&self) -> &str {
        &self.name
    }
    fn port_spec(&self) -> PortSpec {
        PortSpec::siso(1, 1)
    }
    fn n_zero_crossings(&self) -> usize {
        2
    }
    fn outputs(&self, _t: f64, _xc: &[f64], _xd: &[f64], u: &[Signal]) -> Vec<Signal> {
        vec![vec![u[0][0].clamp(self.lo, self.hi)]]
    }
    fn zero_crossings(&self, _t: f64, _xc: &[f64], _xd: &[f64], u: &[Signal]) -> Vec<f64> {
        vec![u[0][0] - self.lo, u[0][0] - self.hi]
    }
}

// ---- Continuous (stateful) blocks ------------------------------------------

/// Pure integrator: `dx/dt = u`, `y = x` (non-feedthrough, so it breaks
/// algebraic loops).
pub struct Integrator {
    name: String,
    init: Vec<f64>,
}

impl Integrator {
    pub fn new(name: &str, init: Vec<f64>) -> Self {
        Integrator {
            name: name.to_string(),
            init,
        }
    }
}

impl Block for Integrator {
    fn name(&self) -> &str {
        &self.name
    }
    fn port_spec(&self) -> PortSpec {
        PortSpec::siso(self.init.len(), self.init.len())
    }
    fn n_cont(&self) -> usize {
        self.init.len()
    }
    fn init_cont(&self) -> Vec<f64> {
        self.init.clone()
    }
    fn feedthrough(&self) -> bool {
        false
    }
    fn outputs(&self, _t: f64, xc: &[f64], _xd: &[f64], _u: &[Signal]) -> Vec<Signal> {
        vec![xc.to_vec()]
    }
    fn derivatives(&self, _t: f64, _xc: &[f64], _xd: &[f64], u: &[Signal]) -> Vec<f64> {
        u[0].clone()
    }
}

/// Continuous LTI state space: `ẋ = A x + B u`, `y = C x + D u`. Feedthrough iff
/// any `D` entry is nonzero.
pub struct StateSpace {
    name: String,
    a: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
    c: Vec<Vec<f64>>,
    d: Vec<Vec<f64>>,
    n: usize,
    m: usize,
    p: usize,
    x0: Vec<f64>,
}

impl StateSpace {
    /// `a`: n×n, `b`: n×m, `c`: p×n, `d`: p×m. Initial state defaults to zero.
    pub fn new(
        name: &str,
        a: Vec<Vec<f64>>,
        b: Vec<Vec<f64>>,
        c: Vec<Vec<f64>>,
        d: Vec<Vec<f64>>,
    ) -> Self {
        let n = a.len();
        let m = if b.is_empty() { 0 } else { b[0].len() };
        let p = c.len();
        StateSpace {
            name: name.to_string(),
            a,
            b,
            c,
            d,
            n,
            m,
            p,
            x0: vec![0.0; n],
        }
    }

    pub fn with_x0(mut self, x0: Vec<f64>) -> Self {
        self.x0 = x0;
        self
    }
}

impl Block for StateSpace {
    fn name(&self) -> &str {
        &self.name
    }
    fn port_spec(&self) -> PortSpec {
        PortSpec::siso(self.m, self.p)
    }
    fn n_cont(&self) -> usize {
        self.n
    }
    fn init_cont(&self) -> Vec<f64> {
        self.x0.clone()
    }
    fn feedthrough(&self) -> bool {
        self.d.iter().flatten().any(|&x| x != 0.0)
    }
    fn outputs(&self, _t: f64, xc: &[f64], _xd: &[f64], u: &[Signal]) -> Vec<Signal> {
        let mut y = matvec(&self.c, xc); // C x
        if self.m > 0 {
            let du = matvec(&self.d, &u[0]); // D u
            for i in 0..self.p {
                y[i] += du[i];
            }
        }
        vec![y]
    }
    fn derivatives(&self, _t: f64, xc: &[f64], _xd: &[f64], u: &[Signal]) -> Vec<f64> {
        let mut dx = matvec(&self.a, xc); // A x
        if self.m > 0 {
            let bu = matvec(&self.b, &u[0]); // B u
            for i in 0..self.n {
                dx[i] += bu[i];
            }
        }
        dx
    }
}

/// Bouncing ball — a self-contained hybrid example: states `[height, velocity]`,
/// `ẋ = [v, −g]`, a zero-crossing at `height = 0`, and an event that reflects
/// the velocity (`v ← −e·v`) and clamps the height to the floor.
pub struct BouncingBall {
    name: String,
    g: f64,
    restitution: f64,
    h0: f64,
    v0: f64,
}

impl BouncingBall {
    pub fn new(name: &str, h0: f64, v0: f64, restitution: f64) -> Self {
        BouncingBall {
            name: name.to_string(),
            g: 9.81,
            restitution,
            h0,
            v0,
        }
    }
}

impl Block for BouncingBall {
    fn name(&self) -> &str {
        &self.name
    }
    fn port_spec(&self) -> PortSpec {
        // one output port carrying [height, velocity]
        PortSpec::source(2)
    }
    fn n_cont(&self) -> usize {
        2
    }
    fn init_cont(&self) -> Vec<f64> {
        vec![self.h0, self.v0]
    }
    fn feedthrough(&self) -> bool {
        false
    }
    fn n_zero_crossings(&self) -> usize {
        1
    }
    fn outputs(&self, _t: f64, xc: &[f64], _xd: &[f64], _u: &[Signal]) -> Vec<Signal> {
        vec![vec![xc[0], xc[1]]]
    }
    fn derivatives(&self, _t: f64, xc: &[f64], _xd: &[f64], _u: &[Signal]) -> Vec<f64> {
        vec![xc[1], -self.g]
    }
    fn zero_crossings(&self, _t: f64, xc: &[f64], _xd: &[f64], _u: &[Signal]) -> Vec<f64> {
        vec![xc[0]] // height crossing the floor
    }
    fn on_event(&self, _t: f64, xc: &mut [f64], _xd: &mut Vec<f64>, _u: &[Signal], _zc: usize) {
        xc[0] = 0.0; // clamp to floor
        xc[1] *= -self.restitution; // reflect velocity, losing energy
    }
}

// ---- Discrete (sampled) blocks ---------------------------------------------

/// Discrete-time PI controller (with zero-order hold on the output). Input is
/// the scalar error `e`; output is the held command `u`. State `[integral,
/// u_held]`; output is non-feedthrough, so it breaks the plant↔controller
/// algebraic loop.
pub struct DiscretePi {
    name: String,
    kp: f64,
    ki: f64,
    period: f64,
    offset: f64,
}

impl DiscretePi {
    pub fn new(name: &str, kp: f64, ki: f64, period: f64) -> Self {
        DiscretePi {
            name: name.to_string(),
            kp,
            ki,
            period,
            offset: 0.0,
        }
    }
}

impl Block for DiscretePi {
    fn name(&self) -> &str {
        &self.name
    }
    fn port_spec(&self) -> PortSpec {
        PortSpec::siso(1, 1)
    }
    fn sample_time(&self) -> SampleTime {
        SampleTime::Discrete {
            period: self.period,
            offset: self.offset,
        }
    }
    fn n_disc(&self) -> usize {
        2 // [integral, u_held]
    }
    fn feedthrough(&self) -> bool {
        false
    }
    fn outputs(&self, _t: f64, _xc: &[f64], xd: &[f64], _u: &[Signal]) -> Vec<Signal> {
        vec![vec![xd[1]]]
    }
    fn update(&self, _t: f64, xd: &mut Vec<f64>, u: &[Signal]) {
        let e = u[0][0];
        let integral = xd[0] + e * self.period;
        let command = self.kp * e + self.ki * integral;
        xd[0] = integral;
        xd[1] = command;
    }
}

/// Discrete counter (pure-discrete source): increments its state at each sample
/// hit and outputs the count. Useful to exercise the executive with no
/// continuous states.
pub struct Counter {
    name: String,
    period: f64,
}

impl Counter {
    pub fn new(name: &str, period: f64) -> Self {
        Counter {
            name: name.to_string(),
            period,
        }
    }
}

impl Block for Counter {
    fn name(&self) -> &str {
        &self.name
    }
    fn port_spec(&self) -> PortSpec {
        PortSpec::source(1)
    }
    fn sample_time(&self) -> SampleTime {
        SampleTime::Discrete {
            period: self.period,
            offset: 0.0,
        }
    }
    fn n_disc(&self) -> usize {
        1
    }
    fn feedthrough(&self) -> bool {
        false
    }
    fn outputs(&self, _t: f64, _xc: &[f64], xd: &[f64], _u: &[Signal]) -> Vec<Signal> {
        vec![vec![xd[0]]]
    }
    fn update(&self, _t: f64, xd: &mut Vec<f64>, _u: &[Signal]) {
        xd[0] += 1.0;
    }
}
