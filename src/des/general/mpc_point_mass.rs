//! Reusable planar (2-D) point-mass Model Predictive Controller.
//!
//! This generalises [`super::mpc_double_integrator`] from a scalar
//! origin-regulator demo into a *stateful, single-step, reference-tracking*
//! controller suitable for driving an agent toward a moving target every tick of
//! a real-time simulation:
//!
//!   * **2-D double integrator.** State is `(pos[2], vel[2])`, control is an
//!     acceleration `a[2]`. Each axis evolves with the exact piecewise-constant-`u`
//!     discretisation `p' = p + dt·v + ½·dt²·a`, `v' = v + dt·a` (identical to the
//!     1-D plant), and the two axes couple *only* through the acceleration limit.
//!
//!   * **Reference tracking.** The stage/terminal cost penalises deviation of the
//!     predicted position and velocity from a reference, not from the origin. The
//!     reference may be a single broadcast target (arrive-and-stop with `vel = 0`,
//!     or run-through-space with `vel = cruise·dir`) or a full per-step reference
//!     trajectory (e.g. a predicted interception path).
//!
//!   * **Circular acceleration limit.** The biomechanically-correct constraint
//!     `‖a‖ ≤ a_max` (a disk, not a box) is the natural projection in a
//!     projected-gradient solver: each control is projected onto the disk after
//!     every gradient step. A box constraint would be both less realistic *and*
//!     more code here.
//!
//!   * **Receding horizon, single step.** [`PlanarPointMassMpc::control`] solves
//!     the horizon-`N` QP from the current measured state, applies (returns) only
//!     the first control, shifts the warm start, and returns — the caller advances
//!     its own plant and calls again next tick. This is the loop an external
//!     simulator (the soccer engine) drives, in contrast to the self-contained
//!     closed-loop driver in [`super::mpc_double_integrator`].
//!
//! Why projected gradient and not [`super::qp::solve_qp_active_set`]: the
//! active-set routine enumerates constraint subsets (`2^m` masks), and an
//! `N`-step problem with per-step acceleration limits has `O(N)` constraints, so
//! enumeration is exponential in the horizon. Projected gradient is `O(N)` per
//! iteration and warm-starts across ticks, which is what real-time MPC needs.

use crate::des::general::des_base::preconditions::{PreconditionError, Preconditions};

/// A reference the controller tracks at one horizon step: a desired position and
/// a desired velocity. Use `vel = [0, 0]` for "arrive and stop", or
/// `vel = cruise_speed · unit_direction` for "run through this point at speed".
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanarReference {
    pub pos: [f64; 2],
    pub vel: [f64; 2],
}

impl PlanarReference {
    /// Arrive at `pos` and stop (zero target velocity).
    pub fn arrive(pos: [f64; 2]) -> Self {
        PlanarReference {
            pos,
            vel: [0.0, 0.0],
        }
    }

    /// Pass through `pos` carrying velocity `vel`.
    pub fn through(pos: [f64; 2], vel: [f64; 2]) -> Self {
        PlanarReference { pos, vel }
    }
}

/// Current measured state of the controlled point mass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanarState {
    pub pos: [f64; 2],
    pub vel: [f64; 2],
}

/// A moving soft keep-out region the controller should avoid — the generic
/// mechanism for making the plan aware of *other agents* (their position **and**
/// velocity). The obstacle's centre is propagated along its `velocity` over the
/// horizon (constant-velocity prediction), so the plan bends around where a mover
/// *will be*, not just where it is now. The penalty is soft (a quadratic well that
/// is zero outside `radius`), so it shapes the trajectory without ever making the
/// QP infeasible — a controller boxed in on all sides still returns a best-effort
/// control rather than failing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanarObstacle {
    /// Obstacle centre at horizon step 0.
    pub center: [f64; 2],
    /// Predicted constant velocity of the obstacle (units/second).
    pub velocity: [f64; 2],
    /// Keep-out radius: the penalty is zero at/beyond this distance and grows
    /// quadratically as the plan pushes inside it.
    pub radius: f64,
    /// Penalty weight (cost per yard² of incursion). Larger = harder avoidance.
    pub weight: f64,
}

impl PlanarObstacle {
    /// A stationary keep-out at `center`.
    pub fn fixed(center: [f64; 2], radius: f64, weight: f64) -> Self {
        PlanarObstacle {
            center,
            velocity: [0.0, 0.0],
            radius,
            weight,
        }
    }
}

/// Configuration for [`PlanarPointMassMpc`]. All weights are non-negative; `r`,
/// `a_max`, `dt` and the horizon must be strictly positive.
#[derive(Clone, Copy, Debug)]
pub struct PlanarMpcConfig {
    /// Horizon length `N` (number of control steps looked ahead).
    pub horizon: usize,
    /// Sampling period (seconds). Should match the simulation tick.
    pub dt: f64,
    /// Stage position-tracking weight.
    pub q_pos: f64,
    /// Stage velocity-tracking weight.
    pub q_vel: f64,
    /// Terminal position-tracking weight.
    pub qf_pos: f64,
    /// Terminal velocity-tracking weight.
    pub qf_vel: f64,
    /// Control-effort weight (`R`); MUST be `> 0` or the QP is ill-posed.
    pub r: f64,
    /// Acceleration magnitude limit (disk radius), in distance/second².
    pub a_max: f64,
    /// Projected-gradient iterations per solve.
    pub iters: usize,
    /// Per-step multiplicative decay applied to obstacle penalty weights as the
    /// horizon extends: the effective weight at step `i` is `weight · decay^i`.
    /// `1.0` (default) trusts every predicted obstacle position equally — fine for
    /// a short horizon. For a multi-second horizon set it `< 1` so the plan trusts
    /// near-term (reliable) obstacle predictions and discounts the far-future ones
    /// (constant-velocity extrapolation is fiction several seconds out), instead of
    /// swerving around where an opponent will *never actually* be. Must be in (0, 1].
    pub obstacle_decay_per_step: f64,
}

impl Default for PlanarMpcConfig {
    fn default() -> Self {
        // Defaults tuned for a ~15 Hz sim driving a runner to a target a few
        // metres away: short horizon (≈0.8 s at dt=1/15), position-dominant cost.
        PlanarMpcConfig {
            horizon: 12,
            dt: 1.0 / 15.0,
            q_pos: 10.0,
            q_vel: 1.0,
            qf_pos: 50.0,
            qf_vel: 5.0,
            r: 0.1,
            a_max: 6.0,
            iters: 60,
            obstacle_decay_per_step: 1.0,
        }
    }
}

/// Stateful receding-horizon controller. Construct once per controlled agent,
/// then call [`PlanarPointMassMpc::control`] every tick. The warm start persists
/// between calls, so successive solves converge in very few iterations.
#[derive(Clone, Debug)]
pub struct PlanarPointMassMpc {
    cfg: PlanarMpcConfig,
    // Discretised single-axis dynamics coefficients (shared by both axes).
    b_p: f64, // ½ dt²  — control → position
    b_v: f64, // dt     — control → velocity
    // Warm-start control sequence, one [ax, ay] per horizon step.
    warm: Vec<[f64; 2]>,
}

impl PlanarPointMassMpc {
    pub fn new(cfg: PlanarMpcConfig) -> Result<Self, PreconditionError> {
        let cls = "PlanarPointMassMpc";
        Preconditions::integer_in_range(cls, "horizon", cfg.horizon as f64, 1.0, 1000.0)?;
        Preconditions::positive(cls, "dt", cfg.dt)?;
        Preconditions::non_negative(cls, "q_pos", cfg.q_pos)?;
        Preconditions::non_negative(cls, "q_vel", cfg.q_vel)?;
        Preconditions::non_negative(cls, "qf_pos", cfg.qf_pos)?;
        Preconditions::non_negative(cls, "qf_vel", cfg.qf_vel)?;
        // R appears as 2·R·u in the gradient; R = 0 leaves the control
        // unbounded along the cost nullspace.
        Preconditions::positive(cls, "r", cfg.r)?;
        Preconditions::positive(cls, "a_max", cfg.a_max)?;
        Preconditions::integer_in_range(cls, "iters", cfg.iters as f64, 1.0, 100_000.0)?;
        Preconditions::in_range(
            cls,
            "obstacle_decay_per_step",
            cfg.obstacle_decay_per_step,
            f64::MIN_POSITIVE,
            1.0,
        )?;
        Ok(PlanarPointMassMpc {
            b_p: 0.5 * cfg.dt * cfg.dt,
            b_v: cfg.dt,
            warm: vec![[0.0, 0.0]; cfg.horizon],
            cfg,
        })
    }

    /// Project an acceleration onto the disk `‖a‖ ≤ a_max`. Any non-finite
    /// component collapses to zero first, so a NaN/∞ can never propagate out of
    /// the controller (and can never be stored back into the warm start).
    fn project_disk(&self, a: [f64; 2]) -> [f64; 2] {
        let a = [
            if a[0].is_finite() { a[0] } else { 0.0 },
            if a[1].is_finite() { a[1] } else { 0.0 },
        ];
        let mag = (a[0] * a[0] + a[1] * a[1]).sqrt();
        if mag <= self.cfg.a_max || mag <= 0.0 {
            return a;
        }
        let s = self.cfg.a_max / mag;
        [a[0] * s, a[1] * s]
    }

    /// True iff a 2-D vector is fully finite.
    fn finite2(v: [f64; 2]) -> bool {
        v[0].is_finite() && v[1].is_finite()
    }

    /// Resolve the reference at horizon step `i` (0-based; step `N` is terminal).
    /// A single-element slice broadcasts; otherwise it is read per step and the
    /// last entry is held for any steps beyond its length.
    fn reference_at(refs: &[PlanarReference], i: usize) -> PlanarReference {
        if refs.len() == 1 {
            return refs[0];
        }
        let idx = i.min(refs.len() - 1);
        refs[idx]
    }

    /// Soft keep-out penalty (and its position gradient) for one predicted state
    /// `p` at horizon step `i`. Each obstacle is propagated to step `i` by its
    /// velocity. The well is `weight · (radius − dist)²` inside `radius`, zero
    /// outside — smooth at the boundary, with a finite gradient everywhere (a
    /// degenerate exactly-on-centre case pushes along +x to avoid a 0/0).
    fn obstacle_cost_grad(
        &self,
        p: [f64; 2],
        i: usize,
        obstacles: &[PlanarObstacle],
    ) -> (f64, [f64; 2]) {
        let mut cost = 0.0;
        let mut grad = [0.0, 0.0];
        let t = self.cfg.dt * i as f64;
        // Trust near-term obstacle predictions, discount far-future ones (their
        // constant-velocity extrapolation is unreliable). `decay == 1` ⇒ no change.
        let decay = if self.cfg.obstacle_decay_per_step >= 1.0 {
            1.0
        } else {
            self.cfg.obstacle_decay_per_step.powi(i as i32)
        };
        for ob in obstacles {
            let weight = ob.weight * decay;
            if !(weight > 0.0) || !(ob.radius > 0.0) {
                continue;
            }
            let cx = ob.center[0] + ob.velocity[0] * t;
            let cy = ob.center[1] + ob.velocity[1] * t;
            let dx = p[0] - cx;
            let dy = p[1] - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist >= ob.radius {
                continue;
            }
            let incursion = ob.radius - dist;
            cost += weight * incursion * incursion;
            // d cost/d p = 2·w·(radius−dist)·(−d dist/d p) = −2·w·incursion·(p−c)/dist.
            // Pushes the plan radially OUTWARD (away from the obstacle centre).
            if dist > 1e-9 {
                let g = -2.0 * weight * incursion / dist;
                grad[0] += g * dx;
                grad[1] += g * dy;
            } else {
                // On the centre: push out along +x (arbitrary but finite).
                grad[0] += -2.0 * weight * ob.radius;
            }
        }
        (cost, grad)
    }

    /// Forward rollout + adjoint backward sweep: returns `(cost, gradient)` of the
    /// horizon objective with respect to the control sequence. Reference tracking
    /// and double-integrator dynamics are axis-separable; the soft obstacle field
    /// couples the axes (distance is 2-D), so the forward rollout is done once in
    /// 2-D, obstacle penalties + gradients are accumulated per step, and the
    /// adjoint sweep runs per axis using the shared per-step position gradient.
    fn cost_and_grad(
        &self,
        state: PlanarState,
        refs: &[PlanarReference],
        obstacles: &[PlanarObstacle],
        useq: &[[f64; 2]],
    ) -> (f64, Vec<[f64; 2]>) {
        let n = self.cfg.horizon;
        let (q_pos, q_vel, qf_pos, qf_vel, r) = (
            self.cfg.q_pos,
            self.cfg.q_vel,
            self.cfg.qf_pos,
            self.cfg.qf_vel,
            self.cfg.r,
        );
        let dt = self.cfg.dt;

        // Single 2-D forward rollout.
        let mut ps = vec![[0.0_f64; 2]; n + 1];
        let mut vs = vec![[0.0_f64; 2]; n + 1];
        ps[0] = state.pos;
        vs[0] = state.vel;
        for i in 0..n {
            for d in 0..2 {
                ps[i + 1][d] = ps[i][d] + dt * vs[i][d] + self.b_p * useq[i][d];
                vs[i + 1][d] = vs[i][d] + self.b_v * useq[i][d];
            }
        }

        // Per-step position-gradient of all *state*-dependent costs (tracking +
        // obstacles), plus the running cost. `pos_grad[i]` = ∂(stage_i)/∂p_i.
        let mut cost = 0.0;
        let mut pos_grad = vec![[0.0_f64; 2]; n + 1];
        for i in 0..=n {
            let rf = Self::reference_at(refs, i);
            let (wq_pos, wq_vel) = if i == n {
                (qf_pos, qf_vel)
            } else {
                (q_pos, q_vel)
            };
            for d in 0..2 {
                let dp = ps[i][d] - rf.pos[d];
                let dv = vs[i][d] - rf.vel[d];
                cost += wq_pos * dp * dp + wq_vel * dv * dv;
                pos_grad[i][d] += 2.0 * wq_pos * dp;
                // velocity-tracking gradient is handled in the adjoint via lam_v seed
                // for the terminal and via the per-step ∂/∂v below.
            }
            if i < n {
                let u = useq[i];
                cost += r * (u[0] * u[0] + u[1] * u[1]);
            }
            if !obstacles.is_empty() {
                let (oc, og) = self.obstacle_cost_grad(ps[i], i, obstacles);
                cost += oc;
                pos_grad[i][0] += og[0];
                pos_grad[i][1] += og[1];
            }
        }

        // Per-axis adjoint backward sweep using the shared per-step position
        // gradient. λ = (λ_p, λ_v), A = [[1, dt],[0,1]], B = [b_p, b_v].
        let mut grad = vec![[0.0; 2]; n];
        for d in 0..2 {
            let rf_n = Self::reference_at(refs, n);
            let dvn = vs[n][d] - rf_n.vel[d];
            let mut lam_p = pos_grad[n][d];
            let mut lam_v = 2.0 * qf_vel * dvn;
            for i in (0..n).rev() {
                grad[i][d] = 2.0 * r * useq[i][d] + self.b_p * lam_p + self.b_v * lam_v;
                let rf = Self::reference_at(refs, i);
                let dv = vs[i][d] - rf.vel[d];
                let lam_p_new = pos_grad[i][d] + lam_p;
                let lam_v_new = 2.0 * q_vel * dv + dt * lam_p + lam_v;
                lam_p = lam_p_new;
                lam_v = lam_v_new;
            }
        }

        (cost, grad)
    }

    /// Projected-gradient descent on the disk-constrained control sequence. Every
    /// candidate goes through [`Self::project_disk`], which both enforces the
    /// `a_max` constraint and scrubs non-finite values, so the returned sequence
    /// is always finite and feasible. A non-finite cost (only reachable via a
    /// non-finite reference, which the caller already guards) aborts the descent
    /// early with the feasible warm start rather than iterating on garbage.
    fn solve(
        &self,
        state: PlanarState,
        refs: &[PlanarReference],
        obstacles: &[PlanarObstacle],
    ) -> Vec<[f64; 2]> {
        let n = self.cfg.horizon;
        let mut useq = self.warm.clone();
        // Ensure the warm start itself is feasible (and finite).
        for u in useq.iter_mut() {
            *u = self.project_disk(*u);
        }
        let mut alpha = 0.05_f64;
        let mut last_cost = f64::INFINITY;
        for _ in 0..self.cfg.iters {
            let (cost, grad) = self.cost_and_grad(state, refs, obstacles, &useq);
            if !cost.is_finite() {
                break;
            }
            if cost > last_cost - 1e-10 {
                alpha *= 0.5;
            } else {
                alpha = (alpha * 1.05).min(0.1);
            }
            last_cost = cost;
            let mut step_norm = 0.0;
            for i in 0..n {
                let candidate = [
                    useq[i][0] - alpha * grad[i][0],
                    useq[i][1] - alpha * grad[i][1],
                ];
                let projected = self.project_disk(candidate);
                step_norm +=
                    (projected[0] - useq[i][0]).powi(2) + (projected[1] - useq[i][1]).powi(2);
                useq[i] = projected;
            }
            if step_norm.sqrt() < 1e-4 {
                break;
            }
        }
        useq
    }

    /// Solve the receding-horizon problem from `state` toward `refs` and return
    /// the first control (`[ax, ay]`, already inside the `a_max` disk). The warm
    /// start is advanced by one step so the next call converges quickly.
    ///
    /// `refs` is either a single broadcast reference or a per-step trajectory of
    /// length `≤ horizon + 1`; it must be non-empty.
    ///
    /// Robust to bad inputs: an empty `refs`, or any non-finite component in the
    /// state or references, coasts (returns `[0, 0]`) and leaves the warm start
    /// untouched — a corrupt measurement can never poison the controller or emit a
    /// NaN/∞ control.
    pub fn control(&mut self, state: PlanarState, refs: &[PlanarReference]) -> [f64; 2] {
        self.control_with_obstacles(state, refs, &[])
    }

    /// Like [`Self::control`], but the plan also avoids a set of moving soft
    /// obstacles — the field-aware path that lets an agent's trajectory bend
    /// around where *other agents* (opponents, teammates, the ball) are predicted
    /// to be over the horizon. Non-finite obstacles are skipped; obstacles never
    /// make the solve fail (the penalty is soft).
    pub fn control_with_obstacles(
        &mut self,
        state: PlanarState,
        refs: &[PlanarReference],
        obstacles: &[PlanarObstacle],
    ) -> [f64; 2] {
        if refs.is_empty() {
            // Degenerate: no target — coast (no acceleration).
            return [0.0, 0.0];
        }
        if !Self::finite2(state.pos)
            || !Self::finite2(state.vel)
            || refs
                .iter()
                .any(|r| !Self::finite2(r.pos) || !Self::finite2(r.vel))
        {
            return [0.0, 0.0];
        }
        // Drop any non-finite obstacle so a bad measurement can't poison the plan.
        let clean: Vec<PlanarObstacle> = obstacles
            .iter()
            .copied()
            .filter(|o| {
                Self::finite2(o.center)
                    && Self::finite2(o.velocity)
                    && o.radius.is_finite()
                    && o.weight.is_finite()
            })
            .collect();
        let useq = self.solve(state, refs, &clean);
        let first = useq[0];
        // Shift the warm start: drop the applied control, append a zero.
        let n = self.cfg.horizon;
        for i in 0..n - 1 {
            self.warm[i] = useq[i + 1];
        }
        self.warm[n - 1] = [0.0, 0.0];
        self.project_disk(first)
    }

    /// Predicted position trajectory (length `horizon + 1`, starting at `state`)
    /// under the *current* warm-start control sequence. Useful for debug overlays
    /// and for feeding the planned path back into the simulation as an intent.
    pub fn predicted_path(&self, state: PlanarState) -> Vec<[f64; 2]> {
        let n = self.cfg.horizon;
        let mut path = Vec::with_capacity(n + 1);
        let mut p = if Self::finite2(state.pos) {
            state.pos
        } else {
            [0.0, 0.0]
        };
        let mut v = if Self::finite2(state.vel) {
            state.vel
        } else {
            [0.0, 0.0]
        };
        path.push(p);
        for i in 0..n {
            let u = self.project_disk(self.warm[i]);
            for d in 0..2 {
                p[d] = p[d] + self.cfg.dt * v[d] + self.b_p * u[d];
                v[d] = v[d] + self.b_v * u[d];
            }
            path.push(p);
        }
        path
    }

    /// Reset the warm start to zero (e.g. after a discontinuity such as a
    /// turnover or teleport where the previous plan is no longer relevant).
    pub fn reset(&mut self) {
        for u in self.warm.iter_mut() {
            *u = [0.0, 0.0];
        }
    }

    pub fn config(&self) -> &PlanarMpcConfig {
        &self.cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_to_target(
        cfg: PlanarMpcConfig,
        start: PlanarState,
        reference: PlanarReference,
        steps: usize,
    ) -> (PlanarState, f64) {
        let mut mpc = PlanarPointMassMpc::new(cfg).unwrap();
        let mut state = start;
        let mut max_a = 0.0_f64;
        for _ in 0..steps {
            let a = mpc.control(state, &[reference]);
            let mag = (a[0] * a[0] + a[1] * a[1]).sqrt();
            max_a = max_a.max(mag);
            // Advance the same double-integrator plant the controller models.
            let (dt, b_p, b_v) = (cfg.dt, 0.5 * cfg.dt * cfg.dt, cfg.dt);
            for d in 0..2 {
                state.pos[d] = state.pos[d] + dt * state.vel[d] + b_p * a[d];
                state.vel[d] = state.vel[d] + b_v * a[d];
            }
        }
        (state, max_a)
    }

    #[test]
    fn drives_to_target_and_stops() {
        let cfg = PlanarMpcConfig::default();
        let start = PlanarState {
            pos: [0.0, 0.0],
            vel: [0.0, 0.0],
        };
        let target = PlanarReference::arrive([5.0, 3.0]);
        let (end, _) = run_to_target(cfg, start, target, 120);
        let dp = ((end.pos[0] - 5.0).powi(2) + (end.pos[1] - 3.0).powi(2)).sqrt();
        let speed = (end.vel[0].powi(2) + end.vel[1].powi(2)).sqrt();
        assert!(dp < 0.2, "did not arrive: dist={dp}, pos={:?}", end.pos);
        assert!(speed < 0.3, "did not stop: speed={speed}");
    }

    #[test]
    fn respects_acceleration_disk() {
        let cfg = PlanarMpcConfig {
            a_max: 2.5,
            ..PlanarMpcConfig::default()
        };
        let start = PlanarState {
            pos: [0.0, 0.0],
            vel: [0.0, 0.0],
        };
        // Far target so the controller wants to push hard.
        let target = PlanarReference::arrive([40.0, 40.0]);
        let (_, max_a) = run_to_target(cfg, start, target, 60);
        assert!(max_a <= 2.5 + 1e-6, "exceeded accel disk: {max_a}");
    }

    #[test]
    fn tracks_run_through_velocity() {
        // "Run into space downfield": a far target carrying a cruise velocity in
        // +x. Because the target stays far ahead over the whole run, the position
        // cost never asks the agent to brake, and the velocity term holds it at
        // cruise — so it should build up and sustain forward x-velocity rather
        // than stop.
        let cfg = PlanarMpcConfig::default();
        let start = PlanarState {
            pos: [0.0, 0.0],
            vel: [0.0, 0.0],
        };
        let target = PlanarReference::through([100.0, 0.0], [6.0, 0.0]);
        let (end, _) = run_to_target(cfg, start, target, 60);
        assert!(
            end.vel[0] > 3.0,
            "expected sustained forward velocity, got {:?}",
            end.vel
        );
    }

    #[test]
    fn follows_moving_reference_trajectory() {
        // Per-step reference trajectory: a point starting at [5,0] advancing in
        // +x at 6 u/s, with matching target velocity. This is the consistent way
        // to encode "pass through this point at speed" — the agent should be
        // moving forward (not parked) when the window ends.
        let cfg = PlanarMpcConfig::default();
        let mut mpc = PlanarPointMassMpc::new(cfg).unwrap();
        let mut state = PlanarState {
            pos: [0.0, 0.0],
            vel: [0.0, 0.0],
        };
        let (dt, b_p, b_v) = (cfg.dt, 0.5 * cfg.dt * cfg.dt, cfg.dt);
        for tick in 0..60 {
            // Build the horizon's reference trajectory advancing at cruise.
            let refs: Vec<PlanarReference> = (0..=cfg.horizon)
                .map(|h| {
                    let t = (tick + h) as f64 * dt;
                    PlanarReference::through([5.0 + 6.0 * t, 0.0], [6.0, 0.0])
                })
                .collect();
            let a = mpc.control(state, &refs);
            for d in 0..2 {
                state.pos[d] = state.pos[d] + dt * state.vel[d] + b_p * a[d];
                state.vel[d] = state.vel[d] + b_v * a[d];
            }
        }
        assert!(
            state.vel[0] > 3.0,
            "expected to be moving with the reference, got {:?}",
            state.vel
        );
    }

    #[test]
    fn warm_start_is_deterministic() {
        let cfg = PlanarMpcConfig::default();
        let state = PlanarState {
            pos: [1.0, -2.0],
            vel: [0.5, 0.0],
        };
        let target = PlanarReference::arrive([4.0, 4.0]);
        let mut a = PlanarPointMassMpc::new(cfg).unwrap();
        let mut b = PlanarPointMassMpc::new(cfg).unwrap();
        for _ in 0..10 {
            let ca = a.control(state, &[target]);
            let cb = b.control(state, &[target]);
            assert_eq!(ca, cb);
        }
    }

    #[test]
    fn empty_reference_coasts() {
        let cfg = PlanarMpcConfig::default();
        let mut mpc = PlanarPointMassMpc::new(cfg).unwrap();
        let state = PlanarState {
            pos: [0.0, 0.0],
            vel: [1.0, 0.0],
        };
        assert_eq!(mpc.control(state, &[]), [0.0, 0.0]);
    }

    fn min_clearance_to(
        cfg: PlanarMpcConfig,
        target: PlanarReference,
        obstacles: &[PlanarObstacle],
        probe: [f64; 2],
        steps: usize,
    ) -> ([f64; 2], f64) {
        let mut mpc = PlanarPointMassMpc::new(cfg).unwrap();
        let mut state = PlanarState {
            pos: [0.0, 0.0],
            vel: [0.0, 0.0],
        };
        let (dt, b_p, b_v) = (cfg.dt, 0.5 * cfg.dt * cfg.dt, cfg.dt);
        let mut min_clearance = f64::INFINITY;
        for _ in 0..steps {
            let a = mpc.control_with_obstacles(state, &[target], obstacles);
            for d in 0..2 {
                state.pos[d] = state.pos[d] + dt * state.vel[d] + b_p * a[d];
                state.vel[d] = state.vel[d] + b_v * a[d];
            }
            let clr =
                ((state.pos[0] - probe[0]).powi(2) + (state.pos[1] - probe[1]).powi(2)).sqrt();
            min_clearance = min_clearance.min(clr);
        }
        (state.pos, min_clearance)
    }

    #[test]
    fn obstacle_pushes_path_off_the_straight_line() {
        // Target straight ahead in +y; a keep-out just off the straight line (as a
        // real defender would be) makes the controller skirt it. We compare the
        // closest approach WITH the obstacle against the obstacle-free run: the
        // soft field must measurably increase the clearance while still arriving.
        let cfg = PlanarMpcConfig {
            a_max: 6.0,
            ..PlanarMpcConfig::default()
        };
        let target = PlanarReference::arrive([0.0, 20.0]);
        let probe = [0.6, 10.0];
        let obstacles = [PlanarObstacle::fixed(probe, 3.5, 120.0)];

        let (_, clearance_free) = min_clearance_to(cfg, target, &[], probe, 110);
        let (end, clearance_obs) = min_clearance_to(cfg, target, &obstacles, probe, 110);

        let dp = ((end[0]).powi(2) + (end[1] - 20.0).powi(2)).sqrt();
        assert!(
            dp < 1.5,
            "should still reach target, dist={dp}, end={end:?}"
        );
        assert!(
            clearance_obs > clearance_free + 0.5,
            "obstacle should increase clearance: free={clearance_free}, obs={clearance_obs}"
        );
    }

    #[test]
    fn timing_one_solve_is_well_under_budget() {
        // Measure the real cost of a warm-started, field-aware solve at the soccer
        // per-player settings: 45-step (3 s @ 15 Hz) horizon, 21 moving obstacles
        // with prediction decay, as the tick controller actually runs it. Prints
        // the per-solve time with --nocapture.
        let cfg = PlanarMpcConfig {
            horizon: 45,
            iters: 90,
            obstacle_decay_per_step: 0.5_f64.powf((1.0 / 15.0) / 1.5),
            ..PlanarMpcConfig::default()
        };
        let mut mpc = PlanarPointMassMpc::new(cfg).unwrap();
        let state = PlanarState {
            pos: [10.0, 10.0],
            vel: [1.0, 2.0],
        };
        let target = PlanarReference::arrive([60.0, 50.0]);
        let obstacles: Vec<PlanarObstacle> = (0..21)
            .map(|k| PlanarObstacle {
                center: [5.0 + (k as f64) * 3.0, 8.0 + (k % 5) as f64 * 6.0],
                velocity: [0.5, -0.3],
                radius: 2.0,
                weight: 40.0,
            })
            .collect();
        // Warm up.
        for _ in 0..5 {
            let _ = mpc.control_with_obstacles(state, &[target], &obstacles);
        }
        let iters = 2000;
        let start = std::time::Instant::now();
        let mut acc = 0.0;
        for _ in 0..iters {
            let a = mpc.control_with_obstacles(state, &[target], &obstacles);
            acc += a[0];
        }
        let per = start.elapsed().as_secs_f64() / iters as f64;
        println!(
            "MPC field-aware solve: {:.1} µs/solve (horizon {}, 21 obstacles); acc={acc:.3}",
            per * 1e6,
            cfg.horizon
        );
        // Generously bounded so it is not flaky, but proves we are far under the
        // ~1-2 ms/player a 10-20 ms whole-tick budget would allow.
        assert!(
            per < 2.0e-3,
            "single field-aware solve took {:.3} ms — too slow",
            per * 1e3
        );
    }

    #[test]
    fn obstacle_avoidance_is_inert_with_no_obstacles() {
        // control_with_obstacles(&[]) must match control() exactly.
        let cfg = PlanarMpcConfig::default();
        let state = PlanarState {
            pos: [1.0, 1.0],
            vel: [0.5, 0.0],
        };
        let target = PlanarReference::arrive([6.0, 4.0]);
        let mut a = PlanarPointMassMpc::new(cfg).unwrap();
        let mut b = PlanarPointMassMpc::new(cfg).unwrap();
        for _ in 0..10 {
            assert_eq!(
                a.control(state, &[target]),
                b.control_with_obstacles(state, &[target], &[])
            );
        }
    }

    #[test]
    fn non_finite_inputs_coast_without_poisoning_controller() {
        let cfg = PlanarMpcConfig::default();
        let mut mpc = PlanarPointMassMpc::new(cfg).unwrap();
        let good = PlanarState {
            pos: [1.0, 1.0],
            vel: [0.0, 0.0],
        };
        // Prime the warm start with a normal solve.
        let _ = mpc.control(good, &[PlanarReference::arrive([5.0, 5.0])]);

        // A NaN state coasts and does NOT corrupt the controller.
        let bad_state = PlanarState {
            pos: [f64::NAN, 0.0],
            vel: [0.0, 0.0],
        };
        let a = mpc.control(bad_state, &[PlanarReference::arrive([5.0, 5.0])]);
        assert_eq!(a, [0.0, 0.0]);

        // A NaN reference likewise coasts.
        let a2 = mpc.control(good, &[PlanarReference::arrive([f64::INFINITY, 0.0])]);
        assert_eq!(a2, [0.0, 0.0]);

        // The controller still produces a sane, finite, in-disk control afterward.
        let a3 = mpc.control(good, &[PlanarReference::arrive([5.0, 5.0])]);
        assert!(a3[0].is_finite() && a3[1].is_finite());
        assert!((a3[0] * a3[0] + a3[1] * a3[1]).sqrt() <= cfg.a_max + 1e-6);
    }

    #[test]
    fn rejects_bad_config() {
        assert!(PlanarPointMassMpc::new(PlanarMpcConfig {
            r: 0.0,
            ..PlanarMpcConfig::default()
        })
        .is_err());
        assert!(PlanarPointMassMpc::new(PlanarMpcConfig {
            a_max: -1.0,
            ..PlanarMpcConfig::default()
        })
        .is_err());
        assert!(PlanarPointMassMpc::new(PlanarMpcConfig {
            horizon: 0,
            ..PlanarMpcConfig::default()
        })
        .is_err());
    }

    #[test]
    fn predicted_path_has_horizon_plus_one_points() {
        let cfg = PlanarMpcConfig::default();
        let mut mpc = PlanarPointMassMpc::new(cfg).unwrap();
        let state = PlanarState {
            pos: [0.0, 0.0],
            vel: [0.0, 0.0],
        };
        let _ = mpc.control(state, &[PlanarReference::arrive([3.0, 3.0])]);
        let path = mpc.predicted_path(state);
        assert_eq!(path.len(), cfg.horizon + 1);
        assert_eq!(path[0], state.pos);
    }
}
