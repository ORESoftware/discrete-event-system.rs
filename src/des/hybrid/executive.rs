//! The hybrid executive: one global clock that integrates continuous state,
//! lands exactly on discrete sample hits (multirate), and bisects to locate
//! zero-crossings between steps.
//!
//! ## Step rule
//!
//! At each iteration the executive first services any discrete blocks *due now*
//! (ZOH causality: outputs reflect the held state, then `update` computes the
//! next held value). It then takes a continuous step capped to the next sample
//! hit, `t_end`, or `max_step` — whichever is closest. If a zero-crossing is
//! detected across that step, it bisects to the crossing, advances there, fires
//! the event, and resumes. Pure-discrete and pure-continuous diagrams are the
//! degenerate cases (no continuous states ⇒ time jumps hit-to-hit).

use serde_json::{json, Value};

use super::diagram::Compiled;

/// Simulation controls.
#[derive(Clone, Copy, Debug)]
pub struct SimOptions {
    pub t_end: f64,
    /// Largest continuous step (the solver may take smaller steps to land on
    /// sample hits or zero-crossings).
    pub max_step: f64,
    /// Time tolerance for zero-crossing bisection.
    pub zc_tol: f64,
}

impl Default for SimOptions {
    fn default() -> Self {
        SimOptions { t_end: 10.0, max_step: 0.01, zc_tol: 1e-9 }
    }
}

impl SimOptions {
    pub fn new(t_end: f64, max_step: f64) -> Self {
        SimOptions { t_end, max_step, zc_tol: 1e-9 }
    }
}

/// Recorded port signals over time. One column per scalar output channel; one
/// row per recorded time point.
#[derive(Clone, Debug, Default)]
pub struct Trace {
    pub columns: Vec<String>,
    pub times: Vec<f64>,
    pub rows: Vec<Vec<f64>>,
    /// Number of recorded events (zero-crossings handled).
    pub events: usize,
}

impl Trace {
    fn new(compiled: &Compiled) -> Self {
        let mut columns = Vec::new();
        for (b, widths) in compiled.output_widths().iter().enumerate() {
            let name = &compiled.names()[b];
            for (port, &w) in widths.iter().enumerate() {
                if w == 1 {
                    columns.push(format!("{name}.p{port}"));
                } else {
                    for i in 0..w {
                        columns.push(format!("{name}.p{port}[{i}]"));
                    }
                }
            }
        }
        Trace { columns, times: Vec::new(), rows: Vec::new(), events: 0 }
    }

    fn record(&mut self, t: f64, outs: &[Vec<Vec<f64>>]) {
        let mut row = Vec::with_capacity(self.columns.len());
        for ports in outs {
            for sig in ports {
                row.extend_from_slice(sig);
            }
        }
        // Overwrite the last sample if we are at the same instant (e.g. a sample
        // hit handled right after a continuous landing) to avoid duplicate rows.
        if let Some(&last) = self.times.last() {
            if (last - t).abs() < 1e-12 {
                *self.times.last_mut().unwrap() = t;
                *self.rows.last_mut().unwrap() = row;
                return;
            }
        }
        self.times.push(t);
        self.rows.push(row);
    }

    /// Index of a named column (e.g. `"plant.p0"`), if present.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c == name)
    }

    /// `(times, values)` for a named column.
    pub fn series(&self, name: &str) -> Option<(Vec<f64>, Vec<f64>)> {
        let idx = self.column_index(name)?;
        Some((self.times.clone(), self.rows.iter().map(|r| r[idx]).collect()))
    }

    /// CSV with a leading `t` column.
    pub fn to_csv(&self) -> String {
        let mut s = String::from("t");
        for c in &self.columns {
            s.push(',');
            s.push_str(c);
        }
        s.push('\n');
        for (k, row) in self.rows.iter().enumerate() {
            s.push_str(&format!("{}", self.times[k]));
            for v in row {
                s.push(',');
                s.push_str(&format!("{v}"));
            }
            s.push('\n');
        }
        s
    }

    /// One JSON object per time point (`{ "t": …, "col": value, … }`) — the JSONL
    /// frame stream the plugin sim-player renders directly.
    pub fn to_jsonl_frames(&self) -> Vec<Value> {
        self.times
            .iter()
            .enumerate()
            .map(|(k, &t)| {
                let mut obj = serde_json::Map::new();
                obj.insert("t".to_string(), json!(t));
                for (c, col) in self.columns.iter().enumerate() {
                    obj.insert(col.clone(), json!(self.rows[k][c]));
                }
                Value::Object(obj)
            })
            .collect()
    }
}

/// Simulate a compiled diagram, returning the recorded trace.
pub fn simulate(compiled: &Compiled, opts: &SimOptions) -> Trace {
    let mut xc = compiled.init_cont_global();
    let mut xd = compiled.init_disc_all();
    let mut trace = Trace::new(compiled);

    // Per-discrete-block hit counter (k); next hit = offset + k*period. Integer
    // counters avoid float drift over long runs.
    let mut hit_k = vec![0u64; compiled.disc.len()];
    let next_hit = |i: usize, k: u64| -> f64 {
        let (_, period, offset) = compiled.disc[i];
        offset + k as f64 * period
    };

    let mut t = 0.0;
    // Initial record (held discrete state, initial continuous state).
    {
        let outs = compiled.propagate(t, &xc, &xd);
        trace.record(t, &outs);
    }

    let eps = 1e-9;
    let mut guard = 0usize;
    let guard_max = 5_000_000usize;

    while t < opts.t_end - eps {
        guard += 1;
        if guard > guard_max {
            break;
        }

        // 1) Service discrete blocks due at the current instant.
        let mut due = Vec::new();
        for i in 0..compiled.disc.len() {
            if (next_hit(i, hit_k[i]) - t).abs() < 1e-9 {
                due.push(i);
            }
        }
        if !due.is_empty() {
            let outs = compiled.propagate(t, &xc, &xd);
            for &i in &due {
                let (b, _, _) = compiled.disc[i];
                compiled.discrete_update(b, t, &mut xd, &outs);
                hit_k[i] += 1;
            }
            let outs2 = compiled.propagate(t, &xc, &xd);
            trace.record(t, &outs2);
            continue;
        }

        // 2) Determine the next stopping time: min(next sample hit, t_end), then
        // cap the step by max_step.
        let mut next_event_time = opts.t_end;
        for i in 0..compiled.disc.len() {
            let nh = next_hit(i, hit_k[i]);
            if nh > t + eps && nh < next_event_time {
                next_event_time = nh;
            }
        }
        let mut h = opts.max_step.min(opts.t_end - t);
        if next_event_time - t < h {
            h = next_event_time - t;
        }
        if h <= eps {
            // No continuous progress possible; jump straight to the next hit.
            if next_event_time > t + eps {
                t = next_event_time;
            } else {
                break;
            }
            continue;
        }

        // 3) Continuous step (with zero-crossing handling).
        if compiled.n_cont_total > 0 {
            let x_trial = compiled.rk4_step(t, &xc, &xd, h);
            if let Some((gi, off)) = compiled.find_crossing(t, &xc, &x_trial, &xd, h, opts.zc_tol) {
                let x_at = compiled.rk4_step(t, &xc, &xd, off);
                let tc = t + off;
                xc = x_at;
                compiled.fire_event(gi, tc, &mut xc, &mut xd);
                t = tc;
                trace.events += 1;
                let outs = compiled.propagate(t, &xc, &xd);
                trace.record(t, &outs);
                continue;
            }
            xc = x_trial;
        }
        t += h;
        let outs = compiled.propagate(t, &xc, &xd);
        trace.record(t, &outs);
    }

    trace
}
