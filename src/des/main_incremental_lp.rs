//! Port of `src/des/main-incremental-lp.ts`.
//!
//! Incremental LP as a DES: each model edit is a movable arriving at the
//! tableau station, each pivot is a tick. A 2-D production-planning LP evolves
//! over time (add/remove constraints, change objective, add/remove variables).
//!
//! Delegates to `crate::des::general::incremental_lp`.
//!
//! PORT NOTE: `ANIMATE=1` rendering (FrameRecorder + incremental-lp scene) and
//! the per-tick recording arrays / `StandardFormShadow` that exist ONLY to feed
//! that animation are omitted. The console pivot trace (the substance of the
//! script) is reproduced faithfully.

#![allow(dead_code)]

use crate::des::general::incremental_lp::{
    IncrementalLP, IncrementalLPInit, LPEvent, PivotMode, Sense, SolverStatus,
};

fn names(xs: &[&str]) -> Option<Vec<String>> {
    Some(xs.iter().map(|s| s.to_string()).collect())
}

fn status_str(s: SolverStatus) -> &'static str {
    match s {
        SolverStatus::Primal => "primal",
        SolverStatus::Dual => "dual",
        SolverStatus::Optimal => "optimal",
        SolverStatus::Infeasible => "infeasible",
        SolverStatus::Unbounded => "unbounded",
    }
}

/// One scheduled modification (the TS `ScenarioStep`).
struct ScenarioStep {
    tick: usize,
    event: LPEvent,
    description: &'static str,
}

/// Default 2-D LP scenario exercising all 5 modification types.
fn build_default_scenario() -> (IncrementalLPInit, Vec<ScenarioStep>) {
    let init = IncrementalLPInit {
        sense: Sense::Max,
        c: vec![3.0, 5.0],
        a: vec![vec![2.0, 1.0], vec![1.0, 3.0]],
        b: vec![100.0, 90.0],
        var_names: names(&["widget", "gadget"]),
        con_names: names(&["labor", "material"]),
    };
    let steps = vec![
        ScenarioStep {
            tick: 4,
            event: LPEvent::AddConstraint { tick: 4.0, coefs: vec![1.0, 0.0], rhs: 30.0, name: Some("cap_widget".into()) },
            description: "add x_widget ≤ 30",
        },
        ScenarioStep {
            tick: 8,
            event: LPEvent::ChangeObjective { tick: 8.0, new_c: vec![5.0, 3.0], name: None },
            description: "change c → (5, 3)",
        },
        ScenarioStep {
            tick: 12,
            event: LPEvent::RemoveConstraint { tick: 12.0, index: 0, name: None },
            description: "remove labor constraint",
        },
        ScenarioStep {
            tick: 16,
            event: LPEvent::AddVariable { tick: 16.0, column: vec![1.0, 1.0], c_new: 7.0, name: Some("thingamajig".into()) },
            description: "add new product x_thingamajig (c=7, col [1,1])",
        },
        ScenarioStep {
            tick: 22,
            event: LPEvent::RemoveConstraint { tick: 22.0, index: 0, name: None },
            description: "remove material constraint (LP becomes unbounded)",
        },
        ScenarioStep {
            tick: 26,
            event: LPEvent::AddConstraint { tick: 26.0, coefs: vec![1.0, 1.0, 1.0], rhs: 50.0, name: Some("budget".into()) },
            description: "add budget: w+g+t ≤ 50  (re-bounds the LP)",
        },
        ScenarioStep {
            tick: 32,
            event: LPEvent::ChangeObjective { tick: 32.0, new_c: vec![1.0, 1.0, 8.0], name: None },
            description: "change c → (1, 1, 8) — favour thingamajig",
        },
        ScenarioStep {
            tick: 36,
            event: LPEvent::RemoveVariable { tick: 36.0, struct_index: 1, name: None },
            description: "remove gadget (line discontinued)",
        },
    ];
    (init, steps)
}

/// Entry point (TS top-level `main`).
pub fn run() {
    let (init, steps) = build_default_scenario();
    let mut inc = IncrementalLP::new(init);

    println!("# Incremental LP solver as DES — adaptive to add/remove/change events");
    println!("# Each pivot = one tick. Each modification = one movable arriving at the tableau.");
    println!("# Initial: max 3·widget + 5·gadget   s.t.  2w+g ≤ 100,  w+3g ≤ 90");
    println!();

    let total_ticks = steps.iter().map(|s| s.tick).max().unwrap_or(0) + 8;

    for tick in 1..=total_ticks {
        // 1. Apply any events scheduled for THIS tick.
        let mut applied_desc: Option<&str> = None;
        for s in &steps {
            if s.tick == tick {
                inc.apply_event(s.event.clone());
                applied_desc = Some(s.description);
            }
        }
        // 2. One pivot per tick.
        let ev = inc.step();
        let pivot_label: Option<String> = match ev.mode {
            PivotMode::Primal | PivotMode::Dual => Some(format!(
                "{}: {} enters, {} leaves",
                if ev.mode == PivotMode::Primal { "primal" } else { "dual" },
                ev.entering_name.as_deref().unwrap_or("?"),
                ev.leaving_name.as_deref().unwrap_or("?")
            )),
            PivotMode::Optimal => Some("optimal".to_string()),
            PivotMode::Idle => None,
            PivotMode::Infeasible => Some("infeasible".to_string()),
            PivotMode::Unbounded => Some("unbounded".to_string()),
        };
        // 3. Console trace.
        let event_str = applied_desc.map(|d| format!("[{d}]  ")).unwrap_or_default();
        let pivot_str = pivot_label.unwrap_or_default();
        println!(
            "tick {:>2}  z={:>8}  x=[{}]  {}{}",
            tick,
            format!("{:.3}", inc.get_z()),
            inc.get_x().iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>().join(", "),
            event_str,
            pivot_str
        );
    }
    println!();
    println!(
        "# Final: z = {:.4}, x = [{}], status = {}",
        inc.get_z(),
        inc.get_x().iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(", "),
        status_str(inc.status)
    );

    if std::env::var("ANIMATE").as_deref() == Ok("1") {
        println!("# (animation omitted in Rust port — see PORT NOTE)");
    }
}
