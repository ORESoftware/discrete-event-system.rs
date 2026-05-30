//! Port of `src/des/observability/validate-epidemic.ts`.
//!
//! Offline validator for the epidemic simulation. Reads the JSONL event stream
//! produced by the improved epidemic run and asserts a battery of invariants,
//! printing a pass/fail report. The TypeScript file is an entry script
//! (shebang + `main()` + `process.exit(code)`); per the migration rules the
//! logic lives in [`run`], which returns the would-be exit code (0 = all
//! invariants passed, 1 = at least one failed) instead of calling
//! `process::exit`. No `fn main` is added here.
//!
//! Invariants checked: I1 topology adherence (every observed transition is a
//! valid graph edge after flattening decision nodes); I2 per-entity continuity;
//! I3 branching probability (within a binomial 99% CI of the configured split);
//! I4 mass conservation; I5 per-cycle death rate; I6 tick monotonicity.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use crate::des::observability::logger::{read_events, JsonValue};

struct Failure {
    invariant: String,
    detail: String,
    context: Option<JsonValue>,
}

/// Inverse normal CDF for a 99% two-sided CI: z_{0.995} = 2.5758.
const Z_99: f64 = 2.5758;

fn binomial_ci99(p_hat: f64, n: u64) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let margin = Z_99 * (p_hat * (1.0 - p_hat) / n as f64).sqrt();
    ((0.0_f64).max(p_hat - margin), (1.0_f64).min(p_hat + margin))
}

fn fmt(n: f64, digits: usize) -> String {
    if n.is_finite() {
        format!("{n:.digits$}")
    } else {
        js_num(n)
    }
}

fn js_num(n: f64) -> String {
    if n.is_nan() {
        "NaN".to_string()
    } else if n == f64::INFINITY {
        "Infinity".to_string()
    } else if n == f64::NEG_INFINITY {
        "-Infinity".to_string()
    } else {
        format!("{n}")
    }
}

fn pass_fail(ok: bool) -> &'static str {
    if ok {
        "PASS"
    } else {
        "FAIL"
    }
}

fn event_kind(e: &JsonValue) -> &str {
    e.get("kind").and_then(|v| v.as_str()).unwrap_or("")
}

fn jstr<'a>(e: &'a JsonValue, key: &str) -> &'a str {
    e.get(key).and_then(|v| v.as_str()).unwrap_or("")
}

fn jnum(e: &JsonValue, key: &str) -> f64 {
    e.get(key).and_then(|v| v.as_f64()).unwrap_or(0.0)
}

fn is_decision(s: &str) -> bool {
    s.ends_with("Decision")
}

fn edge_pair(edge: &JsonValue) -> Option<(&str, &str)> {
    let arr = edge.as_array()?;
    if arr.len() < 2 {
        return None;
    }
    Some((arr[0].as_str()?, arr[1].as_str()?))
}

/// Default event-log path, the analog of the TS
/// `path.resolve(__dirname, '..','..','..','out','epidemic-events.jsonl')`
/// (repository-root `out/`).
pub fn default_event_log_path() -> std::path::PathBuf {
    std::path::Path::new("out").join("epidemic-events.jsonl")
}

/// Run the validator against `event_log_path`. Returns the exit code: 0 if all
/// invariants pass, 1 otherwise.
pub fn run(event_log_path: &str) -> i32 {
    let events = read_events(event_log_path).unwrap_or_else(|e| panic!("{e}"));

    let mut failures: Vec<Failure> = Vec::new();

    let start = events
        .iter()
        .find(|e| event_kind(e) == "sim_start")
        .unwrap_or_else(|| panic!("no sim_start event found"));
    let end = events
        .iter()
        .find(|e| event_kind(e) == "sim_end")
        .unwrap_or_else(|| panic!("no sim_end event found"));

    let transitions: Vec<&JsonValue> = events
        .iter()
        .filter(|e| event_kind(e) == "transition")
        .collect();
    let ticks: Vec<&JsonValue> = events.iter().filter(|e| event_kind(e) == "tick").collect();

    println!("================================================================");
    println!("epidemic event log validator");
    println!("  file:        {event_log_path}");
    println!(
        "  events:      {}  (transitions={}, ticks={})",
        events.len(),
        transitions.len(),
        ticks.len()
    );
    println!("  sim wall ms: {}", js_num(jnum(end, "elapsedMs")));
    println!("================================================================");
    println!();

    // ----- I1: topology adherence -------------------------------------------
    let empty: Vec<JsonValue> = Vec::new();
    let edges = start
        .pointer(&["config", "edges"])
        .and_then(|e| e.as_array())
        .unwrap_or(&empty);

    let mut decision_targets: HashMap<String, Vec<String>> = HashMap::new();
    for edge in edges {
        if let Some((a, b)) = edge_pair(edge) {
            if is_decision(a) {
                decision_targets
                    .entry(a.to_string())
                    .or_default()
                    .push(b.to_string());
            }
        }
    }

    let mut flat: HashSet<String> = HashSet::new();
    flat.insert("__source__->S".to_string());
    for edge in edges {
        let (a, b) = match edge_pair(edge) {
            Some(p) => p,
            None => continue,
        };
        if is_decision(a) {
            continue; // handled via the predecessor below
        }
        if a == "main-source" {
            continue; // already added as __source__->S
        }
        if is_decision(b) {
            if let Some(targets) = decision_targets.get(b) {
                for tgt in targets {
                    flat.insert(format!("{a}->{tgt}"));
                }
            }
        } else {
            flat.insert(format!("{a}->{b}"));
        }
    }

    let mut i1_bad = 0;
    for &t in &transitions {
        let from = jstr(t, "from");
        let to = jstr(t, "to");
        if !flat.contains(&format!("{from}->{to}")) {
            i1_bad += 1;
            if i1_bad <= 3 {
                let ctx = JsonValue::Object(vec![
                    (
                        "t".to_string(),
                        t.get("t").cloned().unwrap_or(JsonValue::Null),
                    ),
                    (
                        "entity".to_string(),
                        t.get("entity").cloned().unwrap_or(JsonValue::Null),
                    ),
                ]);
                failures.push(Failure {
                    invariant: "I1 topology".to_string(),
                    detail: format!("unexpected transition {from} -> {to}"),
                    context: Some(ctx),
                });
            }
        }
    }
    if i1_bad > 3 {
        failures.push(Failure {
            invariant: "I1 topology".to_string(),
            detail: format!("... and {} more invalid transitions", i1_bad - 3),
            context: None,
        });
    }
    println!(
        "I1 topology adherence:        {}  ({}/{} bad)",
        pass_fail(i1_bad == 0),
        i1_bad,
        transitions.len()
    );

    // ----- I2: per-entity continuity ----------------------------------------
    let mut last_seen: HashMap<String, String> = HashMap::new();
    let mut i2_bad = 0;
    for &t in &transitions {
        let entity = jstr(t, "entity").to_string();
        let from = jstr(t, "from").to_string();
        let to = jstr(t, "to").to_string();
        let expected = last_seen
            .get(&entity)
            .cloned()
            .unwrap_or_else(|| "__source__".to_string());
        if from != expected {
            i2_bad += 1;
            if i2_bad <= 3 {
                failures.push(Failure {
                    invariant: "I2 continuity".to_string(),
                    detail: format!(
                        "entity {entity} jumped from {expected} to {from} at t={}",
                        js_num(jnum(t, "t"))
                    ),
                    context: None,
                });
            }
        }
        last_seen.insert(entity, to);
    }
    println!(
        "I2 per-entity continuity:     {}  ({} jump(s))",
        pass_fail(i2_bad == 0),
        i2_bad
    );

    // ----- I3: branching probability ----------------------------------------
    let mut transitions_by_from: HashMap<String, HashMap<String, u64>> = HashMap::new();
    for &t in &transitions {
        let from = jstr(t, "from").to_string();
        let to = jstr(t, "to").to_string();
        *transitions_by_from
            .entry(from)
            .or_default()
            .entry(to)
            .or_insert(0) += 1;
    }

    let asym = start
        .pointer(&["config", "probabilities", "asymptomaticShare"])
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let hosp = start
        .pointer(&["config", "probabilities", "hospitalizationGivenSymptom"])
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let cfr = start
        .pointer(&["config", "probabilities", "caseFatalityGivenHospital"])
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    let expected_splits: Vec<(&str, Vec<(&str, f64)>)> = vec![
        ("I-P", vec![("I-A", asym), ("I-S", 1.0 - asym)]),
        ("I-S", vec![("R", 1.0 - hosp), ("I-H", hosp)]),
        ("I-H", vec![("R", 1.0 - cfr), ("D", cfr)]),
    ];

    println!("I3 branching probabilities:");
    for (from, exp) in &expected_splits {
        let row = transitions_by_from.get(*from);
        let total: u64 = match row {
            Some(r) => r.values().sum(),
            None => 0,
        };
        for (to, p_expected) in exp {
            let observed = row.and_then(|r| r.get(*to)).copied().unwrap_or(0);
            let p_hat = if total > 0 {
                observed as f64 / total as f64
            } else {
                0.0
            };
            let (lo, hi) = binomial_ci99(p_hat, total);
            let within = *p_expected >= lo && *p_expected <= hi;
            println!(
                "  {:<3} -> {:<3}  expected={}  observed={}  99%CI=[{}, {}]  n={}  {}",
                from,
                to,
                fmt(*p_expected, 3),
                fmt(p_hat, 3),
                fmt(lo, 3),
                fmt(hi, 3),
                total,
                pass_fail(within)
            );
            if !within {
                failures.push(Failure {
                    invariant: "I3 branching".to_string(),
                    detail: format!(
                        "{from} -> {to} expected {} not in 99% CI [{}, {}]",
                        fmt(*p_expected, 3),
                        fmt(lo, 3),
                        fmt(hi, 3)
                    ),
                    context: None,
                });
            }
        }
    }

    // ----- I4: mass conservation --------------------------------------------
    let source_out = transitions
        .iter()
        .filter(|&&t| jstr(t, "from") == "__source__")
        .count() as i64;
    let sink_in = transitions
        .iter()
        .filter(|&&t| jstr(t, "to") == "main-sink")
        .count() as i64;

    let created = end
        .pointer(&["totals", "created"])
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let absorbed = end
        .pointer(&["totals", "absorbed"])
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let total_alive: f64 = end
        .pointer(&["totals", "finalPopulations"])
        .and_then(|v| v.as_object())
        .map(|o| o.iter().map(|(_, v)| v.as_f64().unwrap_or(0.0)).sum())
        .unwrap_or(0.0);

    let i4_source_ok = source_out as f64 == created;
    let i4_sink_ok = sink_in as f64 == absorbed;
    let i4_mass_ok = created == absorbed + total_alive;

    println!("I4 mass conservation:");
    println!(
        "  source emissions in log: {}     createdCount: {}     {}",
        source_out,
        js_num(created),
        pass_fail(i4_source_ok)
    );
    println!(
        "  sink absorptions in log: {}        destroyedCount: {}  {}",
        sink_in,
        js_num(absorbed),
        pass_fail(i4_sink_ok)
    );
    println!(
        "  created == absorbed + alive: {} == {} + {}  {}",
        js_num(created),
        js_num(absorbed),
        js_num(total_alive),
        pass_fail(i4_mass_ok)
    );
    if !i4_source_ok {
        failures.push(Failure {
            invariant: "I4 mass".to_string(),
            detail: format!(
                "source emissions {} != createdCount {}",
                source_out,
                js_num(created)
            ),
            context: None,
        });
    }
    if !i4_sink_ok {
        failures.push(Failure {
            invariant: "I4 mass".to_string(),
            detail: format!(
                "sink absorptions {} != destroyedCount {}",
                sink_in,
                js_num(absorbed)
            ),
            context: None,
        });
    }
    if !i4_mass_ok {
        failures.push(Failure {
            invariant: "I4 mass".to_string(),
            detail: format!(
                "created {} != absorbed {} + alive {}",
                js_num(created),
                js_num(absorbed),
                js_num(total_alive)
            ),
            context: None,
        });
    }

    // ----- I5: per-cycle death rate -----------------------------------------
    let s_visits = transitions
        .iter()
        .filter(|&&t| jstr(t, "to") == "S")
        .count() as u64;
    let deaths = transitions
        .iter()
        .filter(|&&t| jstr(t, "to") == "D")
        .count() as u64;
    let q_theoretical = (1.0 - asym) * hosp * cfr;
    let q_observed = if s_visits > 0 {
        deaths as f64 / s_visits as f64
    } else {
        0.0
    };
    let (q_lo, q_hi) = binomial_ci99(q_observed, s_visits);
    let i5_ok = q_theoretical >= q_lo && q_theoretical <= q_hi;
    println!("I5 per-cycle death rate:");
    println!("  q_theoretical = {}", fmt(q_theoretical, 4));
    println!(
        "  q_observed    = {}  99%CI=[{}, {}]  S-visits={}  deaths={}  {}",
        fmt(q_observed, 4),
        fmt(q_lo, 4),
        fmt(q_hi, 4),
        s_visits,
        deaths,
        pass_fail(i5_ok)
    );
    if !i5_ok {
        failures.push(Failure {
            invariant: "I5 death rate".to_string(),
            detail: format!("theoretical {} not in 99% CI", fmt(q_theoretical, 4)),
            context: None,
        });
    }

    // ----- I6: tick monotonicity --------------------------------------------
    let mut i6_bad = 0;
    let mut prev_t = -1.0_f64;
    for &e in &ticks {
        let t = jnum(e, "t");
        if t <= prev_t {
            i6_bad += 1;
        }
        prev_t = t;
    }
    let t_first = ticks.first().map(|e| jnum(e, "t")).unwrap_or(0.0);
    let t_last = ticks.last().map(|e| jnum(e, "t")).unwrap_or(0.0);
    let mut oob = 0;
    for &t in &transitions {
        let tt = jnum(t, "t");
        if tt < t_first - 1.0 || tt > t_last + 1.0 {
            oob += 1;
        }
    }
    println!(
        "I6 tick monotonicity:         {}  ({} non-monotonic, {} out-of-band transitions)",
        pass_fail(i6_bad == 0),
        i6_bad,
        oob
    );
    if i6_bad > 0 {
        failures.push(Failure {
            invariant: "I6 tick monotonicity".to_string(),
            detail: format!("{i6_bad} non-monotonic ticks"),
            context: None,
        });
    }
    if oob > 0 {
        failures.push(Failure {
            invariant: "I6 tick monotonicity".to_string(),
            detail: format!("{oob} out-of-band transitions"),
            context: None,
        });
    }

    // ----- Summary -----------------------------------------------------------
    println!();
    println!("================================================================");
    if failures.is_empty() {
        println!("All invariants PASSED.");
    } else {
        println!("{} invariant failure(s):", failures.len());
        for f in &failures {
            println!("  - [{}] {}", f.invariant, f.detail);
        }
    }
    println!("================================================================");

    if failures.is_empty() {
        0
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("des_validate_{}_{}", std::process::id(), name));
        p
    }

    #[test]
    fn passes_on_a_trivially_consistent_log() {
        let path = temp_path("ok.jsonl");
        let p = path.to_str().unwrap();
        let log = concat!(
            r#"{"kind":"sim_start","config":{"edges":[["main-source","S"]],"#,
            r#""probabilities":{"asymptomaticShare":0.4,"hospitalizationGivenSymptom":0.2,"caseFatalityGivenHospital":0.12}}}"#,
            "\n",
            r#"{"kind":"tick","t":1,"populations":{}}"#,
            "\n",
            r#"{"kind":"tick","t":2,"populations":{}}"#,
            "\n",
            r#"{"kind":"sim_end","t":2,"elapsedMs":1,"totals":{"created":0,"absorbed":0,"finalPopulations":{}}}"#,
            "\n",
        );
        fs::write(p, log).unwrap();
        let code = run(p);
        assert_eq!(code, 0, "expected all invariants to pass");
        let _ = fs::remove_file(p);
    }

    #[test]
    fn flags_a_bad_transition() {
        let path = temp_path("bad.jsonl");
        let p = path.to_str().unwrap();
        let log = concat!(
            r#"{"kind":"sim_start","config":{"edges":[["main-source","S"]],"#,
            r#""probabilities":{"asymptomaticShare":0.4,"hospitalizationGivenSymptom":0.2,"caseFatalityGivenHospital":0.12}}}"#,
            "\n",
            r#"{"kind":"tick","t":1,"populations":{}}"#,
            "\n",
            r#"{"kind":"transition","t":1,"entity":"f0","from":"X","to":"Y"}"#,
            "\n",
            r#"{"kind":"sim_end","t":1,"elapsedMs":1,"totals":{"created":0,"absorbed":0,"finalPopulations":{}}}"#,
            "\n",
        );
        fs::write(p, log).unwrap();
        let code = run(p);
        assert_eq!(code, 1, "expected an invariant failure");
        let _ = fs::remove_file(p);
    }
}
