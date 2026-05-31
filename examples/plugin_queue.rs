//! Example **external plugin program** (Rust): an M/M/1-style queue that streams
//! one JSON object per line (JSONL) — one frame per tick. Each frame carries
//! `shapes` (the animation schema the sim player understands) plus numeric
//! fields (`n`, `serverBusy`) that the player charts on a timeline.
//!
//! The host (`des_engine::des::plugin`) neither imports nor links this — it just
//! spawns the compiled binary and reads stdout. Build + render via:
//!
//! ```bash
//! cargo build --example plugin_queue --example plugin_lp
//! cargo run   --example render_demo
//! ```

fn main() {
    // Tiny deterministic LCG so the demo is reproducible (no rand dependency).
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 32) as u32) as f64 / u32::MAX as f64
    };

    let lambda = 0.55; // arrival probability per tick
    let mu = 0.70; // service-completion probability per tick when busy
    let steps = 160;
    let mut n: i64 = 0;

    for tick in 0..steps {
        if next() < lambda {
            n += 1;
        }
        if n > 0 && next() < mu {
            n -= 1;
        }
        let busy = if n > 0 { 1 } else { 0 };

        // Build the SVG shapes for this frame.
        let mut shapes = String::new();
        shapes.push_str(&format!(
            r##"{{"kind":"rect","x":40,"y":150,"w":80,"h":80,"rx":10,"fill":"{}","stroke":"#0f172a","strokeWidth":2,"label":"server"}}"##,
            if busy == 1 { "#16a34a" } else { "#cbd5e1" }
        ));
        let shown = n.min(12);
        for i in 0..shown {
            let x = 170 + i * 36;
            shapes.push_str(&format!(
                r##",{{"kind":"circle","x":{x},"y":190,"r":14,"fill":"#2563eb","stroke":"#1e3a8a","strokeWidth":1.5}}"##
            ));
        }
        if n > 12 {
            shapes.push_str(&format!(
                r##",{{"kind":"text","x":620,"y":196,"text":"+{} more","fontSize":13,"fill":"#475569"}}"##,
                n - 12
            ));
        }

        let caption = format!(
            "tick {tick} \u{2014} {n} in system, server {}",
            if busy == 1 { "busy" } else { "idle" }
        );
        println!(
            r#"{{"t":{:.1},"tick":{tick},"n":{n},"serverBusy":{busy},"shapes":[{shapes}],"caption":"{caption}"}}"#,
            tick as f64
        );
    }
}
