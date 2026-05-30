//! Port of `src/des/animation/render.ts`.
//!
//! Post-hoc renderer: read a `.frames.jsonl` file produced by [`FrameRecorder`]
//! and emit a self-contained HTML animation, without re-running the simulation.
//!
//! ## Conversion notes
//!
//! * The `#!/usr/bin/env ts-node` shebang and `require.main === module` guard
//!   collapse to a plain entry point. Because this crate is built as a library,
//!   the `main()` body becomes [`run`] (call it from a `src/bin/*.rs` shim or a
//!   subcommand dispatcher).
//! * `process.argv.slice(2)` → `std::env::args().skip(1)`; `process.exit(n)` →
//!   [`std::process::exit`].
//! * `fs.existsSync` / `fs.writeFileSync` → [`Path::exists`] / [`fs::write`].
//! * `inputPath.replace(/\.frames\.jsonl$/, '.html').replace(/\.jsonl$/, '.html')`
//!   → the two sequential suffix swaps in [`default_output`].
//!
//! [`FrameRecorder`]: crate::des::animation::frame_recorder::FrameRecorder

#![allow(dead_code)]

use std::fs;
use std::path::Path;

use crate::des::animation::frame_recorder::read_animation;
use crate::des::animation::html_player::build_html;
use crate::des::animation::types::js_num;

/// CLI entry point (the TS `main()`): `render <input.frames.jsonl> [output.html]`.
pub fn run() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    run_with_args(&args);
}

/// Testable core: mirrors `main()` but takes the argument vector explicitly.
pub fn run_with_args(args: &[String]) {
    if args.is_empty() || args.len() > 2 {
        eprintln!("usage: render <input.frames.jsonl> [output.html]");
        std::process::exit(2);
    }
    let input_path = &args[0];
    let output_path = match args.get(1) {
        Some(o) => o.clone(),
        None => default_output(input_path),
    };

    if !Path::new(input_path).exists() {
        eprintln!("render: input not found: {input_path}");
        std::process::exit(1);
    }
    let anim = match read_animation(input_path) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    if let Err(e) = fs::write(&output_path, build_html(&anim)) {
        eprintln!("render: cannot write {output_path}: {e}");
        std::process::exit(1);
    }
    println!(
        "render: {input_path} \u{2192} {output_path}  ({} frames, {}\u{00d7}{})",
        anim.frames.len(),
        js_num(anim.width),
        js_num(anim.height)
    );
}

/// Default output path: swap a trailing `.frames.jsonl` for `.html`, otherwise
/// a trailing `.jsonl` for `.html` (the two `$`-anchored `String.replace`s).
fn default_output(input: &str) -> String {
    let s = match input.strip_suffix(".frames.jsonl") {
        Some(stem) => format!("{stem}.html"),
        None => input.to_string(),
    };
    match s.strip_suffix(".jsonl") {
        Some(stem) => format!("{stem}.html"),
        None => s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_output_suffix_swaps() {
        assert_eq!(
            default_output("out/two-disease.frames.jsonl"),
            "out/two-disease.html"
        );
        assert_eq!(default_output("out/raw.jsonl"), "out/raw.html");
        assert_eq!(default_output("out/keep.txt"), "out/keep.txt");
    }
}
