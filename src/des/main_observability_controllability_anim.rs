//! Port of `src/des/main-observability-controllability-anim.ts`.
//!
//! Generates an HTML slideshow walking through the controllability /
//! observability tests for LTI, MDP, and POMDP systems.
//!
//! Conversion notes:
//!   - `class ObsCtrlAnimator` → struct + impl; async `run()` → [`run`].
//!
//! PORT NOTE: this entry point is a pure renderer — all of its content comes
//! from `animation/scenes/obs-ctrl-scene` (`ObsCtrlScene`, `OC_STAGE_W`,
//! `OC_STAGE_H`), which is NOT yet ported (`animation::scenes` has no
//! `obs_ctrl_scene.rs`). There is no simulation to run faithfully here, so the
//! render is stubbed with a note. Wire `ObsCtrlScene` +
//! `crate::des::animation::frame_recorder::FrameRecorder` once the scene exists.
//! The structural evaluator itself is fully ported in
//! `crate::des::main_observability_controllability`.

/// Entry point (TS top-level script).
pub fn run() {
    let out = std::path::Path::new("out").join("obs-ctrl").join("animation.html");
    println!(
        "Obs/Ctrl animation: omitted in Rust port — the storyboard scene \
         (animation::scenes::obs_ctrl_scene) is not yet ported; would write {} (see PORT NOTE). \
         Run `main_observability_controllability` for the evaluator itself.",
        out.display()
    );
}
