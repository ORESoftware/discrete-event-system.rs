//! `des::signals` — signal-flow entities (port of `src/des/signals/`).
//! (`abstract` is a reserved keyword, so the signal base module is reached via
//! the raw identifier `r#abstract`.)

pub mod r#abstract;
pub mod adder;
pub mod differential;
pub mod incrementer;
pub mod integral;
pub mod multi_directional_signal_entity;
pub mod mux;
pub mod signal_value;
pub mod single_direction_signal_entity;
