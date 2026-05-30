//! `des::r#abstract` — root of the queueing-network entity model (port of
//! `src/des/abstract/`). The directory/file names stay literally `abstract`
//! (matching the TS tree); the Rust module is reached via the raw identifier
//! `r#abstract` because `abstract` is a reserved keyword.

#[allow(clippy::module_inception)]
pub mod r#abstract;
pub mod composers;
pub mod interfaces;
pub mod test;
