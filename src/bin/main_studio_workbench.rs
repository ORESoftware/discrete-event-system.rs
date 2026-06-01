//! Generate the self-contained Studio workbench.
//!
//! Run with: `cargo run --bin main_studio_workbench`
//! Output: `out/studio/workbench.html`

fn main() {
    match des_engine::des::studio::write_workbench("out/studio/workbench.html") {
        Ok(path) => println!("studio workbench: {}", path.display()),
        Err(err) => {
            eprintln!("studio workbench generation failed: {err}");
            std::process::exit(1);
        }
    }
}
