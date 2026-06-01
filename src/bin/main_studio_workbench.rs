//! Generate the self-contained Studio model workbench.

use des_engine::des::studio::{starter_model_spec, write_workbench_html};

fn main() {
    let path = "out/studio/workbench.html";
    match write_workbench_html(path, &starter_model_spec()) {
        Ok(()) => println!("studio workbench: {path}"),
        Err(err) => {
            eprintln!("studio workbench generation failed: {err}");
            std::process::exit(1);
        }
    }
}
