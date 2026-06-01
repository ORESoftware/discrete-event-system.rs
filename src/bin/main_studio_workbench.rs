//! Generate the self-contained Studio model workbench.

use des_engine::des::studio::{starter_model_spec, write_studio_player_html, write_workbench_html};

fn main() {
    let spec = starter_model_spec();
    let path = "out/studio/workbench.html";
    match write_workbench_html(path, &spec) {
        Ok(()) => println!("studio workbench: {path}"),
        Err(err) => {
            eprintln!("studio workbench generation failed: {err}");
            std::process::exit(1);
        }
    }

    match write_studio_player_html("out", &spec) {
        Ok(paths) => {
            for path in paths {
                println!("studio player: {}", path.display());
            }
        }
        Err(err) => {
            eprintln!("studio player generation failed: {err}");
            std::process::exit(1);
        }
    }
}
