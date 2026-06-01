//! Generate the self-contained Modeling Studio workbench.

use des_engine::des::studio::{starter_model_spec, write_workbench_html};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "out/studio/workbench.html";
    write_workbench_html(path, &starter_model_spec())?;
    println!("{path}");
    Ok(())
}
