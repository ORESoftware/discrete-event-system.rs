//! Generate the standalone Modeling Studio HTML page.

pub fn run() {
    match crate::des::studio::write_studio_editor_html("out") {
        Ok(path) => println!("{}", path.display()),
        Err(err) => {
            eprintln!("failed to generate Modeling Studio: {err}");
            std::process::exit(1);
        }
    }
}
