//! Entry point for the animated voting-algorithm lab.

use crate::des::voting_algorithms::write_voting_lab_html;

pub fn run() {
    match write_voting_lab_html("out") {
        Ok(path) => {
            println!("Voting algorithm lab: {}", path.display());
            if let Ok(abs) = std::fs::canonicalize(&path) {
                println!("Open in browser: file://{}", abs.display());
            }
        }
        Err(err) => panic!("write voting algorithm lab: {err}"),
    }
}
