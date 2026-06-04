//! Report TypeScript-to-Rust migration coverage.
//!
//! This is the Rust counterpart to `scripts/migration_status.py`: it walks the
//! TypeScript source tree, maps every `.ts` file to its expected Rust path
//! (`-` to `_`, `index.ts` to `mod.rs`), and reports matched/missing coverage.

use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fs;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputMode {
    Summary,
    MissingGroups,
    ListMissing,
}

#[derive(Debug)]
struct Args {
    ts_root: PathBuf,
    rs_root: PathBuf,
    mode: OutputMode,
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, message.into())
}

fn home_dir() -> PathBuf {
    env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn default_ts_root() -> PathBuf {
    home_dir().join("codes/ores/des-engine/src")
}

fn default_rs_root() -> PathBuf {
    home_dir().join("codes/ores/discrete-event-system.rs/src")
}

fn next_arg<I>(args: &mut I, flag: &str) -> Result<PathBuf, Box<dyn Error>>
where
    I: Iterator<Item = OsString>,
{
    let value = args
        .next()
        .ok_or_else(|| invalid_input(format!("{flag} requires a value")))?;
    Ok(PathBuf::from(value))
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut ts_root = env::var_os("MIGRATION_TS_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(default_ts_root);
    let mut rs_root = env::var_os("MIGRATION_RS_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(default_rs_root);
    let mut mode = OutputMode::Summary;

    let mut args = env::args_os().skip(1);
    while let Some(arg) = args.next() {
        let flag = arg
            .into_string()
            .map_err(|_| invalid_input("argument flag must be valid UTF-8"))?;
        match flag.as_str() {
            "--missing" => mode = OutputMode::MissingGroups,
            "--list-missing" => mode = OutputMode::ListMissing,
            "--ts-root" => ts_root = next_arg(&mut args, "--ts-root")?,
            "--rs-root" => rs_root = next_arg(&mut args, "--rs-root")?,
            "--help" | "-h" => {
                println!(
                    "usage: migration_status [--missing|--list-missing] [--ts-root PATH] [--rs-root PATH]"
                );
                std::process::exit(0);
            }
            _ => return Err(invalid_input(format!("unknown argument {flag}")).into()),
        }
    }

    Ok(Args {
        ts_root,
        rs_root,
        mode,
    })
}

fn collect_ts_files(root: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    if !root.exists() {
        return Err(invalid_input(format!(
            "TypeScript root does not exist: {}",
            root.display()
        ))
        .into());
    }
    if !root.is_dir() {
        return Err(invalid_input(format!(
            "TypeScript root is not a directory: {}",
            root.display()
        ))
        .into());
    }
    let mut out = Vec::new();
    collect_ts_files_inner(root, root, &mut out)?;
    out.sort();
    Ok(out)
}

fn collect_ts_files_inner(
    root: &Path,
    dir: &Path,
    out: &mut Vec<PathBuf>,
) -> Result<(), Box<dyn Error>> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_ts_files_inner(root, &path, out)?;
            continue;
        }
        if !file_type.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if name.ends_with(".ts") && !name.ends_with(".d.ts") {
            out.push(path.strip_prefix(root)?.to_path_buf());
        }
    }
    Ok(())
}

fn path_parts(path: &Path) -> Vec<String> {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect()
}

fn display_rel(path: &Path) -> String {
    path_parts(path).join("/")
}

fn expected_rs(rel: &Path) -> PathBuf {
    let parts = path_parts(rel);
    let Some((file_name, dirs)) = parts.split_last() else {
        return PathBuf::new();
    };
    let mut out = PathBuf::new();
    for dir in dirs {
        out.push(dir.replace('-', "_"));
    }
    if file_name == "index.ts" {
        out.push("mod.rs");
    } else {
        let stem = file_name
            .strip_suffix(".ts")
            .unwrap_or(file_name)
            .replace('-', "_");
        out.push(format!("{stem}.rs"));
    }
    out
}

fn missing_group_key(rel: &Path) -> String {
    let parts = path_parts(rel);
    if parts.len() > 3 {
        parts[..3].join("/")
    } else if parts.len() > 1 {
        parts[..parts.len() - 1].join("/")
    } else {
        String::new()
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let ts_files = collect_ts_files(&args.ts_root)?;
    let mut matched = 0usize;
    let mut missing = Vec::<(PathBuf, PathBuf)>::new();

    for rel in &ts_files {
        let rs_rel = expected_rs(rel);
        if args.rs_root.join(&rs_rel).exists() {
            matched += 1;
        } else {
            missing.push((rel.clone(), rs_rel));
        }
    }

    let total = ts_files.len();
    let coverage = if total == 0 { 0 } else { matched * 100 / total };
    println!("TOTAL TS: {total}");
    println!("MATCHED:  {matched}");
    println!("MISSING:  {}", missing.len());
    println!("COVERAGE: {coverage}%  ({matched}/{total})");

    match args.mode {
        OutputMode::Summary => {}
        OutputMode::MissingGroups => {
            let mut groups = BTreeMap::<String, usize>::new();
            for (rel, _) in &missing {
                *groups.entry(missing_group_key(rel)).or_default() += 1;
            }
            let mut grouped = groups.into_iter().collect::<Vec<_>>();
            grouped.sort_by(|(left_key, left_count), (right_key, right_count)| {
                right_count
                    .cmp(left_count)
                    .then_with(|| left_key.cmp(right_key))
            });
            for (key, count) in grouped {
                println!("  [{count:3}] {key}/");
            }
        }
        OutputMode::ListMissing => {
            for (rel, rs_rel) in &missing {
                println!("{}  ->  {}", display_rel(rel), display_rel(rs_rel));
            }
        }
    }

    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("migration_status: {err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_rs_maps_index_and_hyphens() {
        assert_eq!(
            display_rel(&expected_rs(Path::new("des/base/index.ts"))),
            "des/base/mod.rs"
        );
        assert_eq!(
            display_rel(&expected_rs(Path::new("des/some-dir/my-file.ts"))),
            "des/some_dir/my_file.rs"
        );
    }

    #[test]
    fn missing_group_matches_python_script_shape() {
        assert_eq!(
            missing_group_key(Path::new("des/general/foo/bar.ts")),
            "des/general/foo"
        );
        assert_eq!(missing_group_key(Path::new("des/main.ts")), "des");
    }
}
