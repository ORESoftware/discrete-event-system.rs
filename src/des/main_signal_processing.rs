//! Port of `src/des/main-signal-processing.ts`.
//!
//! Small CLI demo of Z / Laplace / Fourier transform models. For JSON-driven
//! runs use the from-json entry point.
//!
//! Conversion notes:
//!   - top-level `main()` → [`run`].
//!   - complex outputs use `general::signal_transforms::ComplexValue`.
//!   - `Number.prototype.toPrecision(6)` reproduced by [`to_precision`].

use std::collections::HashMap;

use crate::des::general::signal_transforms::{
    format_complex, run_fourier_transform, run_laplace_transform, run_z_transform,
    ComplexPointInput, FourierTransformParams, LaplaceTransformParams, TransformRunResult,
    ZTransformParams,
};

/// JS `Number.prototype.toPrecision(p)` — `p` significant digits.
fn to_precision(x: f64, p: usize) -> String {
    if x == 0.0 {
        return format!("{:.*}", p.saturating_sub(1), 0.0);
    }
    let exp = x.abs().log10().floor() as i32;
    if exp < -6 || exp >= p as i32 {
        format!("{:.*e}", p.saturating_sub(1), x)
    } else {
        let decimals = (p as i32 - 1 - exp).max(0) as usize;
        format!("{:.*}", decimals, x)
    }
}

fn print_result(result: &TransformRunResult) {
    println!("\n{} TRANSFORM", result.kind.as_str().to_uppercase());
    println!("  {}", result.convention);
    println!(
        "  samples={} points={}",
        result.samples.len(),
        result.outputs.len()
    );
    println!(
        "  source={} stations={} sink={}",
        result.entity_framework.sources.join(", "),
        result.entity_framework.stations.join(" -> "),
        result.entity_framework.sinks.join(", ")
    );
    for output in &result.outputs {
        println!(
            "  {:<12} {}  |.|={}",
            output.label,
            format_complex(output.value, 6),
            to_precision(output.magnitude, 6)
        );
    }
}

/// Entry point (`main()` in the TS source).
pub fn run() {
    print_result(&run_z_transform(ZTransformParams {
        sequence: Some(vec![1.0, 0.5, 0.25, 0.125, 0.0625]),
        z_values: Some(vec![
            ComplexPointInput {
                label: Some("z=2".to_string()),
                re: 2.0,
                im: None,
            },
            ComplexPointInput {
                label: Some("z=1".to_string()),
                re: 1.0,
                im: None,
            },
        ]),
        ..Default::default()
    }));

    let mut laplace_constants = HashMap::new();
    laplace_constants.insert("a".to_string(), 2.0);
    print_result(&run_laplace_transform(LaplaceTransformParams {
        expression: Some("exp(-a*t)".to_string()),
        constants: Some(laplace_constants),
        t0: Some(0.0),
        t1: Some(8.0),
        dt: Some(0.01),
        s_values: Some(vec![
            ComplexPointInput {
                label: Some("s=1".to_string()),
                re: 1.0,
                im: None,
            },
            ComplexPointInput {
                label: Some("s=0.5+i".to_string()),
                re: 0.5,
                im: Some(1.0),
            },
        ]),
        ..Default::default()
    }));

    let mut fourier_constants = HashMap::new();
    fourier_constants.insert("omega0".to_string(), 2.0);
    print_result(&run_fourier_transform(FourierTransformParams {
        expression: Some("sin(omega0*t)".to_string()),
        constants: Some(fourier_constants),
        t0: Some(0.0),
        t1: Some(2.0 * std::f64::consts::PI),
        dt: Some(2.0 * std::f64::consts::PI / 2000.0),
        omega_values: Some(vec![0.0, 2.0, -2.0]),
        ..Default::default()
    }));
}
