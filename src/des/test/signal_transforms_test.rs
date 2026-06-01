//! Port of src/des/test/signal-transforms-test.ts
//!
//! Tests Z, Laplace, and Fourier transform DES station graphs. Groups [1]-[3]
//! and the direct-call validation cases [5.1]-[5.3] are ported faithfully. The
//! JSON-registry integration groups [4] and the Zod-schema cases [5.4]/[5.5]
//! depend on `des-registry` (`get_model`/`run_from_spec`) which is not yet
//! ported; those are deferred (see PORT NOTE below).
#![allow(dead_code)]

// PORT NOTE: groups [4], [5.4], [5.5] depend on general::des_registry
// (get_model / run_from_spec) which is not yet ported; those cases are deferred.

#[cfg(test)]
mod tests {
    use crate::des::general::signal_transforms::{
        run_dft_transform, run_fft_transform, run_fourier_transform, run_laplace_transform,
        run_mellin_transform, run_radon_transform, run_wavelet_transform, run_z_transform,
        ComplexPointInput, DftTransformParams, FftTransformParams, FourierTransformParams,
        LaplaceTransformParams, MellinTransformParams, QuadratureRule, RadonTransformParams,
        WaveletKind, WaveletTransformParams, ZTransformParams,
    };
    use std::collections::HashMap;

    /// Relative comparison mirroring the TS `close(a,b,tol)`.
    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * 1.0_f64.max(a.abs()).max(b.abs())
    }

    fn constants(pairs: &[(&str, f64)]) -> Option<HashMap<String, f64>> {
        Some(pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect())
    }

    fn point(label: &str, re: f64) -> ComplexPointInput {
        ComplexPointInput {
            label: Some(label.to_string()),
            re,
            im: None,
        }
    }

    fn panic_message(err: Box<dyn std::any::Any + Send>) -> String {
        if let Some(s) = err.downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = err.downcast_ref::<String>() {
            s.clone()
        } else {
            "<non-string panic>".to_string()
        }
    }

    fn assert_panics_with<F, R>(f: F, fragment: &str)
    where
        F: FnOnce() -> R + std::panic::UnwindSafe,
    {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let res = std::panic::catch_unwind(f);
        std::panic::set_hook(prev);
        let err = res.err().expect("expected a panic");
        let msg = panic_message(err);
        assert!(
            msg.contains(fragment),
            "panic {msg:?} did not contain {fragment:?}"
        );
    }

    // [1] Z-transform station graph
    #[test]
    fn z_transform_station_graph() {
        let z = run_z_transform(ZTransformParams {
            sequence: Some(vec![1.0, 2.0, 3.0]),
            z_values: Some(vec![point("z=2", 2.0), point("z=-1", -1.0)]),
            ..Default::default()
        });
        assert!(z.validation.iter().all(|c| c.passed));
        assert!(close(z.outputs[0].value.re, 2.75, 1e-12));
        assert!(z.outputs[0].value.im.abs() < 1e-12);
        assert!(close(z.outputs[1].value.re, 2.0, 1e-12));
        assert!(z.outputs[1].value.im.abs() < 1e-12);
        assert_eq!(z.entity_framework.sources.len(), 1);
        assert_eq!(z.entity_framework.stations.len(), 2);
        assert_eq!(z.entity_framework.sinks.len(), 1);
        assert!(z
            .entity_framework
            .movable_entities
            .iter()
            .any(|s| s == "TransformSampleToken"));
        assert!(z
            .entity_framework
            .movable_entities
            .iter()
            .any(|s| s == "TransformContributionToken"));

        let geometric = run_z_transform(ZTransformParams {
            expression: Some("a^n".to_string()),
            constants: constants(&[("a", 0.5)]),
            terms: Some(4),
            z_values: Some(vec![point("z=2", 2.0)]),
            ..Default::default()
        });
        assert!(close(geometric.outputs[0].value.re, 1.328125, 1e-12));
    }

    // [2] Laplace transform station graph
    #[test]
    fn laplace_transform_station_graph() {
        let laplace = run_laplace_transform(LaplaceTransformParams {
            expression: Some("exp(-a*t)".to_string()),
            constants: constants(&[("a", 2.0)]),
            t0: Some(0.0),
            t1: Some(8.0),
            dt: Some(0.002),
            quadrature: Some(QuadratureRule::Trapezoid),
            s_values: Some(vec![point("s=1", 1.0)]),
            ..Default::default()
        });
        let exact_finite_window = (1.0 - (-24.0_f64).exp()) / 3.0;
        assert!(laplace.validation.iter().all(|c| c.passed));
        assert!(close(
            laplace.outputs[0].value.re,
            exact_finite_window,
            1e-6
        ));
        assert!(close(laplace.outputs[0].value.im, 0.0, 1e-9));
    }

    // [3] Fourier transform station graph
    #[test]
    fn fourier_transform_station_graph() {
        let dt = 2.0 * std::f64::consts::PI / 2000.0;
        let fourier = run_fourier_transform(FourierTransformParams {
            expression: Some("sin(omega0*t)".to_string()),
            constants: constants(&[("omega0", 2.0)]),
            t0: Some(0.0),
            t1: Some(2.0 * std::f64::consts::PI),
            dt: Some(dt),
            quadrature: Some(QuadratureRule::Trapezoid),
            omega_values: Some(vec![0.0, 2.0, -2.0]),
            ..Default::default()
        });
        assert!(fourier.validation.iter().all(|c| c.passed));
        // DC component near zero.
        assert!(close(fourier.outputs[0].value.re, 0.0, 1e-9));
        assert!(close(fourier.outputs[0].value.im, 0.0, 1e-9));
        // Positive frequency coefficient is -i*pi.
        assert!(close(fourier.outputs[1].value.re, 0.0, 1e-6));
        assert!(close(
            fourier.outputs[1].value.im,
            -std::f64::consts::PI,
            1e-6
        ));
        // Negative frequency coefficient is +i*pi.
        assert!(close(fourier.outputs[2].value.re, 0.0, 1e-6));
        assert!(close(
            fourier.outputs[2].value.im,
            std::f64::consts::PI,
            1e-6
        ));
    }

    #[test]
    fn dft_and_fft_transform_bins() {
        let params = DftTransformParams {
            sequence: Some(vec![1.0, 0.0, -1.0, 0.0]),
            k_values: Some(vec![0, 1, 2, 3]),
            ..Default::default()
        };
        let dft = run_dft_transform(params);
        assert!(dft.validation.iter().all(|c| c.passed));
        assert!(close(dft.outputs[0].value.re, 0.0, 1e-12));
        assert!(close(dft.outputs[1].value.re, 2.0, 1e-12));
        assert!(close(dft.outputs[2].value.re, 0.0, 1e-12));
        assert!(close(dft.outputs[3].value.re, 2.0, 1e-12));
        assert!(dft.outputs.iter().all(|o| o.value.im.abs() < 1e-12));

        let fft = run_fft_transform(FftTransformParams {
            sequence: Some(vec![1.0, 0.0, -1.0, 0.0]),
            ..Default::default()
        });
        assert!(fft.validation.iter().all(|c| c.passed));
        assert_eq!(fft.outputs.len(), 4);
        assert!(close(
            fft.outputs[1].value.re,
            dft.outputs[1].value.re,
            1e-12
        ));
        assert_eq!(fft.kind.as_str(), "fft");
    }

    #[test]
    fn wavelet_mellin_and_radon_station_graphs() {
        let wavelet = run_wavelet_transform(WaveletTransformParams {
            expression: Some("1".to_string()),
            t0: Some(0.0),
            t1: Some(1.0),
            dt: Some(0.001),
            quadrature: Some(QuadratureRule::Trapezoid),
            scales: Some(vec![1.0]),
            translations: Some(vec![0.0]),
            mother: Some(WaveletKind::Haar),
            ..Default::default()
        });
        assert!(wavelet.validation.iter().all(|c| c.passed));
        assert!(wavelet.outputs[0].value.re.abs() < 1e-3);

        let mellin = run_mellin_transform(MellinTransformParams {
            expression: Some("1".to_string()),
            x0: Some(1.0),
            x1: Some(3.0),
            dx: Some(0.001),
            quadrature: Some(QuadratureRule::Trapezoid),
            s_values: Some(vec![point("s=1", 1.0)]),
            ..Default::default()
        });
        assert!(mellin.validation.iter().all(|c| c.passed));
        assert!(close(mellin.outputs[0].value.re, 2.0, 1e-6));

        let radon = run_radon_transform(RadonTransformParams {
            image: Some(vec![
                vec![0.0, 1.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 1.0, 0.0],
            ]),
            theta_values: Some(vec![0.0]),
            rho_values: Some(vec![0.0]),
            line_width: Some(1.0),
            ..Default::default()
        });
        assert!(radon.validation.iter().all(|c| c.passed));
        assert!(close(radon.outputs[0].value.re, 3.0, 1e-12));
    }

    // [5] Input validation (direct-call cases 5.1–5.3)
    #[test]
    fn input_validation_requires_sequence_or_expression() {
        assert_panics_with(
            || {
                run_z_transform(ZTransformParams {
                    z_values: Some(vec![ComplexPointInput {
                        label: None,
                        re: 1.0,
                        im: None,
                    }]),
                    ..Default::default()
                })
            },
            "requires either",
        );
        assert_panics_with(
            || {
                run_laplace_transform(LaplaceTransformParams {
                    s_values: Some(vec![ComplexPointInput {
                        label: None,
                        re: 1.0,
                        im: None,
                    }]),
                    ..Default::default()
                })
            },
            "requires either",
        );
        assert_panics_with(
            || {
                run_fourier_transform(FourierTransformParams {
                    omega_values: Some(vec![1.0]),
                    ..Default::default()
                })
            },
            "requires either",
        );
    }
}
