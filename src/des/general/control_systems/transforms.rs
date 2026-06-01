//! Control-system view of the transform engines.
//!
//! The numerical implementations live in `general::signal_transforms`; this
//! module gives control code a single place to ask what a transform is for.

pub use crate::des::general::signal_transforms::{
    run_dft_transform, run_discrete_fourier_transform, run_fft_transform, run_fourier_transform,
    run_laplace_transform, run_mellin_transform, run_radon_transform, run_wavelet_transform,
    run_z_transform, ComplexPointInput, ComplexValue, DiscreteFourierTransformParams,
    FastFourierTransformParams, FourierTransformParams, LaplaceTransformParams,
    MellinTransformParams, QuadratureRule, RadonProjectionInput, RadonRunResult,
    RadonTransformParams, TransformKind, TransformRunResult, WaveletMother, WaveletPointInput,
    WaveletTransformParams, ZTransformParams,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlAnalysisDomain {
    ContinuousTime,
    DiscreteTime,
    FrequencyAnalysis,
    TimeFrequency,
    ScaleInvariant,
    Tomography,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControlTransformDescriptor {
    pub kind: TransformKind,
    pub domain: ControlAnalysisDomain,
    pub diagonalizes: &'static str,
    pub control_use: &'static str,
}

pub fn transform_descriptor(kind: TransformKind) -> ControlTransformDescriptor {
    match kind {
        TransformKind::Z => ControlTransformDescriptor {
            kind,
            domain: ControlAnalysisDomain::DiscreteTime,
            diagonalizes: "shift and difference operators",
            control_use: "digital control, sampled-data plants, and difference equations",
        },
        TransformKind::Laplace => ControlTransformDescriptor {
            kind,
            domain: ControlAnalysisDomain::ContinuousTime,
            diagonalizes: "linear differential operators",
            control_use: "transfer functions, poles, stability, and continuous-time feedback",
        },
        TransformKind::Fourier => ControlTransformDescriptor {
            kind,
            domain: ControlAnalysisDomain::FrequencyAnalysis,
            diagonalizes: "translation-invariant convolution",
            control_use: "frequency response, filtering, spectra, and disturbance analysis",
        },
        TransformKind::Dft => ControlTransformDescriptor {
            kind,
            domain: ControlAnalysisDomain::DiscreteTime,
            diagonalizes: "finite circular shifts and circular convolution",
            control_use: "sampled spectra, FIR/IIR diagnostics, and finite-horizon DSP",
        },
        TransformKind::Fft => ControlTransformDescriptor {
            kind,
            domain: ControlAnalysisDomain::DiscreteTime,
            diagonalizes: "finite circular shifts using a fast DFT factorization",
            control_use: "real-time spectral monitoring and fast convolution in controllers",
        },
        TransformKind::Wavelet => ControlTransformDescriptor {
            kind,
            domain: ControlAnalysisDomain::TimeFrequency,
            diagonalizes: "localized multiscale structure",
            control_use:
                "transient detection, denoising, anomaly features, and multiresolution models",
        },
        TransformKind::Mellin => ControlTransformDescriptor {
            kind,
            domain: ControlAnalysisDomain::ScaleInvariant,
            diagonalizes: "scaling into translation in log coordinates",
            control_use: "scale-invariant signatures and gain/size normalization studies",
        },
        TransformKind::Radon => ControlTransformDescriptor {
            kind,
            domain: ControlAnalysisDomain::Tomography,
            diagonalizes: "line-integral projection geometry",
            control_use: "tomographic sensing, inverse problems, and imaging feedback loops",
        },
    }
}

pub fn engineering_core_trio() -> [TransformKind; 3] {
    [
        TransformKind::Fourier,
        TransformKind::Laplace,
        TransformKind::Z,
    ]
}
