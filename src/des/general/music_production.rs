//! Generative music production primitives.
//!
//! This module is intentionally dependency-free and deterministic. It provides
//! a small audio buffer, microtonal tuning, pitch-bend curves, simple synthesis,
//! audio effects, FFT-backed spectrum analysis, and a default three-minute
//! instrumental generator with generated vocal-chop textures. External samples
//! are represented through a license manifest so callers can keep provenance at
//! the audio boundary.

use std::f64::consts::TAU;
use std::io::{self, Read, Write};
use std::net::{Ipv4Addr, Ipv6Addr, TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::time::Duration;

use native_tls::TlsConnector;

use crate::des::general::des_base::visual_block::{
    visual_block_graph_ir, Metadata, VisualBlock, VisualBlockConnectionOptions, VisualBlockLayout,
    VisualBlockOptions, VisualBlockPortSpec, VisualBlockRole, VisualBlockStyle, VisualPortInput,
    VisualPortOptions,
};
use crate::des::general::des_spec::JsonValue;
use crate::des::general::prng::mulberry32;
use crate::des::general::signal_transforms::{run_fft_transform, FastFourierTransformParams};
use crate::des::shared::capabilities::RandomSource;

pub const DEFAULT_SAMPLE_RATE: u32 = 44_100;
pub const DEFAULT_SONG_SECONDS: f64 = 180.0;
pub const MAX_MUSIC_SAMPLE_SEED_BYTES: u64 = 96 * 1024 * 1024;
pub const MAX_MUSIC_PUBLIC_MEDIA_SEED_BYTES: usize = 2 * 1024 * 1024;

const MUSIC_MEDIA_HTTP_TIMEOUT_MS: u64 = 20_000;
const MUSIC_MEDIA_HTTP_MAX_TEXT_BYTES: usize = 6 * 1024 * 1024;
const MUSIC_MEDIA_HTTP_REDIRECT_LIMIT: usize = 5;
const MUSIC_MEDIA_USER_AGENT: &str =
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 \
     (KHTML, like Gecko) Chrome/125 Safari/537.36 des-rs-music-media";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MusicFeatureCategory {
    RhythmMeter,
    MelodyPitchTuning,
    HarmonyTonality,
    FormArrangement,
    PerformanceArticulation,
    ProductionMixing,
    SequencingSampling,
    TextureSpectrum,
}

impl MusicFeatureCategory {
    pub fn as_str(self) -> &'static str {
        match self {
            MusicFeatureCategory::RhythmMeter => "rhythm-meter",
            MusicFeatureCategory::MelodyPitchTuning => "melody-pitch-tuning",
            MusicFeatureCategory::HarmonyTonality => "harmony-tonality",
            MusicFeatureCategory::FormArrangement => "form-arrangement",
            MusicFeatureCategory::PerformanceArticulation => "performance-articulation",
            MusicFeatureCategory::ProductionMixing => "production-mixing",
            MusicFeatureCategory::SequencingSampling => "sequencing-sampling",
            MusicFeatureCategory::TextureSpectrum => "texture-spectrum",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MusicFeature {
    pub italian: &'static str,
    pub english: &'static str,
    pub category: MusicFeatureCategory,
}

pub const ITALIAN_MUSIC_FEATURES: [MusicFeature; 50] = [
    MusicFeature {
        italian: "ritmo",
        english: "rhythm",
        category: MusicFeatureCategory::RhythmMeter,
    },
    MusicFeature {
        italian: "melodia",
        english: "melody",
        category: MusicFeatureCategory::MelodyPitchTuning,
    },
    MusicFeature {
        italian: "armonia",
        english: "harmony",
        category: MusicFeatureCategory::HarmonyTonality,
    },
    MusicFeature {
        italian: "timbro",
        english: "timbre",
        category: MusicFeatureCategory::TextureSpectrum,
    },
    MusicFeature {
        italian: "tempo",
        english: "tempo",
        category: MusicFeatureCategory::RhythmMeter,
    },
    MusicFeature {
        italian: "dinamica",
        english: "dynamics",
        category: MusicFeatureCategory::PerformanceArticulation,
    },
    MusicFeature {
        italian: "intensità",
        english: "intensity",
        category: MusicFeatureCategory::PerformanceArticulation,
    },
    MusicFeature {
        italian: "altezza",
        english: "pitch height",
        category: MusicFeatureCategory::MelodyPitchTuning,
    },
    MusicFeature {
        italian: "durata",
        english: "duration",
        category: MusicFeatureCategory::RhythmMeter,
    },
    MusicFeature {
        italian: "pausa",
        english: "rest",
        category: MusicFeatureCategory::FormArrangement,
    },
    MusicFeature {
        italian: "battito",
        english: "beat",
        category: MusicFeatureCategory::RhythmMeter,
    },
    MusicFeature {
        italian: "misura",
        english: "bar",
        category: MusicFeatureCategory::RhythmMeter,
    },
    MusicFeature {
        italian: "accordo",
        english: "chord",
        category: MusicFeatureCategory::HarmonyTonality,
    },
    MusicFeature {
        italian: "tonalità",
        english: "key",
        category: MusicFeatureCategory::HarmonyTonality,
    },
    MusicFeature {
        italian: "scala",
        english: "scale",
        category: MusicFeatureCategory::MelodyPitchTuning,
    },
    MusicFeature {
        italian: "fraseggio",
        english: "phrasing",
        category: MusicFeatureCategory::PerformanceArticulation,
    },
    MusicFeature {
        italian: "articolazione",
        english: "articulation",
        category: MusicFeatureCategory::PerformanceArticulation,
    },
    MusicFeature {
        italian: "legato",
        english: "legato",
        category: MusicFeatureCategory::PerformanceArticulation,
    },
    MusicFeature {
        italian: "staccato",
        english: "staccato",
        category: MusicFeatureCategory::PerformanceArticulation,
    },
    MusicFeature {
        italian: "vibrato",
        english: "vibrato",
        category: MusicFeatureCategory::PerformanceArticulation,
    },
    MusicFeature {
        italian: "cadenza",
        english: "cadence",
        category: MusicFeatureCategory::HarmonyTonality,
    },
    MusicFeature {
        italian: "modulazione",
        english: "modulation",
        category: MusicFeatureCategory::HarmonyTonality,
    },
    MusicFeature {
        italian: "improvvisazione",
        english: "improvisation",
        category: MusicFeatureCategory::PerformanceArticulation,
    },
    MusicFeature {
        italian: "arrangiamento",
        english: "arrangement",
        category: MusicFeatureCategory::FormArrangement,
    },
    MusicFeature {
        italian: "orchestrazione",
        english: "orchestration",
        category: MusicFeatureCategory::FormArrangement,
    },
    MusicFeature {
        italian: "tessitura",
        english: "tessitura",
        category: MusicFeatureCategory::TextureSpectrum,
    },
    MusicFeature {
        italian: "metro",
        english: "meter",
        category: MusicFeatureCategory::RhythmMeter,
    },
    MusicFeature {
        italian: "sincope",
        english: "syncopation",
        category: MusicFeatureCategory::RhythmMeter,
    },
    MusicFeature {
        italian: "accento",
        english: "accent",
        category: MusicFeatureCategory::RhythmMeter,
    },
    MusicFeature {
        italian: "groove",
        english: "groove",
        category: MusicFeatureCategory::RhythmMeter,
    },
    MusicFeature {
        italian: "andamento",
        english: "movement",
        category: MusicFeatureCategory::RhythmMeter,
    },
    MusicFeature {
        italian: "espressione",
        english: "expression",
        category: MusicFeatureCategory::PerformanceArticulation,
    },
    MusicFeature {
        italian: "interpretazione",
        english: "interpretation",
        category: MusicFeatureCategory::PerformanceArticulation,
    },
    MusicFeature {
        italian: "registrazione",
        english: "recording",
        category: MusicFeatureCategory::ProductionMixing,
    },
    MusicFeature {
        italian: "equalizzazione",
        english: "equalization",
        category: MusicFeatureCategory::ProductionMixing,
    },
    MusicFeature {
        italian: "riverbero",
        english: "reverb",
        category: MusicFeatureCategory::ProductionMixing,
    },
    MusicFeature {
        italian: "eco",
        english: "echo",
        category: MusicFeatureCategory::ProductionMixing,
    },
    MusicFeature {
        italian: "distorsione",
        english: "distortion",
        category: MusicFeatureCategory::ProductionMixing,
    },
    MusicFeature {
        italian: "compressione",
        english: "compression",
        category: MusicFeatureCategory::ProductionMixing,
    },
    MusicFeature {
        italian: "campionamento",
        english: "sampling",
        category: MusicFeatureCategory::SequencingSampling,
    },
    MusicFeature {
        italian: "loop",
        english: "loop",
        category: MusicFeatureCategory::SequencingSampling,
    },
    MusicFeature {
        italian: "sequenza",
        english: "sequence",
        category: MusicFeatureCategory::SequencingSampling,
    },
    MusicFeature {
        italian: "armonizzazione",
        english: "harmonization",
        category: MusicFeatureCategory::HarmonyTonality,
    },
    MusicFeature {
        italian: "contrappunto",
        english: "counterpoint",
        category: MusicFeatureCategory::HarmonyTonality,
    },
    MusicFeature {
        italian: "polifonia",
        english: "polyphony",
        category: MusicFeatureCategory::TextureSpectrum,
    },
    MusicFeature {
        italian: "monodia",
        english: "monody",
        category: MusicFeatureCategory::TextureSpectrum,
    },
    MusicFeature {
        italian: "modalità",
        english: "modality",
        category: MusicFeatureCategory::HarmonyTonality,
    },
    MusicFeature {
        italian: "intonazione",
        english: "intonation",
        category: MusicFeatureCategory::MelodyPitchTuning,
    },
    MusicFeature {
        italian: "pulsazione",
        english: "pulse",
        category: MusicFeatureCategory::RhythmMeter,
    },
    MusicFeature {
        italian: "spettro",
        english: "spectrum",
        category: MusicFeatureCategory::TextureSpectrum,
    },
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MusicFeatureCoverage {
    pub covered: Vec<String>,
    pub missing: Vec<String>,
    pub extras: Vec<String>,
}

impl MusicFeatureCoverage {
    pub fn is_complete(&self) -> bool {
        self.missing.is_empty()
    }
}

pub fn italian_music_feature_catalog() -> &'static [MusicFeature] {
    &ITALIAN_MUSIC_FEATURES
}

pub fn all_italian_music_feature_terms() -> Vec<String> {
    italian_music_feature_catalog()
        .iter()
        .map(|feature| feature.italian.to_string())
        .collect()
}

pub fn find_italian_music_feature(term: &str) -> Option<&'static MusicFeature> {
    let term = canonical_music_feature_term(term);
    italian_music_feature_catalog()
        .iter()
        .find(|feature| canonical_music_feature_term(feature.italian) == term)
}

pub fn music_feature_coverage<'a, I>(terms: I) -> MusicFeatureCoverage
where
    I: IntoIterator<Item = &'a str>,
{
    let supplied = terms
        .into_iter()
        .map(canonical_music_feature_term)
        .filter(|term| !term.is_empty())
        .collect::<Vec<_>>();
    let mut covered = Vec::new();
    let mut missing = Vec::new();
    for feature in italian_music_feature_catalog() {
        let term = canonical_music_feature_term(feature.italian);
        if supplied.iter().any(|supplied| supplied == &term) {
            covered.push(feature.italian.to_string());
        } else {
            missing.push(feature.italian.to_string());
        }
    }
    let mut extras = Vec::new();
    for term in supplied {
        if find_italian_music_feature(&term).is_none() && !extras.iter().any(|e| e == &term) {
            extras.push(term);
        }
    }
    MusicFeatureCoverage {
        covered,
        missing,
        extras,
    }
}

pub fn validate_italian_music_feature_coverage(
    terms: &[String],
) -> Result<MusicFeatureCoverage, String> {
    let coverage = music_feature_coverage(terms.iter().map(String::as_str));
    if coverage.is_complete() {
        Ok(coverage)
    } else {
        Err(format!(
            "missing Italian music features: {}",
            coverage.missing.join(", ")
        ))
    }
}

fn canonical_music_feature_term(term: &str) -> String {
    term.trim().to_lowercase()
}

fn require_finite(name: &str, value: f64) {
    assert!(value.is_finite(), "{name} must be finite");
}

fn clamp_unit(v: f64) -> f64 {
    v.clamp(-1.0, 1.0)
}

fn cents_to_ratio(cents: f64) -> f64 {
    2.0_f64.powf(cents / 1200.0)
}

/// Mono floating-point audio buffer. Samples are not hard-clipped until export.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioBuffer {
    pub sample_rate: u32,
    pub samples: Vec<f64>,
}

impl AudioBuffer {
    pub fn silence(sample_rate: u32, duration_seconds: f64) -> Self {
        assert!(sample_rate > 0, "sample_rate must be positive");
        require_finite("duration_seconds", duration_seconds);
        assert!(
            duration_seconds >= 0.0,
            "duration_seconds must be non-negative"
        );
        let len = (duration_seconds * sample_rate as f64).round() as usize;
        Self {
            sample_rate,
            samples: vec![0.0; len],
        }
    }

    pub fn from_samples(sample_rate: u32, samples: Vec<f64>) -> Self {
        assert!(sample_rate > 0, "sample_rate must be positive");
        assert!(
            samples.iter().all(|s| s.is_finite()),
            "samples must all be finite"
        );
        Self {
            sample_rate,
            samples,
        }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn duration_seconds(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate as f64
    }

    pub fn sample_index(&self, time_seconds: f64) -> usize {
        require_finite("time_seconds", time_seconds);
        (time_seconds.max(0.0) * self.sample_rate as f64).round() as usize
    }

    pub fn mix_sample(&mut self, index: usize, sample: f64) {
        if index < self.samples.len() {
            self.samples[index] += sample;
        }
    }

    pub fn mix_in(&mut self, other: &AudioBuffer, start_seconds: f64, gain: f64) {
        assert_eq!(
            self.sample_rate, other.sample_rate,
            "sample rates must match to mix buffers"
        );
        require_finite("gain", gain);
        let start = self.sample_index(start_seconds);
        for (i, sample) in other.samples.iter().enumerate() {
            self.mix_sample(start + i, sample * gain);
        }
    }

    pub fn peak(&self) -> f64 {
        self.samples
            .iter()
            .fold(0.0, |peak, sample| peak.max(sample.abs()))
    }

    pub fn rms(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let energy: f64 = self.samples.iter().map(|s| s * s).sum();
        (energy / self.samples.len() as f64).sqrt()
    }

    pub fn normalize_peak(&mut self, target_peak: f64) {
        require_finite("target_peak", target_peak);
        assert!(target_peak >= 0.0, "target_peak must be non-negative");
        let peak = self.peak();
        if peak > 0.0 {
            let scale = target_peak / peak;
            for sample in &mut self.samples {
                *sample *= scale;
            }
        }
    }

    pub fn write_wav16(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let data_bytes = self
            .samples
            .len()
            .checked_mul(2)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "wav too large"))?;
        if data_bytes > u32::MAX as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "wav data chunk exceeds 32-bit RIFF size",
            ));
        }
        let mut file = std::fs::File::create(path)?;
        let data_bytes = data_bytes as u32;
        let riff_size = 36u32
            .checked_add(data_bytes)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "wav too large"))?;

        file.write_all(b"RIFF")?;
        file.write_all(&riff_size.to_le_bytes())?;
        file.write_all(b"WAVE")?;
        file.write_all(b"fmt ")?;
        file.write_all(&16u32.to_le_bytes())?;
        file.write_all(&1u16.to_le_bytes())?;
        file.write_all(&1u16.to_le_bytes())?;
        file.write_all(&self.sample_rate.to_le_bytes())?;
        file.write_all(&(self.sample_rate * 2).to_le_bytes())?;
        file.write_all(&2u16.to_le_bytes())?;
        file.write_all(&16u16.to_le_bytes())?;
        file.write_all(b"data")?;
        file.write_all(&data_bytes.to_le_bytes())?;
        for sample in &self.samples {
            let pcm = (clamp_unit(*sample) * i16::MAX as f64).round() as i16;
            file.write_all(&pcm.to_le_bytes())?;
        }
        Ok(())
    }
}

/// Equal-division-of-octave scale with an arbitrary mode.
#[derive(Clone, Debug, PartialEq)]
pub struct MicrotonalScale {
    pub name: String,
    pub base_frequency_hz: f64,
    pub divisions_per_octave: u32,
    pub octave_ratio: f64,
    pub mode_steps: Vec<i32>,
}

impl MicrotonalScale {
    pub fn edo(
        name: impl Into<String>,
        divisions_per_octave: u32,
        base_frequency_hz: f64,
        mode_steps: Vec<i32>,
    ) -> Self {
        assert!(
            divisions_per_octave > 0,
            "divisions_per_octave must be positive"
        );
        require_finite("base_frequency_hz", base_frequency_hz);
        assert!(
            base_frequency_hz > 0.0,
            "base_frequency_hz must be positive"
        );
        assert!(!mode_steps.is_empty(), "mode_steps must not be empty");
        Self {
            name: name.into(),
            base_frequency_hz,
            divisions_per_octave,
            octave_ratio: 2.0,
            mode_steps,
        }
    }

    /// A broad 19-EDO mode suited to microtonal bass lines and bent leads.
    pub fn rave_collage_19_edo() -> Self {
        Self::edo(
            "19-EDO breakbeat mode",
            19,
            55.0,
            vec![0, 2, 3, 5, 7, 8, 10, 12, 14, 15, 17],
        )
    }

    pub fn step_to_frequency(&self, step: i32) -> f64 {
        self.base_frequency_hz
            * self
                .octave_ratio
                .powf(step as f64 / self.divisions_per_octave as f64)
    }

    pub fn degree_to_step(&self, degree: i32, octave: i32) -> i32 {
        let len = self.mode_steps.len() as i32;
        let mode_cycle = degree.div_euclid(len);
        let index = degree.rem_euclid(len) as usize;
        self.mode_steps[index] + (mode_cycle + octave) * self.divisions_per_octave as i32
    }

    pub fn degree_to_frequency(&self, degree: i32, octave: i32) -> f64 {
        self.step_to_frequency(self.degree_to_step(degree, octave))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PitchBendPoint {
    /// Normalized position inside the note, from `0.0` to `1.0`.
    pub position: f64,
    pub cents: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PitchBendCurve {
    pub points: Vec<PitchBendPoint>,
}

impl PitchBendCurve {
    pub fn flat() -> Self {
        Self {
            points: vec![
                PitchBendPoint {
                    position: 0.0,
                    cents: 0.0,
                },
                PitchBendPoint {
                    position: 1.0,
                    cents: 0.0,
                },
            ],
        }
    }

    pub fn from_points(mut points: Vec<PitchBendPoint>) -> Self {
        assert!(
            !points.is_empty(),
            "pitch bend curve needs at least one point"
        );
        for point in &points {
            require_finite("pitch bend position", point.position);
            require_finite("pitch bend cents", point.cents);
            assert!(
                (0.0..=1.0).contains(&point.position),
                "pitch bend positions must be in [0, 1]"
            );
        }
        points.sort_by(|a, b| a.position.partial_cmp(&b.position).unwrap());
        if points[0].position > 0.0 {
            let cents = points[0].cents;
            points.insert(
                0,
                PitchBendPoint {
                    position: 0.0,
                    cents,
                },
            );
        }
        if points.last().unwrap().position < 1.0 {
            let cents = points.last().unwrap().cents;
            points.push(PitchBendPoint {
                position: 1.0,
                cents,
            });
        }
        Self { points }
    }

    pub fn swoop(start_cents: f64, end_cents: f64) -> Self {
        Self::from_points(vec![
            PitchBendPoint {
                position: 0.0,
                cents: start_cents,
            },
            PitchBendPoint {
                position: 1.0,
                cents: end_cents,
            },
        ])
    }

    pub fn cents_at(&self, position: f64) -> f64 {
        let position = position.clamp(0.0, 1.0);
        for pair in self.points.windows(2) {
            let a = pair[0];
            let b = pair[1];
            if (a.position..=b.position).contains(&position) {
                let span = b.position - a.position;
                if span <= f64::EPSILON {
                    return b.cents;
                }
                let u = (position - a.position) / span;
                return a.cents + (b.cents - a.cents) * u;
            }
        }
        self.points.last().map(|p| p.cents).unwrap_or(0.0)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NoteEvent {
    pub start_seconds: f64,
    pub duration_seconds: f64,
    pub degree: i32,
    pub octave: i32,
    pub velocity: f64,
    pub bend: PitchBendCurve,
}

impl NoteEvent {
    pub fn new(start_seconds: f64, duration_seconds: f64, degree: i32, octave: i32) -> Self {
        Self {
            start_seconds,
            duration_seconds,
            degree,
            octave,
            velocity: 1.0,
            bend: PitchBendCurve::flat(),
        }
    }

    pub fn with_velocity(mut self, velocity: f64) -> Self {
        self.velocity = velocity;
        self
    }

    pub fn with_bend(mut self, bend: PitchBendCurve) -> Self {
        self.bend = bend;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Waveform {
    Sine,
    Triangle,
    Saw,
    Square,
    Pulse(f64),
    Noise,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdsrEnvelope {
    pub attack: f64,
    pub decay: f64,
    pub sustain: f64,
    pub release: f64,
}

impl AdsrEnvelope {
    pub fn percussive() -> Self {
        Self {
            attack: 0.004,
            decay: 0.08,
            sustain: 0.35,
            release: 0.06,
        }
    }

    pub fn pad() -> Self {
        Self {
            attack: 0.08,
            decay: 0.35,
            sustain: 0.72,
            release: 0.4,
        }
    }

    pub fn amplitude_at(&self, t: f64, duration: f64) -> f64 {
        if t < 0.0 || duration <= 0.0 || t > duration {
            return 0.0;
        }
        if self.attack > 0.0 && t < self.attack {
            return t / self.attack;
        }
        let decay_end = self.attack + self.decay;
        if self.decay > 0.0 && t < decay_end {
            let u = (t - self.attack) / self.decay;
            return 1.0 + (self.sustain - 1.0) * u;
        }
        let release = self.release.min(duration);
        let release_start = (duration - release).max(0.0);
        if t < release_start || release <= 0.0 {
            return self.sustain;
        }
        let u = ((t - release_start) / release).clamp(0.0, 1.0);
        self.sustain * (1.0 - u)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SynthVoice {
    pub waveform: Waveform,
    pub envelope: AdsrEnvelope,
    pub gain: f64,
    pub detune_cents: f64,
}

impl SynthVoice {
    pub fn render_note(
        &self,
        buffer: &mut AudioBuffer,
        scale: &MicrotonalScale,
        note: &NoteEvent,
        rng: &mut impl RandomSource,
    ) {
        if note.duration_seconds <= 0.0 || note.velocity <= 0.0 {
            return;
        }
        require_finite("note.start_seconds", note.start_seconds);
        require_finite("note.duration_seconds", note.duration_seconds);
        let start = buffer.sample_index(note.start_seconds);
        let end = buffer
            .sample_index(note.start_seconds + note.duration_seconds)
            .min(buffer.len());
        let base_frequency = scale.degree_to_frequency(note.degree, note.octave);
        let sample_rate = buffer.sample_rate as f64;
        let mut phase = 0.0;
        for i in start..end {
            let elapsed = (i - start) as f64 / sample_rate;
            let position = elapsed / note.duration_seconds;
            let bend_cents = note.bend.cents_at(position) + self.detune_cents;
            let frequency = base_frequency * cents_to_ratio(bend_cents);
            phase = (phase + TAU * frequency / sample_rate).rem_euclid(TAU);
            let raw = oscillator_sample(self.waveform, phase, rng);
            let amp = self.envelope.amplitude_at(elapsed, note.duration_seconds)
                * note.velocity
                * self.gain;
            buffer.mix_sample(i, raw * amp);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InstrumentRole {
    Bass,
    Lead,
    Pad,
    Percussion,
    Texture,
}

impl InstrumentRole {
    pub fn as_str(self) -> &'static str {
        match self {
            InstrumentRole::Bass => "bass",
            InstrumentRole::Lead => "lead",
            InstrumentRole::Pad => "pad",
            InstrumentRole::Percussion => "percussion",
            InstrumentRole::Texture => "texture",
        }
    }

    fn probe_note(self) -> (i32, i32, f64) {
        match self {
            InstrumentRole::Bass => (0, 0, 0.48),
            InstrumentRole::Lead => (9, 2, 0.42),
            InstrumentRole::Pad => (5, 2, 0.72),
            InstrumentRole::Percussion => (10, 1, 0.16),
            InstrumentRole::Texture => (14, 2, 0.34),
        }
    }

    fn target_centroid_hz(self) -> f64 {
        match self {
            InstrumentRole::Bass => 260.0,
            InstrumentRole::Lead => 1_650.0,
            InstrumentRole::Pad => 820.0,
            InstrumentRole::Percussion => 4_200.0,
            InstrumentRole::Texture => 2_600.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstrumentDiscoveryMethod {
    /// Treat each synthesis operation as an action and keep the highest-value
    /// policy trajectory for the role.
    MdpPolicySearch,
    /// Prefer candidates whose score remains robust under noisy observations.
    PomdpBeliefSearch,
    /// Select a role-covering, spectrally diverse subset with integer choices.
    IntegerProgramSelection,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiscoveryObjective {
    pub role_fit_weight: f64,
    pub novelty_weight: f64,
    pub spectral_spread_weight: f64,
    pub anti_mimicry_weight: f64,
}

impl Default for DiscoveryObjective {
    fn default() -> Self {
        Self {
            role_fit_weight: 0.38,
            novelty_weight: 0.28,
            spectral_spread_weight: 0.18,
            anti_mimicry_weight: 0.16,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct InstrumentDiscoverySpec {
    pub seed: u32,
    pub sample_rate: u32,
    pub candidates_per_role: usize,
    pub roles: Vec<InstrumentRole>,
    pub methods: Vec<InstrumentDiscoveryMethod>,
    pub objective: DiscoveryObjective,
}

impl Default for InstrumentDiscoverySpec {
    fn default() -> Self {
        Self {
            seed: 0x1a7e_cafe,
            sample_rate: 16_000,
            candidates_per_role: 10,
            roles: vec![
                InstrumentRole::Bass,
                InstrumentRole::Lead,
                InstrumentRole::Pad,
                InstrumentRole::Percussion,
                InstrumentRole::Texture,
            ],
            methods: vec![
                InstrumentDiscoveryMethod::MdpPolicySearch,
                InstrumentDiscoveryMethod::PomdpBeliefSearch,
                InstrumentDiscoveryMethod::IntegerProgramSelection,
            ],
            objective: DiscoveryObjective::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SyntheticLayer {
    pub waveform: Waveform,
    pub gain: f64,
    pub harmonic_ratio: f64,
    pub detune_cents: f64,
    pub fm_ratio: f64,
    pub fm_index: f64,
    pub fold: f64,
    pub noise_level: f64,
    pub filter_cutoff_hz: f64,
    pub filter_q: f64,
    pub envelope: AdsrEnvelope,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InstrumentDiscoveryTrace {
    pub method: InstrumentDiscoveryMethod,
    pub score: f64,
    pub role_fit: f64,
    pub novelty: f64,
    pub anti_mimicry: f64,
    pub spectral_centroid_hz: f64,
    pub spectral_flatness: f64,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SyntheticInstrument {
    pub id: String,
    pub display_name: String,
    pub role: InstrumentRole,
    pub layers: Vec<SyntheticLayer>,
    pub transient_noise: f64,
    pub output_gain: f64,
    pub discovery: InstrumentDiscoveryTrace,
}

impl SyntheticInstrument {
    pub fn render_note(
        &self,
        buffer: &mut AudioBuffer,
        scale: &MicrotonalScale,
        note: &NoteEvent,
        rng: &mut impl RandomSource,
    ) {
        if note.duration_seconds <= 0.0 || note.velocity <= 0.0 {
            return;
        }
        let start = buffer.sample_index(note.start_seconds);
        let end = buffer
            .sample_index(note.start_seconds + note.duration_seconds)
            .min(buffer.len());
        let base_frequency = scale.degree_to_frequency(note.degree, note.octave);
        let sample_rate = buffer.sample_rate as f64;
        let mut phases = vec![0.0; self.layers.len()];
        let mut fm_phases = vec![0.0; self.layers.len()];
        let mut filters: Vec<Option<BiquadFilter>> = self
            .layers
            .iter()
            .map(|layer| {
                if layer.filter_cutoff_hz > 0.0 {
                    Some(BiquadFilter::new(
                        FilterMode::LowPass,
                        buffer.sample_rate,
                        layer.filter_cutoff_hz,
                        layer.filter_q.max(0.05),
                    ))
                } else {
                    None
                }
            })
            .collect();

        for i in start..end {
            let elapsed = (i - start) as f64 / sample_rate;
            let position = elapsed / note.duration_seconds;
            let bend_cents = note.bend.cents_at(position);
            let mut sample = 0.0;
            for (layer_index, layer) in self.layers.iter().enumerate() {
                let frequency = base_frequency
                    * layer.harmonic_ratio
                    * cents_to_ratio(bend_cents + layer.detune_cents);
                phases[layer_index] =
                    (phases[layer_index] + TAU * frequency / sample_rate).rem_euclid(TAU);
                fm_phases[layer_index] = (fm_phases[layer_index]
                    + TAU * frequency * layer.fm_ratio.max(0.01) / sample_rate)
                    .rem_euclid(TAU);
                let phase = phases[layer_index] + layer.fm_index * fm_phases[layer_index].sin();
                let mut layer_sample = oscillator_sample(layer.waveform, phase, rng);
                if layer.noise_level > 0.0 {
                    layer_sample += (rng.next_float() * 2.0 - 1.0) * layer.noise_level;
                }
                layer_sample = wavefold(layer_sample, layer.fold);
                if let Some(filter) = filters[layer_index].as_mut() {
                    layer_sample = filter.process_sample(layer_sample);
                }
                let amp = layer.envelope.amplitude_at(elapsed, note.duration_seconds)
                    * layer.gain
                    * note.velocity;
                sample += layer_sample * amp;
            }
            if self.transient_noise > 0.0 {
                let transient = (1.0 - position / 0.08).clamp(0.0, 1.0);
                sample += (rng.next_float() * 2.0 - 1.0) * transient * self.transient_noise;
            }
            buffer.mix_sample(i, sample * self.output_gain);
        }
    }

    pub fn render_probe(
        &self,
        sample_rate: u32,
        scale: &MicrotonalScale,
        seed: u32,
    ) -> AudioBuffer {
        let (degree, octave, duration) = self.role.probe_note();
        let mut rng = mulberry32(seed);
        let mut buffer = AudioBuffer::silence(sample_rate, duration + 0.08);
        let bend = match self.role {
            InstrumentRole::Bass => PitchBendCurve::swoop(-24.0, 8.0),
            InstrumentRole::Lead => PitchBendCurve::swoop(-35.0, 42.0),
            InstrumentRole::Pad => PitchBendCurve::swoop(-8.0, 8.0),
            InstrumentRole::Percussion => PitchBendCurve::swoop(90.0, -120.0),
            InstrumentRole::Texture => PitchBendCurve::swoop(55.0, -35.0),
        };
        let note = NoteEvent::new(0.02, duration, degree, octave)
            .with_velocity(1.0)
            .with_bend(bend);
        self.render_note(&mut buffer, scale, &note, &mut rng);
        buffer
    }

    pub fn uses_forbidden_acoustic_label(&self) -> bool {
        let text = format!("{} {}", self.id, self.display_name).to_ascii_lowercase();
        [
            "harp",
            "accordion",
            "harmonica",
            "guitar",
            "piano",
            "violin",
            "cello",
            "trumpet",
            "sax",
            "flute",
            "clarinet",
            "drum kit",
        ]
        .iter()
        .any(|word| text.contains(word))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct InstrumentPalette {
    pub instruments: Vec<SyntheticInstrument>,
    pub methods_used: Vec<InstrumentDiscoveryMethod>,
}

impl InstrumentPalette {
    pub fn for_role(&self, role: InstrumentRole) -> Option<&SyntheticInstrument> {
        self.instruments
            .iter()
            .find(|instrument| instrument.role == role)
    }

    pub fn names(&self) -> Vec<String> {
        self.instruments
            .iter()
            .map(|instrument| instrument.display_name.clone())
            .collect()
    }

    pub fn validate_anti_mimicry(&self) -> Result<(), String> {
        for instrument in &self.instruments {
            if instrument.uses_forbidden_acoustic_label() {
                return Err(format!(
                    "instrument {} uses an acoustic-emulation label",
                    instrument.display_name
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
struct CandidateScore {
    instrument: SyntheticInstrument,
    score: f64,
    role_fit: f64,
    novelty: f64,
    anti_mimicry: f64,
    spectral_centroid_hz: f64,
    spectral_flatness: f64,
}

#[derive(Clone, Debug, PartialEq)]
struct CandidateMetrics {
    score: f64,
    role_fit: f64,
    novelty: f64,
    anti_mimicry: f64,
    spectral_centroid_hz: f64,
    spectral_flatness: f64,
}

pub fn discover_instrument_palette(spec: InstrumentDiscoverySpec) -> InstrumentPalette {
    assert!(
        spec.candidates_per_role > 0,
        "candidates_per_role must be positive"
    );
    let mut all_scores: Vec<CandidateScore> = Vec::new();
    for &role in &spec.roles {
        let mut rng = mulberry32(spec.seed ^ role_seed(role));
        for index in 0..spec.candidates_per_role {
            let mut candidate = candidate_instrument(role, index, &mut rng);
            let score = score_candidate(
                &candidate,
                spec.sample_rate,
                spec.seed ^ ((index as u32 + 1) * 0x45d9_f3b),
                &spec.objective,
            );
            candidate.discovery = InstrumentDiscoveryTrace {
                method: first_method(&spec.methods),
                score: score.score,
                role_fit: score.role_fit,
                novelty: score.novelty,
                anti_mimicry: score.anti_mimicry,
                spectral_centroid_hz: score.spectral_centroid_hz,
                spectral_flatness: score.spectral_flatness,
                notes: vec![
                    "invented synthesis recipe".to_string(),
                    "anti-mimicry acoustic-label guard applied".to_string(),
                ],
            };
            all_scores.push(CandidateScore {
                instrument: candidate,
                score: score.score,
                role_fit: score.role_fit,
                novelty: score.novelty,
                anti_mimicry: score.anti_mimicry,
                spectral_centroid_hz: score.spectral_centroid_hz,
                spectral_flatness: score.spectral_flatness,
            });
        }
    }

    let mut selected = if spec
        .methods
        .contains(&InstrumentDiscoveryMethod::IntegerProgramSelection)
    {
        select_integer_program_style(&spec.roles, &all_scores)
    } else {
        select_best_per_role(&spec.roles, &all_scores)
    };

    if spec
        .methods
        .contains(&InstrumentDiscoveryMethod::PomdpBeliefSearch)
    {
        for instrument in &mut selected {
            instrument.discovery.method = InstrumentDiscoveryMethod::PomdpBeliefSearch;
            instrument
                .discovery
                .notes
                .push("belief robustness favored stable high-novelty candidates".to_string());
        }
    }
    if spec
        .methods
        .contains(&InstrumentDiscoveryMethod::IntegerProgramSelection)
    {
        for instrument in &mut selected {
            instrument.discovery.method = InstrumentDiscoveryMethod::IntegerProgramSelection;
            instrument
                .discovery
                .notes
                .push("integer selection enforced role coverage and contrast".to_string());
        }
    }

    let palette = InstrumentPalette {
        instruments: selected,
        methods_used: spec.methods,
    };
    palette
        .validate_anti_mimicry()
        .expect("discovered palette should avoid acoustic-emulation labels");
    palette
}

fn role_seed(role: InstrumentRole) -> u32 {
    match role {
        InstrumentRole::Bass => 0xb455,
        InstrumentRole::Lead => 0x1ead,
        InstrumentRole::Pad => 0x9ad0,
        InstrumentRole::Percussion => 0x9e2c,
        InstrumentRole::Texture => 0x7e17,
    }
}

fn first_method(methods: &[InstrumentDiscoveryMethod]) -> InstrumentDiscoveryMethod {
    methods
        .first()
        .copied()
        .unwrap_or(InstrumentDiscoveryMethod::MdpPolicySearch)
}

fn wavefold(sample: f64, fold: f64) -> f64 {
    if fold <= 0.0 {
        sample
    } else {
        (sample * (1.0 + fold * 3.0)).sin() / (1.0 + fold * 0.35)
    }
}

fn candidate_instrument(
    role: InstrumentRole,
    index: usize,
    rng: &mut impl RandomSource,
) -> SyntheticInstrument {
    let names_a = [
        "Ion", "Lumen", "Phase", "Cinder", "Glyph", "Nova", "Prism", "Torsion",
    ];
    let names_b = [
        "Fold", "Bloom", "Index", "Array", "Drift", "Circuit", "Halo", "Vector",
    ];
    let name = format!(
        "{} {} {}",
        names_a[(index + role as usize) % names_a.len()],
        names_b[(index * 3 + role as usize) % names_b.len()],
        role.as_str()
    );
    let layer_count = match role {
        InstrumentRole::Bass | InstrumentRole::Percussion => 2,
        InstrumentRole::Pad | InstrumentRole::Texture => 3,
        InstrumentRole::Lead => 2 + rng.next_int(0, 2) as usize,
    };
    let mut layers = Vec::new();
    for layer_index in 0..layer_count {
        let waveform = match rng.next_int(0, 6) {
            0 => Waveform::Sine,
            1 => Waveform::Triangle,
            2 => Waveform::Saw,
            3 => Waveform::Square,
            4 => Waveform::Pulse(0.18 + rng.next_float() * 0.62),
            _ => Waveform::Noise,
        };
        let harmonic_ratio = match role {
            InstrumentRole::Bass => [0.5, 1.0, 1.5, 2.0][layer_index.min(3)],
            InstrumentRole::Lead => 1.0 + layer_index as f64 * (0.33 + 0.2 * rng.next_float()),
            InstrumentRole::Pad => 0.5 + layer_index as f64 * 0.5 + rng.next_float() * 0.12,
            InstrumentRole::Percussion => 1.0 + layer_index as f64 * (1.5 + rng.next_float()),
            InstrumentRole::Texture => 0.75 + layer_index as f64 * (0.71 + 0.28 * rng.next_float()),
        };
        let envelope = role_envelope(role, rng);
        let gain = match role {
            InstrumentRole::Bass => 0.24,
            InstrumentRole::Lead => 0.16,
            InstrumentRole::Pad => 0.07,
            InstrumentRole::Percussion => 0.21,
            InstrumentRole::Texture => 0.12,
        } * (0.65 + rng.next_float() * 0.7)
            / layer_count as f64;
        let cutoff = match role {
            InstrumentRole::Bass => 700.0 + rng.next_float() * 1_500.0,
            InstrumentRole::Lead => 1_800.0 + rng.next_float() * 6_000.0,
            InstrumentRole::Pad => 900.0 + rng.next_float() * 3_200.0,
            InstrumentRole::Percussion => 3_000.0 + rng.next_float() * 8_000.0,
            InstrumentRole::Texture => 1_200.0 + rng.next_float() * 8_800.0,
        };
        layers.push(SyntheticLayer {
            waveform,
            gain,
            harmonic_ratio,
            detune_cents: (rng.next_float() - 0.5) * 42.0,
            fm_ratio: 0.25 + rng.next_float() * 4.5,
            fm_index: rng.next_float() * role_fm_depth(role),
            fold: rng.next_float() * role_fold_depth(role),
            noise_level: rng.next_float() * role_noise_depth(role),
            filter_cutoff_hz: cutoff,
            filter_q: 0.45 + rng.next_float() * 1.6,
            envelope,
        });
    }

    SyntheticInstrument {
        id: format!("invented-{}-{index}", role.as_str()),
        display_name: name,
        role,
        layers,
        transient_noise: if role == InstrumentRole::Percussion {
            0.12 + rng.next_float() * 0.22
        } else {
            rng.next_float() * 0.035
        },
        output_gain: 0.75 + rng.next_float() * 0.55,
        discovery: InstrumentDiscoveryTrace {
            method: InstrumentDiscoveryMethod::MdpPolicySearch,
            score: 0.0,
            role_fit: 0.0,
            novelty: 0.0,
            anti_mimicry: 1.0,
            spectral_centroid_hz: 0.0,
            spectral_flatness: 0.0,
            notes: Vec::new(),
        },
    }
}

fn role_envelope(role: InstrumentRole, rng: &mut impl RandomSource) -> AdsrEnvelope {
    match role {
        InstrumentRole::Bass => AdsrEnvelope {
            attack: 0.003 + rng.next_float() * 0.012,
            decay: 0.08 + rng.next_float() * 0.18,
            sustain: 0.32 + rng.next_float() * 0.3,
            release: 0.04 + rng.next_float() * 0.12,
        },
        InstrumentRole::Lead => AdsrEnvelope {
            attack: 0.004 + rng.next_float() * 0.03,
            decay: 0.06 + rng.next_float() * 0.16,
            sustain: 0.18 + rng.next_float() * 0.34,
            release: 0.07 + rng.next_float() * 0.22,
        },
        InstrumentRole::Pad => AdsrEnvelope {
            attack: 0.05 + rng.next_float() * 0.28,
            decay: 0.22 + rng.next_float() * 0.5,
            sustain: 0.42 + rng.next_float() * 0.35,
            release: 0.35 + rng.next_float() * 0.9,
        },
        InstrumentRole::Percussion => AdsrEnvelope {
            attack: 0.001 + rng.next_float() * 0.006,
            decay: 0.035 + rng.next_float() * 0.16,
            sustain: rng.next_float() * 0.22,
            release: 0.015 + rng.next_float() * 0.08,
        },
        InstrumentRole::Texture => AdsrEnvelope {
            attack: 0.008 + rng.next_float() * 0.1,
            decay: 0.08 + rng.next_float() * 0.26,
            sustain: 0.12 + rng.next_float() * 0.52,
            release: 0.12 + rng.next_float() * 0.45,
        },
    }
}

fn role_fm_depth(role: InstrumentRole) -> f64 {
    match role {
        InstrumentRole::Bass => 0.8,
        InstrumentRole::Lead => 2.8,
        InstrumentRole::Pad => 1.2,
        InstrumentRole::Percussion => 4.0,
        InstrumentRole::Texture => 3.4,
    }
}

fn role_fold_depth(role: InstrumentRole) -> f64 {
    match role {
        InstrumentRole::Bass => 0.7,
        InstrumentRole::Lead => 1.5,
        InstrumentRole::Pad => 0.45,
        InstrumentRole::Percussion => 1.8,
        InstrumentRole::Texture => 2.2,
    }
}

fn role_noise_depth(role: InstrumentRole) -> f64 {
    match role {
        InstrumentRole::Bass => 0.04,
        InstrumentRole::Lead => 0.08,
        InstrumentRole::Pad => 0.025,
        InstrumentRole::Percussion => 0.35,
        InstrumentRole::Texture => 0.22,
    }
}

fn score_candidate(
    instrument: &SyntheticInstrument,
    sample_rate: u32,
    seed: u32,
    objective: &DiscoveryObjective,
) -> CandidateMetrics {
    let scale = MicrotonalScale::rave_collage_19_edo();
    let probe = instrument.render_probe(sample_rate, &scale, seed);
    let fft_len = 1024usize.min(probe.samples.len().next_power_of_two());
    let bins = if fft_len.is_power_of_two() && fft_len <= probe.samples.len() {
        analyze_fft(
            &probe.samples[..fft_len],
            probe.sample_rate,
            WindowFunction::Hann,
        )
    } else {
        Vec::new()
    };
    let centroid = spectral_centroid(&bins);
    let flatness = spectral_flatness(&bins);
    let target = instrument.role.target_centroid_hz();
    let role_fit = if centroid > 0.0 {
        let octave_error = (centroid / target).log2().abs();
        (1.0 - octave_error / 4.0).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let layer_variety = instrument.layers.len() as f64 / 4.0;
    let fm = instrument.layers.iter().map(|l| l.fm_index).sum::<f64>()
        / instrument.layers.len().max(1) as f64;
    let fold = instrument.layers.iter().map(|l| l.fold).sum::<f64>()
        / instrument.layers.len().max(1) as f64;
    let novelty =
        (0.22 + flatness * 0.38 + layer_variety * 0.18 + (fm + fold) * 0.08).clamp(0.0, 1.0);
    let anti_mimicry = if instrument.uses_forbidden_acoustic_label() {
        0.0
    } else {
        1.0
    };
    let spectral_spread = (flatness * 1.8).clamp(0.0, 1.0);
    let score = objective.role_fit_weight * role_fit
        + objective.novelty_weight * novelty
        + objective.spectral_spread_weight * spectral_spread
        + objective.anti_mimicry_weight * anti_mimicry;
    CandidateMetrics {
        score,
        role_fit,
        novelty,
        anti_mimicry,
        spectral_centroid_hz: centroid,
        spectral_flatness: flatness,
    }
}

fn spectral_flatness(bins: &[SpectrumBin]) -> f64 {
    let mags: Vec<f64> = bins
        .iter()
        .skip(1)
        .filter_map(|bin| {
            if bin.magnitude.is_finite() && bin.magnitude > 1e-12 {
                Some(bin.magnitude)
            } else {
                None
            }
        })
        .collect();
    if mags.is_empty() {
        return 0.0;
    }
    let geo = (mags.iter().map(|m| m.ln()).sum::<f64>() / mags.len() as f64).exp();
    let arith = mags.iter().sum::<f64>() / mags.len() as f64;
    if arith <= f64::EPSILON {
        0.0
    } else {
        (geo / arith).clamp(0.0, 1.0)
    }
}

fn select_best_per_role(
    roles: &[InstrumentRole],
    scores: &[CandidateScore],
) -> Vec<SyntheticInstrument> {
    roles
        .iter()
        .filter_map(|role| {
            scores
                .iter()
                .filter(|score| score.instrument.role == *role)
                .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
                .map(|score| score.instrument.clone())
        })
        .collect()
}

fn select_integer_program_style(
    roles: &[InstrumentRole],
    scores: &[CandidateScore],
) -> Vec<SyntheticInstrument> {
    let mut selected_scores: Vec<&CandidateScore> = Vec::new();
    for role in roles {
        let best = scores
            .iter()
            .filter(|score| score.instrument.role == *role)
            .max_by(|a, b| {
                let a_total = a.score + diversity_bonus(a, &selected_scores);
                let b_total = b.score + diversity_bonus(b, &selected_scores);
                a_total.partial_cmp(&b_total).unwrap()
            });
        if let Some(best) = best {
            selected_scores.push(best);
        }
    }
    selected_scores
        .into_iter()
        .map(|score| score.instrument.clone())
        .collect()
}

fn diversity_bonus(candidate: &CandidateScore, selected: &[&CandidateScore]) -> f64 {
    if selected.is_empty() {
        return 0.0;
    }
    let mut min_distance = f64::INFINITY;
    for other in selected {
        let a = candidate.spectral_centroid_hz.max(1.0).log2();
        let b = other.spectral_centroid_hz.max(1.0).log2();
        min_distance = min_distance.min((a - b).abs());
    }
    (min_distance / 3.0).clamp(0.0, 0.2)
}

fn oscillator_sample(waveform: Waveform, phase: f64, rng: &mut impl RandomSource) -> f64 {
    let cycle = (phase / TAU).fract();
    match waveform {
        Waveform::Sine => phase.sin(),
        Waveform::Triangle => 1.0 - 4.0 * (cycle - 0.5).abs(),
        Waveform::Saw => 2.0 * cycle - 1.0,
        Waveform::Square => {
            if cycle < 0.5 {
                1.0
            } else {
                -1.0
            }
        }
        Waveform::Pulse(width) => {
            if cycle < width.clamp(0.02, 0.98) {
                1.0
            } else {
                -1.0
            }
        }
        Waveform::Noise => rng.next_float() * 2.0 - 1.0,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterMode {
    LowPass,
    HighPass,
}

/// RBJ-cookbook biquad filter in direct-form-II transposed form.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiquadFilter {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl BiquadFilter {
    pub fn new(mode: FilterMode, sample_rate: u32, cutoff_hz: f64, q: f64) -> Self {
        require_finite("cutoff_hz", cutoff_hz);
        require_finite("q", q);
        assert!(cutoff_hz > 0.0, "cutoff_hz must be positive");
        assert!(q > 0.0, "q must be positive");
        let nyquist = sample_rate as f64 * 0.5;
        let omega = TAU * cutoff_hz.min(nyquist * 0.99) / sample_rate as f64;
        let cos = omega.cos();
        let sin = omega.sin();
        let alpha = sin / (2.0 * q);
        let (b0, b1, b2) = match mode {
            FilterMode::LowPass => ((1.0 - cos) * 0.5, 1.0 - cos, (1.0 - cos) * 0.5),
            FilterMode::HighPass => ((1.0 + cos) * 0.5, -(1.0 + cos), (1.0 + cos) * 0.5),
        };
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos;
        let a2 = 1.0 - alpha;
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    pub fn process_sample(&mut self, input: f64) -> f64 {
        let output = self.b0 * input + self.z1;
        self.z1 = self.b1 * input - self.a1 * output + self.z2;
        self.z2 = self.b2 * input - self.a2 * output;
        output
    }
}

pub fn apply_filter(buffer: &mut AudioBuffer, mut filter: BiquadFilter) {
    for sample in &mut buffer.samples {
        *sample = filter.process_sample(*sample);
    }
}

pub fn apply_soft_clip(buffer: &mut AudioBuffer, drive: f64, mix: f64) {
    require_finite("drive", drive);
    assert!(drive > 0.0, "drive must be positive");
    let mix = mix.clamp(0.0, 1.0);
    for sample in &mut buffer.samples {
        let wet = (*sample * drive).tanh();
        *sample = *sample * (1.0 - mix) + wet * mix;
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BitCrusherEffect {
    pub bits: u8,
    pub hold_samples: usize,
    pub mix: f64,
}

pub fn apply_bitcrusher(buffer: &mut AudioBuffer, effect: BitCrusherEffect) {
    assert!((2..=16).contains(&effect.bits), "bits must be in 2..=16");
    let hold_samples = effect.hold_samples.max(1);
    let mix = effect.mix.clamp(0.0, 1.0);
    let levels = (1u32 << effect.bits) as f64 - 1.0;
    let mut held = 0.0;
    for (i, sample) in buffer.samples.iter_mut().enumerate() {
        if i % hold_samples == 0 {
            held = *sample;
        }
        let normalized = ((held * 0.5 + 0.5).clamp(0.0, 1.0) * levels).round() / levels;
        let wet = normalized * 2.0 - 1.0;
        *sample = *sample * (1.0 - mix) + wet * mix;
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DelayEffect {
    pub delay_seconds: f64,
    pub feedback: f64,
    pub mix: f64,
}

pub fn apply_feedback_delay(buffer: &mut AudioBuffer, effect: DelayEffect) {
    require_finite("delay_seconds", effect.delay_seconds);
    assert!(effect.delay_seconds > 0.0, "delay_seconds must be positive");
    let delay_samples = (effect.delay_seconds * buffer.sample_rate as f64)
        .round()
        .max(1.0) as usize;
    let feedback = effect.feedback.clamp(0.0, 0.98);
    let mix = effect.mix.clamp(0.0, 1.0);
    let mut line = vec![0.0; delay_samples];
    let mut pos = 0usize;
    for sample in &mut buffer.samples {
        let delayed = line[pos];
        let input = *sample;
        line[pos] = input + delayed * feedback;
        *sample = input * (1.0 - mix) + delayed * mix;
        pos = (pos + 1) % delay_samples;
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReverbEffect {
    pub room_size: f64,
    pub damping: f64,
    pub mix: f64,
}

pub fn apply_schroeder_reverb(buffer: &mut AudioBuffer, effect: ReverbEffect) {
    let room_size = effect.room_size.clamp(0.05, 0.95);
    let damping = effect.damping.clamp(0.0, 0.95);
    let mix = effect.mix.clamp(0.0, 1.0);
    let delays = [0.0297, 0.0371, 0.0411, 0.0437];
    let mut wet = vec![0.0; buffer.len()];
    for delay in delays {
        let delay_samples = (delay * buffer.sample_rate as f64).round().max(1.0) as usize;
        let mut line = vec![0.0; delay_samples];
        let mut pos = 0usize;
        let mut damped = 0.0;
        for (i, &input) in buffer.samples.iter().enumerate() {
            let delayed = line[pos];
            damped = delayed * (1.0 - damping) + damped * damping;
            line[pos] = input + damped * room_size;
            wet[i] += delayed;
            pos = (pos + 1) % delay_samples;
        }
    }
    for (sample, wet_sample) in buffer.samples.iter_mut().zip(wet) {
        *sample = *sample * (1.0 - mix) + (wet_sample * 0.25) * mix;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowFunction {
    Rectangular,
    Hann,
    Hamming,
}

pub fn windowed_samples(samples: &[f64], window: WindowFunction) -> Vec<f64> {
    let n = samples.len();
    if n <= 1 {
        return samples.to_vec();
    }
    samples
        .iter()
        .enumerate()
        .map(|(i, &sample)| {
            let u = i as f64 / (n - 1) as f64;
            let weight = match window {
                WindowFunction::Rectangular => 1.0,
                WindowFunction::Hann => 0.5 - 0.5 * (TAU * u).cos(),
                WindowFunction::Hamming => 0.54 - 0.46 * (TAU * u).cos(),
            };
            sample * weight
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpectrumBin {
    pub bin: usize,
    pub frequency_hz: f64,
    pub magnitude: f64,
    pub phase: f64,
}

pub fn analyze_fft(samples: &[f64], sample_rate: u32, window: WindowFunction) -> Vec<SpectrumBin> {
    assert!(!samples.is_empty(), "samples must not be empty");
    assert!(
        samples.len().is_power_of_two(),
        "FFT analysis requires a power-of-two sample count"
    );
    assert!(
        samples.iter().all(|s| s.is_finite()),
        "samples must all be finite"
    );
    let windowed = windowed_samples(samples, window);
    let result = run_fft_transform(FastFourierTransformParams {
        sequence: Some(windowed),
        ..Default::default()
    });
    let n = samples.len();
    let half = n / 2;
    result
        .outputs
        .into_iter()
        .take(half + 1)
        .enumerate()
        .map(|(bin, output)| {
            let edge_bin = bin == 0 || bin == half;
            let magnitude_scale = if edge_bin { 1.0 } else { 2.0 };
            SpectrumBin {
                bin,
                frequency_hz: bin as f64 * sample_rate as f64 / n as f64,
                magnitude: output.magnitude * magnitude_scale / n as f64,
                phase: output.phase,
            }
        })
        .collect()
}

pub fn spectral_centroid(bins: &[SpectrumBin]) -> f64 {
    let total: f64 = bins.iter().map(|bin| bin.magnitude).sum();
    if total <= f64::EPSILON {
        return 0.0;
    }
    bins.iter()
        .map(|bin| bin.frequency_hz * bin.magnitude)
        .sum::<f64>()
        / total
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SampleLicense {
    GeneratedByThisEngine,
    PublicDomain,
    CreativeCommonsZero,
    CreativeCommonsAttribution {
        attribution: String,
        source_url: String,
    },
    OpenLicensed {
        license_name: String,
        source_url: String,
    },
    Unknown,
}

impl SampleLicense {
    pub fn is_open_or_generated(&self) -> bool {
        !matches!(self, SampleLicense::Unknown)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SampleSource {
    pub id: String,
    pub path: Option<String>,
    pub license: SampleLicense,
    pub tags: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SampleManifest {
    pub sources: Vec<SampleSource>,
}

impl SampleManifest {
    pub fn validate_legal_sources(&self) -> Result<(), String> {
        for source in &self.sources {
            if !source.license.is_open_or_generated() {
                return Err(format!(
                    "sample source {} does not declare an open/generated license",
                    source.id
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MusicStudioBusKind {
    Midi,
    Audio,
    Control,
    Analysis,
}

impl MusicStudioBusKind {
    pub fn as_str(self) -> &'static str {
        match self {
            MusicStudioBusKind::Midi => "midi",
            MusicStudioBusKind::Audio => "audio",
            MusicStudioBusKind::Control => "control",
            MusicStudioBusKind::Analysis => "analysis",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MusicStudioLineage {
    MidiSequencing,
    RecordingStudios,
    ElectronicMusic,
    WindowsLoops,
    BudgetDaw,
    MacComposers,
    ModularElectronic,
    TrackerScene,
}

impl MusicStudioLineage {
    pub fn as_str(self) -> &'static str {
        match self {
            MusicStudioLineage::MidiSequencing => "midi-sequencing",
            MusicStudioLineage::RecordingStudios => "recording-studios",
            MusicStudioLineage::ElectronicMusic => "electronic-music",
            MusicStudioLineage::WindowsLoops => "windows-loops",
            MusicStudioLineage::BudgetDaw => "budget-daw",
            MusicStudioLineage::MacComposers => "mac-composers",
            MusicStudioLineage::ModularElectronic => "modular-electronic",
            MusicStudioLineage::TrackerScene => "tracker-scene",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MusicStudioReferenceApp {
    pub lineage: MusicStudioLineage,
    pub software: &'static str,
}

pub const MUSIC_STUDIO_REFERENCE_APPS: [MusicStudioReferenceApp; 11] = [
    MusicStudioReferenceApp {
        lineage: MusicStudioLineage::MidiSequencing,
        software: "Cubase",
    },
    MusicStudioReferenceApp {
        lineage: MusicStudioLineage::MidiSequencing,
        software: "Logic",
    },
    MusicStudioReferenceApp {
        lineage: MusicStudioLineage::RecordingStudios,
        software: "Pro Tools",
    },
    MusicStudioReferenceApp {
        lineage: MusicStudioLineage::ElectronicMusic,
        software: "FL Studio",
    },
    MusicStudioReferenceApp {
        lineage: MusicStudioLineage::ElectronicMusic,
        software: "Ableton Live",
    },
    MusicStudioReferenceApp {
        lineage: MusicStudioLineage::WindowsLoops,
        software: "ACID Pro",
    },
    MusicStudioReferenceApp {
        lineage: MusicStudioLineage::BudgetDaw,
        software: "Cakewalk",
    },
    MusicStudioReferenceApp {
        lineage: MusicStudioLineage::MacComposers,
        software: "Digital Performer",
    },
    MusicStudioReferenceApp {
        lineage: MusicStudioLineage::ModularElectronic,
        software: "Reason",
    },
    MusicStudioReferenceApp {
        lineage: MusicStudioLineage::ModularElectronic,
        software: "Orion",
    },
    MusicStudioReferenceApp {
        lineage: MusicStudioLineage::TrackerScene,
        software: "Buzz",
    },
];

pub fn music_studio_reference_apps() -> &'static [MusicStudioReferenceApp] {
    &MUSIC_STUDIO_REFERENCE_APPS
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MusicStudioBlockKind {
    ClockTransport,
    ArrangementLane,
    MidiSequencer,
    GrooveQuantizer,
    ChordHarmony,
    AutomationLane,
    PerformanceController,
    SynthVoice,
    DrumMachine,
    Sampler,
    LoopPlayer,
    AudioTrack,
    MixerChannel,
    Equalizer,
    Compressor,
    Distortion,
    DelayEcho,
    Reverb,
    SpectrumAnalyzer,
    MasterOutput,
}

impl MusicStudioBlockKind {
    pub fn as_str(self) -> &'static str {
        match self {
            MusicStudioBlockKind::ClockTransport => "clock-transport",
            MusicStudioBlockKind::ArrangementLane => "arrangement-lane",
            MusicStudioBlockKind::MidiSequencer => "midi-sequencer",
            MusicStudioBlockKind::GrooveQuantizer => "groove-quantizer",
            MusicStudioBlockKind::ChordHarmony => "chord-harmony",
            MusicStudioBlockKind::AutomationLane => "automation-lane",
            MusicStudioBlockKind::PerformanceController => "performance-controller",
            MusicStudioBlockKind::SynthVoice => "synth-voice",
            MusicStudioBlockKind::DrumMachine => "drum-machine",
            MusicStudioBlockKind::Sampler => "sampler",
            MusicStudioBlockKind::LoopPlayer => "loop-player",
            MusicStudioBlockKind::AudioTrack => "audio-track",
            MusicStudioBlockKind::MixerChannel => "mixer-channel",
            MusicStudioBlockKind::Equalizer => "equalizer",
            MusicStudioBlockKind::Compressor => "compressor",
            MusicStudioBlockKind::Distortion => "distortion",
            MusicStudioBlockKind::DelayEcho => "delay-echo",
            MusicStudioBlockKind::Reverb => "reverb",
            MusicStudioBlockKind::SpectrumAnalyzer => "spectrum-analyzer",
            MusicStudioBlockKind::MasterOutput => "master-output",
        }
    }

    pub fn default_label(self) -> &'static str {
        match self {
            MusicStudioBlockKind::ClockTransport => "Transport / Clock",
            MusicStudioBlockKind::ArrangementLane => "Arrangement Lane",
            MusicStudioBlockKind::MidiSequencer => "MIDI Sequencer",
            MusicStudioBlockKind::GrooveQuantizer => "Groove Quantizer",
            MusicStudioBlockKind::ChordHarmony => "Chord + Counterpoint",
            MusicStudioBlockKind::AutomationLane => "Automation Lane",
            MusicStudioBlockKind::PerformanceController => "Performance Controls",
            MusicStudioBlockKind::SynthVoice => "Synth Voice",
            MusicStudioBlockKind::DrumMachine => "Drum Machine",
            MusicStudioBlockKind::Sampler => "Sampler",
            MusicStudioBlockKind::LoopPlayer => "Loop Player",
            MusicStudioBlockKind::AudioTrack => "Audio Recording Track",
            MusicStudioBlockKind::MixerChannel => "Mixer Channel",
            MusicStudioBlockKind::Equalizer => "Equalizer",
            MusicStudioBlockKind::Compressor => "Compressor",
            MusicStudioBlockKind::Distortion => "Distortion",
            MusicStudioBlockKind::DelayEcho => "Delay / Echo",
            MusicStudioBlockKind::Reverb => "Reverb",
            MusicStudioBlockKind::SpectrumAnalyzer => "Spectrum Analyzer",
            MusicStudioBlockKind::MasterOutput => "Master Output",
        }
    }

    pub fn visual_role(self) -> VisualBlockRole {
        match self {
            MusicStudioBlockKind::ClockTransport
            | MusicStudioBlockKind::ArrangementLane
            | MusicStudioBlockKind::AutomationLane
            | MusicStudioBlockKind::PerformanceController
            | MusicStudioBlockKind::LoopPlayer
            | MusicStudioBlockKind::AudioTrack => VisualBlockRole::Source,
            MusicStudioBlockKind::MasterOutput => VisualBlockRole::Sink,
            MusicStudioBlockKind::MixerChannel => VisualBlockRole::Station,
            MusicStudioBlockKind::SpectrumAnalyzer => VisualBlockRole::Observer,
            _ => VisualBlockRole::Transform,
        }
    }

    pub fn input_ports(self) -> Vec<MusicStudioPortTemplate> {
        match self {
            MusicStudioBlockKind::ClockTransport
            | MusicStudioBlockKind::ArrangementLane
            | MusicStudioBlockKind::AutomationLane
            | MusicStudioBlockKind::PerformanceController
            | MusicStudioBlockKind::LoopPlayer
            | MusicStudioBlockKind::AudioTrack => Vec::new(),
            MusicStudioBlockKind::MidiSequencer => vec![
                music_port("clock", "clock", MusicStudioBusKind::Control, true),
                music_port(
                    "arrangement",
                    "arrangement",
                    MusicStudioBusKind::Control,
                    false,
                ),
            ],
            MusicStudioBlockKind::GrooveQuantizer | MusicStudioBlockKind::ChordHarmony => {
                vec![music_port(
                    "midi_in",
                    "MIDI in",
                    MusicStudioBusKind::Midi,
                    true,
                )]
            }
            MusicStudioBlockKind::SynthVoice => vec![
                music_port("midi_in", "MIDI in", MusicStudioBusKind::Midi, true),
                music_port(
                    "modulation",
                    "modulation",
                    MusicStudioBusKind::Control,
                    false,
                ),
            ],
            MusicStudioBlockKind::DrumMachine | MusicStudioBlockKind::Sampler => {
                vec![music_port(
                    "midi_in",
                    "MIDI in",
                    MusicStudioBusKind::Midi,
                    true,
                )]
            }
            MusicStudioBlockKind::MixerChannel => vec![
                music_port("audio_in", "audio in", MusicStudioBusKind::Audio, true),
                music_port("control", "control", MusicStudioBusKind::Control, false),
            ],
            MusicStudioBlockKind::Equalizer
            | MusicStudioBlockKind::Compressor
            | MusicStudioBlockKind::Distortion
            | MusicStudioBlockKind::DelayEcho
            | MusicStudioBlockKind::Reverb
            | MusicStudioBlockKind::SpectrumAnalyzer
            | MusicStudioBlockKind::MasterOutput => {
                vec![music_port(
                    "audio_in",
                    "audio in",
                    MusicStudioBusKind::Audio,
                    true,
                )]
            }
        }
    }

    pub fn output_ports(self) -> Vec<MusicStudioPortTemplate> {
        match self {
            MusicStudioBlockKind::ClockTransport => {
                vec![music_port(
                    "clock",
                    "clock",
                    MusicStudioBusKind::Control,
                    true,
                )]
            }
            MusicStudioBlockKind::ArrangementLane => vec![music_port(
                "arrangement",
                "arrangement",
                MusicStudioBusKind::Control,
                true,
            )],
            MusicStudioBlockKind::AutomationLane => vec![music_port(
                "automation",
                "automation",
                MusicStudioBusKind::Control,
                true,
            )],
            MusicStudioBlockKind::PerformanceController => vec![music_port(
                "performance",
                "performance",
                MusicStudioBusKind::Control,
                true,
            )],
            MusicStudioBlockKind::MidiSequencer
            | MusicStudioBlockKind::GrooveQuantizer
            | MusicStudioBlockKind::ChordHarmony => {
                vec![music_port(
                    "midi_out",
                    "MIDI out",
                    MusicStudioBusKind::Midi,
                    true,
                )]
            }
            MusicStudioBlockKind::SynthVoice
            | MusicStudioBlockKind::DrumMachine
            | MusicStudioBlockKind::Sampler
            | MusicStudioBlockKind::LoopPlayer
            | MusicStudioBlockKind::AudioTrack
            | MusicStudioBlockKind::MixerChannel
            | MusicStudioBlockKind::Equalizer
            | MusicStudioBlockKind::Compressor
            | MusicStudioBlockKind::Distortion
            | MusicStudioBlockKind::DelayEcho
            | MusicStudioBlockKind::Reverb => {
                vec![music_port(
                    "audio_out",
                    "audio out",
                    MusicStudioBusKind::Audio,
                    true,
                )]
            }
            MusicStudioBlockKind::SpectrumAnalyzer => vec![music_port(
                "analysis",
                "analysis",
                MusicStudioBusKind::Analysis,
                false,
            )],
            MusicStudioBlockKind::MasterOutput => Vec::new(),
        }
    }

    pub fn default_feature_terms(self) -> Vec<&'static str> {
        match self {
            MusicStudioBlockKind::ClockTransport => vec![
                "ritmo",
                "tempo",
                "durata",
                "pausa",
                "battito",
                "misura",
                "metro",
                "andamento",
                "pulsazione",
            ],
            MusicStudioBlockKind::ArrangementLane => vec![
                "fraseggio",
                "arrangiamento",
                "orchestrazione",
                "dinamica",
                "intensità",
                "improvvisazione",
            ],
            MusicStudioBlockKind::MidiSequencer => vec![
                "melodia",
                "altezza",
                "scala",
                "tonalità",
                "modalità",
                "intonazione",
                "sequenza",
            ],
            MusicStudioBlockKind::GrooveQuantizer => {
                vec![
                    "sincope",
                    "accento",
                    "groove",
                    "articolazione",
                    "staccato",
                    "legato",
                ]
            }
            MusicStudioBlockKind::ChordHarmony => vec![
                "armonia",
                "accordo",
                "cadenza",
                "modulazione",
                "armonizzazione",
                "contrappunto",
                "polifonia",
                "monodia",
            ],
            MusicStudioBlockKind::AutomationLane => vec!["vibrato", "espressione"],
            MusicStudioBlockKind::PerformanceController => {
                vec!["interpretazione", "espressione", "improvvisazione"]
            }
            MusicStudioBlockKind::SynthVoice => vec!["timbro", "tessitura", "intonazione"],
            MusicStudioBlockKind::DrumMachine => vec!["ritmo", "battito", "accento", "groove"],
            MusicStudioBlockKind::Sampler => vec!["campionamento", "registrazione", "timbro"],
            MusicStudioBlockKind::LoopPlayer => vec!["loop", "sequenza", "campionamento"],
            MusicStudioBlockKind::AudioTrack => vec!["registrazione", "durata"],
            MusicStudioBlockKind::MixerChannel => vec!["dinamica", "intensità", "timbro"],
            MusicStudioBlockKind::Equalizer => vec!["equalizzazione", "spettro", "timbro"],
            MusicStudioBlockKind::Compressor => vec!["compressione", "dinamica", "intensità"],
            MusicStudioBlockKind::Distortion => vec!["distorsione", "timbro"],
            MusicStudioBlockKind::DelayEcho => vec!["eco", "durata"],
            MusicStudioBlockKind::Reverb => vec!["riverbero", "spettro"],
            MusicStudioBlockKind::SpectrumAnalyzer => vec!["spettro", "timbro"],
            MusicStudioBlockKind::MasterOutput => {
                vec!["equalizzazione", "compressione", "spettro"]
            }
        }
    }

    fn visual_style(self) -> VisualBlockStyle {
        match self {
            MusicStudioBlockKind::ClockTransport
            | MusicStudioBlockKind::ArrangementLane
            | MusicStudioBlockKind::MidiSequencer
            | MusicStudioBlockKind::GrooveQuantizer
            | MusicStudioBlockKind::ChordHarmony => music_visual_style("#edf7ed", "#187047"),
            MusicStudioBlockKind::SynthVoice
            | MusicStudioBlockKind::DrumMachine
            | MusicStudioBlockKind::Sampler
            | MusicStudioBlockKind::LoopPlayer
            | MusicStudioBlockKind::AudioTrack => music_visual_style("#f7f1e8", "#a15c10"),
            MusicStudioBlockKind::AutomationLane | MusicStudioBlockKind::PerformanceController => {
                music_visual_style("#eef4ff", "#265da8")
            }
            MusicStudioBlockKind::MixerChannel
            | MusicStudioBlockKind::Equalizer
            | MusicStudioBlockKind::Compressor
            | MusicStudioBlockKind::Distortion
            | MusicStudioBlockKind::DelayEcho
            | MusicStudioBlockKind::Reverb => music_visual_style("#f5f3ff", "#6846b7"),
            MusicStudioBlockKind::SpectrumAnalyzer | MusicStudioBlockKind::MasterOutput => {
                music_visual_style("#eef6f7", "#08747f")
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MusicStudioPortTemplate {
    pub id: &'static str,
    pub label: &'static str,
    pub bus: MusicStudioBusKind,
    pub required: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MusicStudioBoardBlock {
    pub id: String,
    pub kind: MusicStudioBlockKind,
    pub label: String,
    pub feature_terms: Vec<String>,
}

impl MusicStudioBoardBlock {
    pub fn new(id: impl Into<String>, kind: MusicStudioBlockKind) -> Self {
        Self {
            id: id.into(),
            kind,
            label: kind.default_label().to_string(),
            feature_terms: kind
                .default_feature_terms()
                .into_iter()
                .map(str::to_string)
                .collect(),
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    pub fn with_features(mut self, feature_terms: Vec<String>) -> Self {
        self.feature_terms = feature_terms;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MusicStudioBoardConnection {
    pub id: String,
    pub from_block: String,
    pub from_port: String,
    pub to_block: String,
    pub to_port: String,
    pub bus: MusicStudioBusKind,
}

impl MusicStudioBoardConnection {
    pub fn new(
        id: impl Into<String>,
        from_block: impl Into<String>,
        from_port: impl Into<String>,
        to_block: impl Into<String>,
        to_port: impl Into<String>,
        bus: MusicStudioBusKind,
    ) -> Self {
        Self {
            id: id.into(),
            from_block: from_block.into(),
            from_port: from_port.into(),
            to_block: to_block.into(),
            to_port: to_port.into(),
            bus,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MusicStudioBoardValidation {
    pub feature_coverage: MusicFeatureCoverage,
    pub source_blocks: Vec<String>,
    pub output_blocks: Vec<String>,
    pub connection_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MusicStudioBoard {
    pub id: String,
    pub label: String,
    pub reference_apps: Vec<MusicStudioReferenceApp>,
    pub blocks: Vec<MusicStudioBoardBlock>,
    pub connections: Vec<MusicStudioBoardConnection>,
}

impl MusicStudioBoard {
    pub fn reason_style_default() -> Self {
        let blocks = vec![
            MusicStudioBoardBlock::new("transport", MusicStudioBlockKind::ClockTransport),
            MusicStudioBoardBlock::new("arrangement", MusicStudioBlockKind::ArrangementLane),
            MusicStudioBoardBlock::new("sequencer", MusicStudioBlockKind::MidiSequencer),
            MusicStudioBoardBlock::new("groove", MusicStudioBlockKind::GrooveQuantizer),
            MusicStudioBoardBlock::new("harmony", MusicStudioBlockKind::ChordHarmony),
            MusicStudioBoardBlock::new("automation", MusicStudioBlockKind::AutomationLane),
            MusicStudioBoardBlock::new("performance", MusicStudioBlockKind::PerformanceController),
            MusicStudioBoardBlock::new("synth", MusicStudioBlockKind::SynthVoice),
            MusicStudioBoardBlock::new("drums", MusicStudioBlockKind::DrumMachine),
            MusicStudioBoardBlock::new("sampler", MusicStudioBlockKind::Sampler),
            MusicStudioBoardBlock::new("loops", MusicStudioBlockKind::LoopPlayer),
            MusicStudioBoardBlock::new("audio_track", MusicStudioBlockKind::AudioTrack),
            MusicStudioBoardBlock::new("mixer", MusicStudioBlockKind::MixerChannel),
            MusicStudioBoardBlock::new("eq", MusicStudioBlockKind::Equalizer),
            MusicStudioBoardBlock::new("compressor", MusicStudioBlockKind::Compressor),
            MusicStudioBoardBlock::new("distortion", MusicStudioBlockKind::Distortion),
            MusicStudioBoardBlock::new("delay", MusicStudioBlockKind::DelayEcho),
            MusicStudioBoardBlock::new("reverb", MusicStudioBlockKind::Reverb),
            MusicStudioBoardBlock::new("spectrum", MusicStudioBlockKind::SpectrumAnalyzer),
            MusicStudioBoardBlock::new("master", MusicStudioBlockKind::MasterOutput),
        ];
        let connections = vec![
            board_conn(
                "transport-clock",
                "transport",
                "clock",
                "sequencer",
                "clock",
                MusicStudioBusKind::Control,
            ),
            board_conn(
                "arrangement-to-sequencer",
                "arrangement",
                "arrangement",
                "sequencer",
                "arrangement",
                MusicStudioBusKind::Control,
            ),
            board_conn(
                "sequencer-to-groove",
                "sequencer",
                "midi_out",
                "groove",
                "midi_in",
                MusicStudioBusKind::Midi,
            ),
            board_conn(
                "groove-to-harmony",
                "groove",
                "midi_out",
                "harmony",
                "midi_in",
                MusicStudioBusKind::Midi,
            ),
            board_conn(
                "harmony-to-synth",
                "harmony",
                "midi_out",
                "synth",
                "midi_in",
                MusicStudioBusKind::Midi,
            ),
            board_conn(
                "groove-to-drums",
                "groove",
                "midi_out",
                "drums",
                "midi_in",
                MusicStudioBusKind::Midi,
            ),
            board_conn(
                "harmony-to-sampler",
                "harmony",
                "midi_out",
                "sampler",
                "midi_in",
                MusicStudioBusKind::Midi,
            ),
            board_conn(
                "automation-to-synth",
                "automation",
                "automation",
                "synth",
                "modulation",
                MusicStudioBusKind::Control,
            ),
            board_conn(
                "performance-to-mixer",
                "performance",
                "performance",
                "mixer",
                "control",
                MusicStudioBusKind::Control,
            ),
            board_conn(
                "synth-to-mixer",
                "synth",
                "audio_out",
                "mixer",
                "audio_in",
                MusicStudioBusKind::Audio,
            ),
            board_conn(
                "drums-to-mixer",
                "drums",
                "audio_out",
                "mixer",
                "audio_in",
                MusicStudioBusKind::Audio,
            ),
            board_conn(
                "sampler-to-mixer",
                "sampler",
                "audio_out",
                "mixer",
                "audio_in",
                MusicStudioBusKind::Audio,
            ),
            board_conn(
                "loops-to-mixer",
                "loops",
                "audio_out",
                "mixer",
                "audio_in",
                MusicStudioBusKind::Audio,
            ),
            board_conn(
                "audio-track-to-mixer",
                "audio_track",
                "audio_out",
                "mixer",
                "audio_in",
                MusicStudioBusKind::Audio,
            ),
            board_conn(
                "mixer-to-eq",
                "mixer",
                "audio_out",
                "eq",
                "audio_in",
                MusicStudioBusKind::Audio,
            ),
            board_conn(
                "eq-to-compressor",
                "eq",
                "audio_out",
                "compressor",
                "audio_in",
                MusicStudioBusKind::Audio,
            ),
            board_conn(
                "compressor-to-distortion",
                "compressor",
                "audio_out",
                "distortion",
                "audio_in",
                MusicStudioBusKind::Audio,
            ),
            board_conn(
                "distortion-to-delay",
                "distortion",
                "audio_out",
                "delay",
                "audio_in",
                MusicStudioBusKind::Audio,
            ),
            board_conn(
                "delay-to-reverb",
                "delay",
                "audio_out",
                "reverb",
                "audio_in",
                MusicStudioBusKind::Audio,
            ),
            board_conn(
                "reverb-to-master",
                "reverb",
                "audio_out",
                "master",
                "audio_in",
                MusicStudioBusKind::Audio,
            ),
            board_conn(
                "reverb-to-spectrum",
                "reverb",
                "audio_out",
                "spectrum",
                "audio_in",
                MusicStudioBusKind::Audio,
            ),
        ];
        Self {
            id: "reason-style-studio-board".to_string(),
            label: "Reason-style Visual Blocks Sound Board".to_string(),
            reference_apps: music_studio_reference_apps().to_vec(),
            blocks,
            connections,
        }
    }

    pub fn validate(&self) -> Result<MusicStudioBoardValidation, String> {
        if self.blocks.is_empty() {
            return Err("music studio board must include at least one block".to_string());
        }

        let mut block_ids = std::collections::HashSet::new();
        let mut block_index = std::collections::HashMap::new();
        for (index, block) in self.blocks.iter().enumerate() {
            if block.id.trim().is_empty() {
                return Err("music studio block ids must not be empty".to_string());
            }
            if !block_ids.insert(block.id.clone()) {
                return Err(format!("duplicate music studio block id {}", block.id));
            }
            block_index.insert(block.id.clone(), index);
        }

        let source_blocks: Vec<String> = self
            .blocks
            .iter()
            .filter(|block| block.kind.visual_role() == VisualBlockRole::Source)
            .map(|block| block.id.clone())
            .collect();
        let output_blocks: Vec<String> = self
            .blocks
            .iter()
            .filter(|block| block.kind == MusicStudioBlockKind::MasterOutput)
            .map(|block| block.id.clone())
            .collect();
        if source_blocks.is_empty() {
            return Err("music studio board needs at least one source block".to_string());
        }
        if output_blocks.is_empty() {
            return Err("music studio board needs a master output block".to_string());
        }
        if self.connections.is_empty() {
            return Err("music studio board needs at least one connection".to_string());
        }

        let mut connection_ids = std::collections::HashSet::new();
        let mut incoming = std::collections::HashSet::new();
        let mut outgoing = std::collections::HashSet::new();
        let mut adjacency: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        for connection in &self.connections {
            if connection.id.trim().is_empty() {
                return Err("music studio connection ids must not be empty".to_string());
            }
            if !connection_ids.insert(connection.id.clone()) {
                return Err(format!(
                    "duplicate music studio connection id {}",
                    connection.id
                ));
            }
            if connection.from_block == connection.to_block {
                return Err(format!(
                    "music studio connection {} loops back into block {}",
                    connection.id, connection.from_block
                ));
            }
            let from = self
                .blocks
                .get(*block_index.get(&connection.from_block).ok_or_else(|| {
                    format!(
                        "music studio connection {} references unknown source block {}",
                        connection.id, connection.from_block
                    )
                })?)
                .expect("block index");
            let to = self
                .blocks
                .get(*block_index.get(&connection.to_block).ok_or_else(|| {
                    format!(
                        "music studio connection {} references unknown target block {}",
                        connection.id, connection.to_block
                    )
                })?)
                .expect("block index");
            let from_bus = find_music_port_bus(&from.kind.output_ports(), &connection.from_port)
                .ok_or_else(|| {
                    format!(
                        "music studio connection {} references unknown output port {}.{}",
                        connection.id, connection.from_block, connection.from_port
                    )
                })?;
            let to_bus = find_music_port_bus(&to.kind.input_ports(), &connection.to_port)
                .ok_or_else(|| {
                    format!(
                        "music studio connection {} references unknown input port {}.{}",
                        connection.id, connection.to_block, connection.to_port
                    )
                })?;
            if from_bus != connection.bus || to_bus != connection.bus {
                return Err(format!(
                    "music studio connection {} bus mismatch: declared {}, source {}, target {}",
                    connection.id,
                    connection.bus.as_str(),
                    from_bus.as_str(),
                    to_bus.as_str()
                ));
            }
            outgoing.insert(format!(
                "{}:{}",
                connection.from_block, connection.from_port
            ));
            incoming.insert(format!("{}:{}", connection.to_block, connection.to_port));
            adjacency
                .entry(connection.from_block.clone())
                .or_default()
                .push(connection.to_block.clone());
        }

        for block in &self.blocks {
            for port in block.kind.input_ports().iter().filter(|port| port.required) {
                let key = format!("{}:{}", block.id, port.id);
                if !incoming.contains(&key) {
                    return Err(format!(
                        "music studio block {} missing required input port {}",
                        block.id, port.id
                    ));
                }
            }
            for port in block
                .kind
                .output_ports()
                .iter()
                .filter(|port| port.required)
            {
                let key = format!("{}:{}", block.id, port.id);
                if !outgoing.contains(&key) {
                    return Err(format!(
                        "music studio block {} missing required output port {}",
                        block.id, port.id
                    ));
                }
            }
        }

        if !music_board_reaches_output(&source_blocks, &output_blocks, &adjacency) {
            return Err(
                "music studio board has no routed path from a source to master output".to_string(),
            );
        }

        let features = self
            .blocks
            .iter()
            .flat_map(|block| block.feature_terms.iter().map(String::as_str));
        let feature_coverage = music_feature_coverage(features);
        if !feature_coverage.is_complete() {
            return Err(format!(
                "music studio board missing Italian music features: {}",
                feature_coverage.missing.join(", ")
            ));
        }

        Ok(MusicStudioBoardValidation {
            feature_coverage,
            source_blocks,
            output_blocks,
            connection_count: self.connections.len(),
        })
    }

    pub fn to_visual_blocks(&self) -> Result<Vec<VisualBlock>, String> {
        self.validate()?;
        let mut blocks: Vec<VisualBlock> = self
            .blocks
            .iter()
            .enumerate()
            .map(|(index, block)| music_studio_visual_block(block, index))
            .collect();

        for connection in &self.connections {
            let from_index = blocks
                .iter()
                .position(|block| block.id() == connection.from_block)
                .expect("validated source block exists");
            let to_index = blocks
                .iter()
                .position(|block| block.id() == connection.to_block)
                .expect("validated target block exists");
            if from_index < to_index {
                let (left, right) = blocks.split_at_mut(to_index);
                let from = &mut left[from_index];
                let to = &mut right[0];
                from.connect_to(to, visual_connection_options(connection));
            } else {
                let (left, right) = blocks.split_at_mut(from_index);
                let to = &mut left[to_index];
                let from = &mut right[0];
                from.connect_to(to, visual_connection_options(connection));
            }
        }
        Ok(blocks)
    }

    pub fn to_visual_block_ir(&self) -> Result<JsonValue, String> {
        let blocks = self.to_visual_blocks()?;
        let refs: Vec<&VisualBlock> = blocks.iter().collect();
        Ok(visual_block_graph_ir(&refs))
    }
}

pub fn default_music_studio_sound_board() -> MusicStudioBoard {
    MusicStudioBoard::reason_style_default()
}

pub fn default_music_studio_sound_board_ir() -> Result<JsonValue, String> {
    default_music_studio_sound_board().to_visual_block_ir()
}

fn music_port(
    id: &'static str,
    label: &'static str,
    bus: MusicStudioBusKind,
    required: bool,
) -> MusicStudioPortTemplate {
    MusicStudioPortTemplate {
        id,
        label,
        bus,
        required,
    }
}

fn board_conn(
    id: &'static str,
    from_block: &'static str,
    from_port: &'static str,
    to_block: &'static str,
    to_port: &'static str,
    bus: MusicStudioBusKind,
) -> MusicStudioBoardConnection {
    MusicStudioBoardConnection::new(id, from_block, from_port, to_block, to_port, bus)
}

fn find_music_port_bus(
    ports: &[MusicStudioPortTemplate],
    port_id: &str,
) -> Option<MusicStudioBusKind> {
    ports
        .iter()
        .find(|port| port.id == port_id)
        .map(|port| port.bus)
}

fn music_board_reaches_output(
    source_blocks: &[String],
    output_blocks: &[String],
    adjacency: &std::collections::HashMap<String, Vec<String>>,
) -> bool {
    let outputs: std::collections::HashSet<&str> =
        output_blocks.iter().map(String::as_str).collect();
    let mut seen = std::collections::HashSet::new();
    let mut stack = source_blocks.to_vec();
    while let Some(block) = stack.pop() {
        if !seen.insert(block.clone()) {
            continue;
        }
        if outputs.contains(block.as_str()) {
            return true;
        }
        if let Some(next) = adjacency.get(&block) {
            stack.extend(next.iter().cloned());
        }
    }
    false
}

fn music_studio_visual_block(block: &MusicStudioBoardBlock, index: usize) -> VisualBlock {
    let layout = music_board_layout(index);
    VisualBlock::new(
        &block.id,
        VisualBlockOptions {
            kind: Some(format!("music-{}", block.kind.as_str())),
            role: Some(block.kind.visual_role()),
            label: Some(block.label.clone()),
            layout: Some(layout),
            ports: Some(VisualBlockPortSpec {
                inputs: block
                    .kind
                    .input_ports()
                    .iter()
                    .map(music_port_to_visual_port)
                    .collect(),
                outputs: block
                    .kind
                    .output_ports()
                    .iter()
                    .map(music_port_to_visual_port)
                    .collect(),
            }),
            style: Some(block.kind.visual_style()),
            metadata: Some(music_block_metadata(block)),
        },
    )
}

fn music_board_layout(index: usize) -> VisualBlockLayout {
    let columns = 5usize;
    let col = index % columns;
    let row = index / columns;
    VisualBlockLayout {
        x: Some(24.0 + col as f64 * 214.0),
        y: Some(24.0 + row as f64 * 92.0),
        w: Some(188.0),
        h: Some(72.0),
    }
}

fn music_port_to_visual_port(port: &MusicStudioPortTemplate) -> VisualPortInput {
    VisualPortInput::Opts(VisualPortOptions {
        id: port.id.to_string(),
        kind: Some(port.bus.as_str().to_string()),
        label: Some(port.label.to_string()),
        data_type: Some(
            match port.bus {
                MusicStudioBusKind::Midi => "MidiEvent",
                MusicStudioBusKind::Audio => "AudioBuffer",
                MusicStudioBusKind::Control => "ControlSignal",
                MusicStudioBusKind::Analysis => "SpectrumAnalysis",
            }
            .to_string(),
        ),
        required: Some(port.required),
        ..Default::default()
    })
}

fn visual_connection_options(
    connection: &MusicStudioBoardConnection,
) -> VisualBlockConnectionOptions {
    VisualBlockConnectionOptions {
        id: Some(connection.id.clone()),
        from_port: Some(connection.from_port.clone()),
        to_port: Some(connection.to_port.clone()),
        kind: Some(connection.bus.as_str().to_string()),
        metadata: Some(music_connection_metadata(connection)),
        wire_des: Some(false),
    }
}

fn music_visual_style(fill: &str, stroke: &str) -> VisualBlockStyle {
    VisualBlockStyle {
        fill: Some(fill.to_string()),
        stroke: Some(stroke.to_string()),
        text: Some("#171717".to_string()),
    }
}

fn music_block_metadata(block: &MusicStudioBoardBlock) -> Metadata {
    let mut metadata = Metadata::new();
    metadata.insert(
        "domain".to_string(),
        JsonValue::String("music-studio".to_string()),
    );
    metadata.insert(
        "blockKind".to_string(),
        JsonValue::String(block.kind.as_str().to_string()),
    );
    metadata.insert(
        "italianFeatures".to_string(),
        JsonValue::Array(
            block
                .feature_terms
                .iter()
                .map(|feature| JsonValue::String(feature.clone()))
                .collect(),
        ),
    );
    metadata
}

fn music_connection_metadata(connection: &MusicStudioBoardConnection) -> Metadata {
    let mut metadata = Metadata::new();
    metadata.insert(
        "domain".to_string(),
        JsonValue::String("music-studio".to_string()),
    );
    metadata.insert(
        "bus".to_string(),
        JsonValue::String(connection.bus.as_str().to_string()),
    );
    metadata
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MusicUrlSourceKind {
    YouTube,
    Facebook,
    Instagram,
    S3,
    CloudFront,
    Cloudflare,
    StaticAssetHost,
    DirectAudio,
    DirectVideo,
    OtherUrl,
}

impl MusicUrlSourceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            MusicUrlSourceKind::YouTube => "youtube",
            MusicUrlSourceKind::Facebook => "facebook",
            MusicUrlSourceKind::Instagram => "instagram",
            MusicUrlSourceKind::S3 => "s3",
            MusicUrlSourceKind::CloudFront => "cloudfront",
            MusicUrlSourceKind::Cloudflare => "cloudflare",
            MusicUrlSourceKind::StaticAssetHost => "static-asset-host",
            MusicUrlSourceKind::DirectAudio => "direct-audio",
            MusicUrlSourceKind::DirectVideo => "direct-video",
            MusicUrlSourceKind::OtherUrl => "other-url",
        }
    }

    pub fn prefers_external_downloader(self) -> bool {
        matches!(
            self,
            MusicUrlSourceKind::YouTube
                | MusicUrlSourceKind::Facebook
                | MusicUrlSourceKind::Instagram
                | MusicUrlSourceKind::OtherUrl
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MusicUrlInputField {
    pub id: String,
    pub label: String,
    pub placeholder: String,
    pub source_kinds: Vec<MusicUrlSourceKind>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MusicUrlSourceExample {
    pub field_id: &'static str,
    pub source_kind: MusicUrlSourceKind,
    pub source_url: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MusicUrlSourceSpec {
    pub raw_url: String,
    pub host: String,
    pub path: String,
    pub kind: MusicUrlSourceKind,
    pub input_field_id: String,
    pub downloader_hint: String,
    pub direct_media_hint: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectedMusicUrlSource {
    pub submitted_field_id: String,
    pub spec: MusicUrlSourceSpec,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedMusicMediaLink {
    pub source_url: String,
    pub media_url: String,
    pub source_kind: MusicUrlSourceKind,
    pub extractor: String,
    pub mime_type: String,
    pub bitrate: Option<u64>,
    pub content_length: Option<u64>,
    pub duration_seconds: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MusicDownloadedMediaSample {
    pub resolved_link: ResolvedMusicMediaLink,
    pub bytes: Vec<u8>,
    pub content_type: Option<String>,
}

pub fn music_url_input_fields() -> Vec<MusicUrlInputField> {
    vec![
        MusicUrlInputField {
            id: "youtube_url".to_string(),
            label: "YouTube URL".to_string(),
            placeholder: "https://www.youtube.com/watch?v=...".to_string(),
            source_kinds: vec![MusicUrlSourceKind::YouTube],
        },
        MusicUrlInputField {
            id: "facebook_url".to_string(),
            label: "Facebook URL".to_string(),
            placeholder: "https://www.facebook.com/reel/...".to_string(),
            source_kinds: vec![MusicUrlSourceKind::Facebook],
        },
        MusicUrlInputField {
            id: "instagram_url".to_string(),
            label: "Instagram URL".to_string(),
            placeholder: "https://www.instagram.com/reel/...".to_string(),
            source_kinds: vec![MusicUrlSourceKind::Instagram],
        },
        MusicUrlInputField {
            id: "s3_url".to_string(),
            label: "S3 URL".to_string(),
            placeholder: "https://bucket.s3.amazonaws.com/audio.mp3".to_string(),
            source_kinds: vec![MusicUrlSourceKind::S3],
        },
        MusicUrlInputField {
            id: "cloudfront_url".to_string(),
            label: "CloudFront URL".to_string(),
            placeholder: "https://d111111abcdef8.cloudfront.net/seed.mp4".to_string(),
            source_kinds: vec![MusicUrlSourceKind::CloudFront],
        },
        MusicUrlInputField {
            id: "cloudflare_url".to_string(),
            label: "Cloudflare URL".to_string(),
            placeholder: "https://example.r2.cloudflarestorage.com/sample.wav".to_string(),
            source_kinds: vec![MusicUrlSourceKind::Cloudflare],
        },
        MusicUrlInputField {
            id: "static_asset_url".to_string(),
            label: "Static asset URL".to_string(),
            placeholder: "https://static.example.com/music/loop.wav".to_string(),
            source_kinds: vec![MusicUrlSourceKind::StaticAssetHost],
        },
        MusicUrlInputField {
            id: "any_audio_url".to_string(),
            label: "Any audio or media URL".to_string(),
            placeholder: "https://media.example.net/beat.flac".to_string(),
            source_kinds: vec![
                MusicUrlSourceKind::DirectAudio,
                MusicUrlSourceKind::DirectVideo,
                MusicUrlSourceKind::OtherUrl,
            ],
        },
    ]
}

pub fn music_url_source_examples() -> Vec<MusicUrlSourceExample> {
    vec![
        MusicUrlSourceExample {
            field_id: "youtube_url",
            source_kind: MusicUrlSourceKind::YouTube,
            source_url: "https://www.youtube.com/watch?v=abc123",
        },
        MusicUrlSourceExample {
            field_id: "facebook_url",
            source_kind: MusicUrlSourceKind::Facebook,
            source_url: "https://www.facebook.com/reel/123456789",
        },
        MusicUrlSourceExample {
            field_id: "instagram_url",
            source_kind: MusicUrlSourceKind::Instagram,
            source_url: "https://www.instagram.com/reel/ABC123/",
        },
        MusicUrlSourceExample {
            field_id: "s3_url",
            source_kind: MusicUrlSourceKind::S3,
            source_url: "https://bucket.s3.amazonaws.com/audio/loop.mp3",
        },
        MusicUrlSourceExample {
            field_id: "cloudfront_url",
            source_kind: MusicUrlSourceKind::CloudFront,
            source_url: "https://d111111abcdef8.cloudfront.net/audio/seed.mp4",
        },
        MusicUrlSourceExample {
            field_id: "cloudflare_url",
            source_kind: MusicUrlSourceKind::Cloudflare,
            source_url: "https://example.r2.cloudflarestorage.com/sample.wav",
        },
        MusicUrlSourceExample {
            field_id: "static_asset_url",
            source_kind: MusicUrlSourceKind::StaticAssetHost,
            source_url: "https://static.example.com/music/loop.wav",
        },
        MusicUrlSourceExample {
            field_id: "any_audio_url",
            source_kind: MusicUrlSourceKind::DirectAudio,
            source_url: "https://media.example.net/beat.flac",
        },
        MusicUrlSourceExample {
            field_id: "any_audio_url",
            source_kind: MusicUrlSourceKind::DirectVideo,
            source_url: "https://media.example.net/clip.webm",
        },
        MusicUrlSourceExample {
            field_id: "any_audio_url",
            source_kind: MusicUrlSourceKind::OtherUrl,
            source_url: "https://example.net/share/opaque-id",
        },
    ]
}

pub fn select_music_url_input(
    fields: &[(&str, &str)],
) -> Result<Option<SelectedMusicUrlSource>, String> {
    let input_fields = music_url_input_fields();
    for input_field in &input_fields {
        if let Some(value) = nonempty_music_form_value(fields, &input_field.id) {
            let spec = classify_music_source_url(value)
                .map_err(|err| format!("{}: {err}", input_field.id))?;
            return Ok(Some(SelectedMusicUrlSource {
                submitted_field_id: input_field.id.clone(),
                spec,
            }));
        }
    }

    let Some(source_url) = nonempty_music_form_value(fields, "source_url")
        .or_else(|| nonempty_music_form_value(fields, "sourceUrl"))
    else {
        return Ok(None);
    };
    let spec = classify_music_source_url(source_url).map_err(|err| format!("source_url: {err}"))?;
    let submitted_field_id = nonempty_music_form_value(fields, "source_input_field")
        .or_else(|| nonempty_music_form_value(fields, "sourceInputField"))
        .filter(|field_id| input_fields.iter().any(|field| field.id == *field_id))
        .map(str::to_string)
        .unwrap_or_else(|| spec.input_field_id.clone());
    Ok(Some(SelectedMusicUrlSource {
        submitted_field_id,
        spec,
    }))
}

pub fn classify_music_source_url(raw_url: &str) -> Result<MusicUrlSourceSpec, String> {
    let parsed = ParsedMusicUrl::parse(raw_url)?;
    let kind = classify_music_source_host_path(&parsed.host, &parsed.path);
    let direct_media_hint = looks_like_audio_download_path(&parsed.path)
        || looks_like_video_download_path(&parsed.path);
    let input_field_id = music_input_field_for_kind(kind).to_string();
    let downloader_hint = match (kind, direct_media_hint) {
        (_, true) => "direct-http",
        (MusicUrlSourceKind::YouTube, false) => "rust-youtube-player-response",
        (kind, false) if kind.prefers_external_downloader() => "yt-dlp-or-platform-extractor",
        _ => "direct-http",
    };
    Ok(MusicUrlSourceSpec {
        raw_url: parsed.raw_url,
        host: parsed.host,
        path: parsed.path,
        kind,
        input_field_id,
        downloader_hint: downloader_hint.to_string(),
        direct_media_hint,
    })
}

pub fn render_music_url_seed_form_html(endpoint: &str) -> String {
    let endpoint = html_escape(endpoint);
    let fields = music_url_input_fields();
    let source_field_ids_json = format!(
        "[{}]",
        fields
            .iter()
            .map(|field| format!("\"{}\"", json_escape(&field.id)))
            .collect::<Vec<_>>()
            .join(",")
    );
    let mut html = String::new();
    html.push_str(r#"<section id="music-url-seed-panel" class="music-url-seed-panel">"#);
    html.push_str(r#"<div class="music-url-grid">"#);
    for field in &fields {
        let kinds = field
            .source_kinds
            .iter()
            .map(|kind| kind.as_str())
            .collect::<Vec<_>>()
            .join(",");
        html.push_str(&format!(
            r#"<label for="{id}">{label}</label><input id="{id}" name="{id}" type="url" inputmode="url" autocomplete="off" data-source-kinds="{kinds}" placeholder="{placeholder}">"#,
            id = html_escape(&field.id),
            label = html_escape(&field.label),
            kinds = html_escape(&kinds),
            placeholder = html_escape(&field.placeholder),
        ));
    }
    html.push_str(r#"</div>"#);
    html.push_str(
        r#"<label for="music_url_title">Title</label><input id="music_url_title" name="title" type="text" value="music-url-source variation">"#,
    );
    html.push_str(
        r#"<label for="music_url_prompt">Prompt / direction</label><textarea id="music_url_prompt" name="prompt" placeholder="Use the linked audio as inspiration, alter the rhythm, expand the melody, and generate a new piece."></textarea>"#,
    );
    html.push_str(
        r#"<label for="music_url_duration_seconds">Duration seconds</label><input id="music_url_duration_seconds" name="duration_seconds" type="number" min="15" max="240" value="180">"#,
    );
    html.push_str(
        r#"<button id="render_music_url_seed" type="button">Render URL-inspired audio</button><output id="music_url_seed_result" role="status"></output><a id="music_url_seed_download" hidden>Download generated WAV</a>"#,
    );
    html.push_str(&format!(
        r#"<script>
const MUSIC_URL_SEED_ENDPOINT = "{endpoint}";
function collectMusicUrlSeedInput() {{
  const fields = {source_field_ids_json};
  for (const id of fields) {{
    const el = document.getElementById(id);
    const value = el && el.value.trim();
    if (value) return {{ id, value, kinds: el.dataset.sourceKinds || "" }};
  }}
  return null;
}}
async function renderMusicUrlSeed() {{
  const selected = collectMusicUrlSeedInput();
  const result = document.getElementById("music_url_seed_result");
  const download = document.getElementById("music_url_seed_download");
  download.hidden = true;
  download.removeAttribute("href");
  if (!selected) {{
    result.textContent = "Add a YouTube, Facebook, Instagram, S3, CloudFront, Cloudflare, static asset, or direct audio/video URL.";
    return;
  }}
  const fd = new FormData();
  fd.append("source_url", selected.value);
  fd.append("source_input_field", selected.id);
  fd.append("source_platform", selected.kinds.split(",")[0] || "auto");
  fd.append("prompt", document.getElementById("music_url_prompt").value);
  fd.append("duration_seconds", document.getElementById("music_url_duration_seconds").value || "180");
  fd.append("title", document.getElementById("music_url_title").value.trim() || "music-url-source variation");
  const response = await fetch(MUSIC_URL_SEED_ENDPOINT, {{ method: "POST", body: fd }});
  const data = await response.json().catch(() => ({{ ok: false, error: response.statusText }}));
  if (!data.ok) {{
    result.textContent = data.error || response.statusText || "render failed";
    return;
  }}
  const wavUrl = data.wav_url || data.wavUrl || "";
  const details = [
    data.source_kind && `source ${{data.source_kind}}`,
    data.host && `host ${{data.host}}`,
    data.genre && `genre ${{data.genre}}`,
    data.bpm && `bpm ${{Number(data.bpm).toFixed(1)}}`,
    data.duration_seconds && `${{Number(data.duration_seconds).toFixed(1)}}s`,
  ].filter(Boolean).join(" | ");
  result.textContent = details || "render complete";
  if (wavUrl) {{
    download.href = wavUrl;
    download.hidden = false;
  }}
}}
document.getElementById("render_music_url_seed").addEventListener("click", renderMusicUrlSeed);
</script>"#,
        endpoint = endpoint,
        source_field_ids_json = source_field_ids_json
    ));
    html.push_str("</section>");
    html
}

pub fn write_music_url_seed_form_html(
    path: impl AsRef<Path>,
    endpoint: &str,
) -> io::Result<PathBuf> {
    let path = path.as_ref();
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let html = render_music_url_seed_form_document_html(endpoint);
    std::fs::write(path, html)?;
    Ok(path.to_path_buf())
}

pub fn render_music_url_seed_form_document_html(endpoint: &str) -> String {
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>Music URL Seed Inputs</title><style>{}</style></head><body>{}</body></html>",
        music_url_seed_form_css(),
        render_music_url_seed_form_html(endpoint)
    )
}

pub fn music_url_seed_endpoint_contract_json(endpoint: &str) -> String {
    let mut lines = Vec::new();
    lines.push("{".to_string());
    lines.push("  \"schema\": \"des/music-url-seed-endpoint/v1\",".to_string());
    lines.push(format!("  \"endpoint\": \"{}\",", json_escape(endpoint)));
    lines.push("  \"method\": \"POST\",".to_string());
    lines.push("  \"content_type\": \"multipart/form-data\",".to_string());
    lines.push(
        "  \"host_policy\": \"public-http-only-no-credentials-no-localhost-private-or-internal\","
            .to_string(),
    );
    lines.push("  \"url_input_fields\": [".to_string());
    let input_fields = music_url_input_fields();
    for (index, field) in input_fields.iter().enumerate() {
        let comma = if index + 1 == input_fields.len() {
            ""
        } else {
            ","
        };
        let kinds = field
            .source_kinds
            .iter()
            .map(|kind| format!("\"{}\"", json_escape(kind.as_str())))
            .collect::<Vec<_>>()
            .join(", ");
        lines.push(format!(
            "    {{ \"id\": \"{}\", \"label\": \"{}\", \"source_kinds\": [{}] }}{comma}",
            json_escape(&field.id),
            json_escape(&field.label),
            kinds
        ));
    }
    lines.push("  ],".to_string());
    lines.push("  \"examples\": [".to_string());
    let examples = music_url_source_examples();
    for (index, example) in examples.iter().enumerate() {
        let comma = if index + 1 == examples.len() { "" } else { "," };
        lines.push(format!(
            "    {{ \"field_id\": \"{}\", \"source_kind\": \"{}\", \"source_url\": \"{}\" }}{comma}",
            json_escape(example.field_id),
            json_escape(example.source_kind.as_str()),
            json_escape(example.source_url),
        ));
    }
    lines.push("  ],".to_string());
    lines.push("  \"request_fields\": [".to_string());
    for (index, field) in [
        "source_url",
        "source_input_field",
        "source_platform",
        "prompt",
        "duration_seconds",
        "title",
    ]
    .iter()
    .enumerate()
    {
        let comma = if index == 5 { "" } else { "," };
        lines.push(format!("    \"{}\"{comma}", json_escape(field)));
    }
    lines.push("  ],".to_string());
    lines.push("  \"response_fields\": [".to_string());
    for (index, field) in [
        "ok",
        "wav_url",
        "source_url",
        "source_input_field",
        "submitted_field",
        "source_kind",
        "host",
        "downloader",
        "direct_media_hint",
        "seed",
        "title",
        "genre",
        "duration_seconds",
        "bpm",
        "wav_bytes",
        "error_code",
        "error",
    ]
    .iter()
    .enumerate()
    {
        let comma = if index == 16 { "" } else { "," };
        lines.push(format!("    \"{}\"{comma}", json_escape(field)));
    }
    lines.push("  ]".to_string());
    lines.push("}".to_string());
    lines.join("\n")
}

pub fn write_music_url_seed_contract_json(
    path: impl AsRef<Path>,
    endpoint: &str,
) -> io::Result<PathBuf> {
    let path = path.as_ref();
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, music_url_seed_endpoint_contract_json(endpoint))?;
    Ok(path.to_path_buf())
}

fn music_url_seed_form_css() -> &'static str {
    r#"body{margin:0;font:16px/1.45 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f7f7f4;color:#171717}.music-url-seed-panel{box-sizing:border-box;max-width:980px;margin:0 auto;padding:24px}.music-url-grid,.music-url-seed-panel{display:grid;gap:12px}.music-url-grid{grid-template-columns:minmax(170px,220px) minmax(0,1fr);align-items:center}label{font-weight:650}input,textarea,button{box-sizing:border-box;width:100%;font:inherit;border:1px solid #b8b8af;background:#fff;color:#171717;border-radius:6px;padding:10px 12px}textarea{min-height:96px;resize:vertical}button{width:max-content;max-width:100%;background:#18212f;color:#fff;border-color:#18212f;cursor:pointer}output{white-space:pre-wrap;min-height:1.5em}a{color:#084c8d;font-weight:650}@media(max-width:680px){.music-url-grid{grid-template-columns:1fr}.music-url-seed-panel{padding:16px}button{width:100%}}"#
}

fn music_input_field_for_kind(kind: MusicUrlSourceKind) -> &'static str {
    match kind {
        MusicUrlSourceKind::YouTube => "youtube_url",
        MusicUrlSourceKind::Facebook => "facebook_url",
        MusicUrlSourceKind::Instagram => "instagram_url",
        MusicUrlSourceKind::S3 => "s3_url",
        MusicUrlSourceKind::CloudFront => "cloudfront_url",
        MusicUrlSourceKind::Cloudflare => "cloudflare_url",
        MusicUrlSourceKind::StaticAssetHost => "static_asset_url",
        MusicUrlSourceKind::DirectAudio
        | MusicUrlSourceKind::DirectVideo
        | MusicUrlSourceKind::OtherUrl => "any_audio_url",
    }
}

fn nonempty_music_form_value<'a>(fields: &'a [(&str, &str)], field_id: &str) -> Option<&'a str> {
    fields.iter().find_map(|(id, value)| {
        if *id == field_id {
            let value = value.trim();
            if value.is_empty() {
                None
            } else {
                Some(value)
            }
        } else {
            None
        }
    })
}

fn music_form_duration_seconds(fields: &[(&str, &str)], default: f64) -> Result<f64, String> {
    let Some(raw) = nonempty_music_form_value(fields, "duration_seconds")
        .or_else(|| nonempty_music_form_value(fields, "durationSeconds"))
        .or_else(|| nonempty_music_form_value(fields, "music_url_duration_seconds"))
    else {
        return Ok(default);
    };
    let duration_seconds = raw
        .parse::<f64>()
        .map_err(|_| format!("duration_seconds must be a number, got {raw:?}"))?;
    if !duration_seconds.is_finite() || duration_seconds <= 0.0 {
        return Err("duration_seconds must be positive and finite".to_string());
    }
    if duration_seconds > 600.0 {
        return Err("duration_seconds must be at most 600 seconds".to_string());
    }
    Ok(duration_seconds)
}

fn classify_music_source_host_path(host: &str, path: &str) -> MusicUrlSourceKind {
    if host_matches(host, &["youtube.com", "youtu.be", "youtube-nocookie.com"]) {
        MusicUrlSourceKind::YouTube
    } else if host_matches(host, &["facebook.com", "fb.watch"]) {
        MusicUrlSourceKind::Facebook
    } else if host_matches(host, &["instagram.com"]) {
        MusicUrlSourceKind::Instagram
    } else if is_s3_host(host) {
        MusicUrlSourceKind::S3
    } else if host.ends_with(".cloudfront.net") {
        MusicUrlSourceKind::CloudFront
    } else if is_cloudflare_host(host) {
        MusicUrlSourceKind::Cloudflare
    } else if is_static_asset_host(host) {
        MusicUrlSourceKind::StaticAssetHost
    } else if looks_like_audio_download_path(path) {
        MusicUrlSourceKind::DirectAudio
    } else if looks_like_video_download_path(path) {
        MusicUrlSourceKind::DirectVideo
    } else {
        MusicUrlSourceKind::OtherUrl
    }
}

fn host_matches(host: &str, domains: &[&str]) -> bool {
    domains
        .iter()
        .any(|domain| host == *domain || host.ends_with(&format!(".{domain}")))
}

fn is_s3_host(host: &str) -> bool {
    host == "s3.amazonaws.com"
        || host.starts_with("s3.")
        || host.starts_with("s3-")
        || host.contains(".s3.")
        || host.contains(".s3-")
        || host.ends_with(".s3.amazonaws.com")
        || host.ends_with(".s3-website.amazonaws.com")
        || host.contains(".s3-website-")
}

fn is_cloudflare_host(host: &str) -> bool {
    host.ends_with(".r2.cloudflarestorage.com")
        || host.ends_with(".pages.dev")
        || host.ends_with(".workers.dev")
        || host.ends_with(".cloudflarestream.com")
}

fn is_static_asset_host(host: &str) -> bool {
    host.starts_with("static.")
        || host.starts_with("assets.")
        || host.starts_with("cdn.")
        || host.contains(".static.")
        || host.contains(".assets.")
        || host.contains(".cdn.")
        || host.ends_with(".storage.googleapis.com")
        || host == "storage.googleapis.com"
        || host.ends_with(".blob.core.windows.net")
        || host.ends_with(".githubusercontent.com")
        || host == "raw.githubusercontent.com"
}

fn looks_like_audio_download_path(path: &str) -> bool {
    has_any_path_extension(
        path,
        &[
            ".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".oga", ".opus", ".aif", ".aiff",
        ],
    )
}

fn looks_like_video_download_path(path: &str) -> bool {
    has_any_path_extension(path, &[".mp4", ".m4v", ".mov", ".webm", ".mkv"])
}

fn has_any_path_extension(path: &str, extensions: &[&str]) -> bool {
    let lower = path.to_ascii_lowercase();
    let without_fragment = lower.split('#').next().unwrap_or(&lower);
    let without_query = without_fragment
        .split('?')
        .next()
        .unwrap_or(without_fragment);
    extensions
        .iter()
        .any(|extension| without_query.ends_with(extension))
}

fn html_escape(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            ch => out.push(ch),
        }
    }
    out
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ParsedMusicUrl {
    raw_url: String,
    host: String,
    path: String,
}

impl ParsedMusicUrl {
    fn parse(raw_url: &str) -> Result<Self, String> {
        let raw_url = raw_url.trim();
        if raw_url.is_empty() {
            return Err("music source URL must not be empty".to_string());
        }
        if raw_url.chars().count() > 4096 {
            return Err("music source URL must be at most 4096 characters".to_string());
        }
        let (scheme, rest) = raw_url
            .split_once("://")
            .ok_or_else(|| "music source URL must include http:// or https://".to_string())?;
        match scheme.to_ascii_lowercase().as_str() {
            "http" | "https" => {}
            _ => return Err("music source URL must use http or https".to_string()),
        }
        let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
        let authority = &rest[..authority_end];
        if authority.is_empty() {
            return Err("music source URL must include a host".to_string());
        }
        if authority.contains('@') {
            return Err(
                "music source URL must not embed credentials; use dedicated auth fields"
                    .to_string(),
            );
        }
        let host = extract_url_host(authority)?;
        if is_disallowed_music_url_host(&host) {
            return Err(
                "music source URL must use a public host, not localhost/private/internal network"
                    .to_string(),
            );
        }
        let path = rest[authority_end..].to_string();
        Ok(Self {
            raw_url: raw_url.to_string(),
            host,
            path,
        })
    }
}

fn extract_url_host(authority: &str) -> Result<String, String> {
    let host = if let Some(without_open) = authority.strip_prefix('[') {
        let (host, _) = without_open
            .split_once(']')
            .ok_or_else(|| "music source URL has an unterminated IPv6 host".to_string())?;
        host
    } else {
        authority.split(':').next().unwrap_or(authority)
    };
    let host = host.trim().trim_end_matches('.').to_ascii_lowercase();
    if host.is_empty() {
        return Err("music source URL must include a host".to_string());
    }
    Ok(host)
}

fn is_disallowed_music_url_host(host: &str) -> bool {
    if host == "localhost" || host.ends_with(".localhost") {
        return true;
    }
    if let Ok(ipv4) = host.parse::<Ipv4Addr>() {
        return is_disallowed_ipv4_music_host(ipv4);
    }
    if let Ok(ipv6) = host.parse::<Ipv6Addr>() {
        return is_disallowed_ipv6_music_host(ipv6);
    }
    if !host.contains('.') {
        return true;
    }
    host.ends_with(".local")
        || host.ends_with(".lan")
        || host.ends_with(".internal")
        || host.ends_with(".home")
        || host.ends_with(".corp")
}

fn is_disallowed_ipv4_music_host(ip: Ipv4Addr) -> bool {
    let octets = ip.octets();
    ip.is_private()
        || ip.is_loopback()
        || ip.is_link_local()
        || ip.is_unspecified()
        || ip.is_broadcast()
        || octets[0] == 0
        || (octets[0] == 100 && (64..=127).contains(&octets[1]))
        || (octets[0] == 192 && octets[1] == 0 && octets[2] == 0)
        || (octets[0] == 198 && (18..=19).contains(&octets[1]))
        || (octets[0] == 203 && octets[1] == 0 && octets[2] == 113)
}

fn is_disallowed_ipv6_music_host(ip: Ipv6Addr) -> bool {
    let segments = ip.segments();
    let unique_local = (segments[0] & 0xfe00) == 0xfc00;
    let link_local = (segments[0] & 0xffc0) == 0xfe80;
    let documentation = segments[0] == 0x2001 && segments[1] == 0x0db8;
    let ipv4_mapped = segments[0] == 0
        && segments[1] == 0
        && segments[2] == 0
        && segments[3] == 0
        && segments[4] == 0
        && segments[5] == 0xffff;
    let mapped_private = if ipv4_mapped {
        let high = segments[6].to_be_bytes();
        let low = segments[7].to_be_bytes();
        is_disallowed_ipv4_music_host(Ipv4Addr::new(high[0], high[1], low[0], low[1]))
    } else {
        false
    };
    ip.is_loopback()
        || ip.is_unspecified()
        || unique_local
        || link_local
        || documentation
        || mapped_private
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StylePalette {
    RaveCollage,
    BrokenBeatDream,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MusicGenre {
    DrumAndBass,
    House,
    Trance,
    Dance,
    Electronica,
    Jazz,
    Ambient,
    AmbientTechno,
    Idm,
    Downtempo,
    Chillout,
    TripHop,
    MinimalTechno,
    Microhouse,
    Techno,
    AcidTechno,
    Breakbeat,
    Breakcore,
    Glitch,
    GlitchHop,
    FutureGarage,
    DubTechno,
    Dubstep,
    UkGarage,
    LiquidFunk,
    Neurofunk,
    Jungle,
    FootworkJuke,
    GhettoTech,
    Electro,
    PostRock,
    MathRock,
    InstrumentalRock,
    ProgressiveRock,
    SurfRock,
    Krautrock,
    Shoegaze,
    SpaceRock,
    Classical,
    Minimalism,
    ContemporaryClassical,
    ChamberMusic,
    FilmScore,
    NeoClassical,
    ImpressionistClassical,
    Flamenco,
    CelticInstrumentalFolk,
    BluegrassInstrumental,
    IndianClassical,
    Gamelan,
    ArabicMaqam,
    TangoNuevo,
    CinematicOrchestral,
    Drone,
    Soundscape,
    PostMinimalElectronic,
    ExperimentalElectronic,
    NoiseMusic,
    AmbientIndustrial,
}

impl MusicGenre {
    pub fn as_str(self) -> &'static str {
        match self {
            MusicGenre::DrumAndBass => "drum-n-bass",
            MusicGenre::House => "house",
            MusicGenre::Trance => "trance",
            MusicGenre::Dance => "dance",
            MusicGenre::Electronica => "electronica",
            MusicGenre::Jazz => "jazz",
            MusicGenre::Ambient => "ambient",
            MusicGenre::AmbientTechno => "ambient techno",
            MusicGenre::Idm => "idm",
            MusicGenre::Downtempo => "downtempo",
            MusicGenre::Chillout => "chillout",
            MusicGenre::TripHop => "trip-hop",
            MusicGenre::MinimalTechno => "minimal techno",
            MusicGenre::Microhouse => "microhouse",
            MusicGenre::Techno => "techno",
            MusicGenre::AcidTechno => "acid techno",
            MusicGenre::Breakbeat => "breakbeat",
            MusicGenre::Breakcore => "breakcore",
            MusicGenre::Glitch => "glitch",
            MusicGenre::GlitchHop => "glitch hop",
            MusicGenre::FutureGarage => "future garage",
            MusicGenre::DubTechno => "dub techno",
            MusicGenre::Dubstep => "dubstep",
            MusicGenre::UkGarage => "uk garage",
            MusicGenre::LiquidFunk => "liquid funk",
            MusicGenre::Neurofunk => "neurofunk",
            MusicGenre::Jungle => "jungle",
            MusicGenre::FootworkJuke => "footwork / juke",
            MusicGenre::GhettoTech => "ghetto tech",
            MusicGenre::Electro => "electro",
            MusicGenre::PostRock => "post-rock",
            MusicGenre::MathRock => "math rock",
            MusicGenre::InstrumentalRock => "instrumental rock",
            MusicGenre::ProgressiveRock => "progressive rock",
            MusicGenre::SurfRock => "surf rock",
            MusicGenre::Krautrock => "krautrock",
            MusicGenre::Shoegaze => "shoegaze",
            MusicGenre::SpaceRock => "space rock",
            MusicGenre::Classical => "classical",
            MusicGenre::Minimalism => "minimalism",
            MusicGenre::ContemporaryClassical => "contemporary classical",
            MusicGenre::ChamberMusic => "chamber music",
            MusicGenre::FilmScore => "film score",
            MusicGenre::NeoClassical => "neo-classical",
            MusicGenre::ImpressionistClassical => "impressionist classical",
            MusicGenre::Flamenco => "flamenco",
            MusicGenre::CelticInstrumentalFolk => "celtic instrumental folk",
            MusicGenre::BluegrassInstrumental => "bluegrass instrumental",
            MusicGenre::IndianClassical => "indian classical",
            MusicGenre::Gamelan => "gamelan",
            MusicGenre::ArabicMaqam => "arabic maqam",
            MusicGenre::TangoNuevo => "tango nuevo",
            MusicGenre::CinematicOrchestral => "cinematic orchestral",
            MusicGenre::Drone => "drone",
            MusicGenre::Soundscape => "soundscape",
            MusicGenre::PostMinimalElectronic => "post-minimal electronic",
            MusicGenre::ExperimentalElectronic => "experimental electronic",
            MusicGenre::NoiseMusic => "noise music",
            MusicGenre::AmbientIndustrial => "ambient industrial",
        }
    }

    pub fn default_bpm(self) -> f64 {
        match self {
            MusicGenre::DrumAndBass | MusicGenre::LiquidFunk | MusicGenre::Jungle => 172.0,
            MusicGenre::Neurofunk => 174.0,
            MusicGenre::Breakcore => 184.0,
            MusicGenre::FootworkJuke => 160.0,
            MusicGenre::House | MusicGenre::Microhouse => 124.0,
            MusicGenre::Trance => 136.0,
            MusicGenre::Dance | MusicGenre::Electronica | MusicGenre::Breakbeat => 128.0,
            MusicGenre::Dubstep | MusicGenre::FutureGarage | MusicGenre::UkGarage => 140.0,
            MusicGenre::Techno | MusicGenre::MinimalTechno | MusicGenre::AcidTechno => 132.0,
            MusicGenre::Ambient
            | MusicGenre::Drone
            | MusicGenre::Soundscape
            | MusicGenre::Chillout => 82.0,
            MusicGenre::Downtempo | MusicGenre::TripHop | MusicGenre::DubTechno => 92.0,
            MusicGenre::Jazz | MusicGenre::TangoNuevo => 118.0,
            MusicGenre::Idm | MusicGenre::Glitch | MusicGenre::GlitchHop => 112.0,
            MusicGenre::Electro | MusicGenre::GhettoTech => 132.0,
            _ => 104.0,
        }
    }

    fn drum_density(self) -> f64 {
        match self {
            MusicGenre::DrumAndBass
            | MusicGenre::Jungle
            | MusicGenre::Neurofunk
            | MusicGenre::Breakcore => 1.0,
            MusicGenre::House
            | MusicGenre::Dance
            | MusicGenre::Trance
            | MusicGenre::Techno
            | MusicGenre::AcidTechno => 0.78,
            MusicGenre::Ambient
            | MusicGenre::Drone
            | MusicGenre::Soundscape
            | MusicGenre::Chillout => 0.28,
            MusicGenre::Jazz | MusicGenre::TripHop | MusicGenre::Downtempo => 0.55,
            _ => 0.66,
        }
    }

    fn percussion_gain(self) -> f64 {
        match self {
            MusicGenre::DrumAndBass
            | MusicGenre::Jungle
            | MusicGenre::Neurofunk
            | MusicGenre::LiquidFunk
            | MusicGenre::Breakbeat
            | MusicGenre::Breakcore => 0.84,
            _ => 0.86,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KeyChangeEvent {
    pub time_seconds: f64,
    pub scale_step_delta: i32,
    pub target_offset_steps: i32,
    pub italian_feature: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TimeSignatureChangeEvent {
    pub time_seconds: f64,
    pub numerator: u8,
    pub denominator: u8,
    pub italian_feature: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PauseEvent {
    pub start_seconds: f64,
    pub duration_seconds: f64,
    pub depth: f64,
    pub italian_feature: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DrumVariationSummary {
    pub pattern_names: Vec<String>,
    pub drum_steps: usize,
    pub fills: usize,
    pub ghost_hits: usize,
    pub syncopations: usize,
    pub meter_sensitive_variations: usize,
    pub micro_variations: usize,
    pub repetition_reduction_target: f64,
    pub percussion_gain: f64,
}

impl Default for DrumVariationSummary {
    fn default() -> Self {
        Self {
            pattern_names: Vec::new(),
            drum_steps: 0,
            fills: 0,
            ghost_hits: 0,
            syncopations: 0,
            meter_sensitive_variations: 0,
            micro_variations: 0,
            repetition_reduction_target: 0.10,
            percussion_gain: 0.84,
        }
    }
}

impl DrumVariationSummary {
    pub fn variation_ratio(&self) -> f64 {
        if self.drum_steps == 0 {
            return 0.0;
        }
        self.micro_variations as f64 / self.drum_steps as f64
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrackStructurePlan {
    pub title: String,
    pub genre: MusicGenre,
    pub key_changes: Vec<KeyChangeEvent>,
    pub time_signature_changes: Vec<TimeSignatureChangeEvent>,
    pub pauses: Vec<PauseEvent>,
    pub italian_features: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SongSpec {
    pub title: String,
    pub genre: MusicGenre,
    pub duration_seconds: f64,
    pub sample_rate: u32,
    pub bpm: f64,
    pub seed: u32,
    pub scale: MicrotonalScale,
    pub palette: StylePalette,
    pub instrument_palette: Option<InstrumentPalette>,
    pub structure_plan: Option<TrackStructurePlan>,
    pub sample_manifest: SampleManifest,
}

impl Default for SongSpec {
    fn default() -> Self {
        let genre = MusicGenre::Electronica;
        Self {
            title: "invented microtonal study".to_string(),
            genre,
            duration_seconds: DEFAULT_SONG_SECONDS,
            sample_rate: DEFAULT_SAMPLE_RATE,
            bpm: genre.default_bpm(),
            seed: 0x5150_1979,
            scale: MicrotonalScale::rave_collage_19_edo(),
            palette: StylePalette::RaveCollage,
            instrument_palette: None,
            structure_plan: None,
            sample_manifest: SampleManifest {
                sources: vec![SampleSource {
                    id: "generated-vocal-chops".to_string(),
                    path: None,
                    license: SampleLicense::GeneratedByThisEngine,
                    tags: vec!["synthetic".to_string(), "vocal-texture".to_string()],
                }],
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SongSection {
    pub name: String,
    pub start_seconds: f64,
    pub end_seconds: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SongPartSummary {
    pub name: String,
    pub role: InstrumentRole,
    pub instrument: String,
    pub events: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ArrangementSummary {
    pub title: String,
    pub genre: MusicGenre,
    pub duration_seconds: f64,
    pub bpm: f64,
    pub scale_name: String,
    pub sections: Vec<SongSection>,
    pub parts: Vec<SongPartSummary>,
    pub instruments: Vec<String>,
    pub key_changes: Vec<KeyChangeEvent>,
    pub time_signature_changes: Vec<TimeSignatureChangeEvent>,
    pub pauses: Vec<PauseEvent>,
    pub italian_features: Vec<String>,
    pub drum_variation: DrumVariationSummary,
    pub rendered_events: usize,
    pub peak: f64,
    pub rms: f64,
    pub spectral_centroid_hz: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SongRender {
    pub audio: AudioBuffer,
    pub summary: ArrangementSummary,
    pub sample_manifest: SampleManifest,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AlbumTrackSummary {
    pub index: usize,
    pub path: String,
    pub summary: ArrangementSummary,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AlbumRenderSummary {
    pub out_dir: String,
    pub manifest_path: String,
    pub tracks: Vec<AlbumTrackSummary>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MusicSampleSeed {
    pub source_path: String,
    pub duration_seconds: f64,
    pub seed: u32,
    pub byte_entropy: f64,
    pub suggested_genre: MusicGenre,
    pub suggested_bpm: f64,
    pub key_bias_steps: i32,
    pub meter_bias: (u8, u8),
    pub descriptors: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MusicSamplePromptInfluence {
    pub prompt_hash: u32,
    pub prompt_chars: usize,
    pub genre: Option<MusicGenre>,
    pub bpm_delta: f64,
    pub key_bias_delta: i32,
    pub meter_bias: Option<(u8, u8)>,
    pub feature_tags: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MusicUrlSeedRenderResult {
    pub selected_source: SelectedMusicUrlSource,
    pub sample_seed: MusicSampleSeed,
    pub output_path: PathBuf,
    pub wav_bytes: u64,
    pub summary: ArrangementSummary,
}

pub fn generate_three_minute_song(seed: u32) -> SongRender {
    generate_microtonal_song(SongSpec {
        seed,
        ..Default::default()
    })
}

pub fn analyze_music_sample_prompt(prompt: &str) -> Option<MusicSamplePromptInfluence> {
    let prompt = prompt.trim();
    if prompt.is_empty() {
        return None;
    }
    let lower = prompt.to_ascii_lowercase();
    let prompt_hash = hash_bytes_to_seed(prompt.as_bytes());
    let genre = if contains_any(&lower, &["neurofunk", "reese", "dark dnb"]) {
        Some(MusicGenre::Neurofunk)
    } else if contains_any(&lower, &["liquid funk", "liquid dnb"]) {
        Some(MusicGenre::LiquidFunk)
    } else if contains_any(&lower, &["jungle", "amen"]) {
        Some(MusicGenre::Jungle)
    } else if contains_any(&lower, &["drum-n-bass", "drum and bass", "dnb", "d&b"]) {
        Some(MusicGenre::DrumAndBass)
    } else if contains_any(&lower, &["breakcore"]) {
        Some(MusicGenre::Breakcore)
    } else if contains_any(&lower, &["breakbeat", "break beat"]) {
        Some(MusicGenre::Breakbeat)
    } else if contains_any(&lower, &["future garage", "uk garage", "garage swing"]) {
        Some(MusicGenre::FutureGarage)
    } else if contains_any(&lower, &["dubstep", "wobble"]) {
        Some(MusicGenre::Dubstep)
    } else if contains_any(&lower, &["trip hop", "trip-hop"]) {
        Some(MusicGenre::TripHop)
    } else if contains_any(&lower, &["glitch hop", "glitch-hop"]) {
        Some(MusicGenre::GlitchHop)
    } else if contains_any(&lower, &["glitch", "stutter"]) {
        Some(MusicGenre::Glitch)
    } else if contains_any(&lower, &["ambient", "drone", "wide space"]) {
        Some(MusicGenre::Ambient)
    } else if contains_any(&lower, &["downtempo", "slow burn"]) {
        Some(MusicGenre::Downtempo)
    } else if contains_any(&lower, &["acid", "303"]) {
        Some(MusicGenre::AcidTechno)
    } else if contains_any(&lower, &["techno"]) {
        Some(MusicGenre::Techno)
    } else if contains_any(&lower, &["house"]) {
        Some(MusicGenre::House)
    } else {
        None
    };

    let mut bpm_delta = 0.0;
    if contains_any(
        &lower,
        &[
            "slower",
            "slow",
            "half time",
            "half-time",
            "laid back",
            "dreamy",
        ],
    ) {
        bpm_delta -= 10.0;
    }
    if contains_any(
        &lower,
        &[
            "faster",
            "fast",
            "accelerate",
            "urgent",
            "sprint",
            "rave",
            "energy",
        ],
    ) {
        bpm_delta += 8.0;
    }
    if contains_any(&lower, &["double time", "double-time"]) {
        bpm_delta += 12.0;
    }
    if contains_any(&lower, &["massive", "heavy", "pressure"]) {
        bpm_delta += 3.0;
    }

    let mut key_bias_delta = (prompt_hash % 7) as i32 - 3;
    if contains_any(&lower, &["brighter", "lift", "uplift", "major"]) {
        key_bias_delta += 2;
    }
    if contains_any(&lower, &["darker", "minor", "shadow", "noir"]) {
        key_bias_delta -= 2;
    }

    let meter_bias = if contains_any(&lower, &["13/16", "thirteen", "stutter"]) {
        Some((13, 16))
    } else if contains_any(&lower, &["11/8", "eleven", "polyrhythm"]) {
        Some((11, 8))
    } else if contains_any(&lower, &["7/8", "seven"]) {
        Some((7, 8))
    } else if contains_any(&lower, &["5/4", "five"]) {
        Some((5, 4))
    } else if contains_any(&lower, &["6/8", "waltz", "swing"]) {
        Some((6, 8))
    } else if contains_any(&lower, &["4/4", "straight"]) {
        Some((4, 4))
    } else {
        None
    };

    let mut feature_tags = vec!["prompt-directed".to_string()];
    for (tag, words) in [
        ("expand", &["expand", "longer arc", "build out"][..]),
        ("alter", &["alter", "mutate", "transform"][..]),
        ("slice", &["slice", "chop", "cut-up", "collage"][..]),
        ("melody", &["melody", "melodic", "hook", "theme"][..]),
        (
            "massive-synth",
            &["massive synth", "big synth", "wall of synth"][..],
        ),
        ("vocal-texture", &["vocal", "voice", "choir"][..]),
        ("space", &["space", "reverb", "wide", "dub"][..]),
        (
            "less-drums",
            &["less drums", "softer drums", "lower drums"][..],
        ),
        (
            "more-drums",
            &["more drums", "drum fills", "busier drums"][..],
        ),
    ] {
        if contains_any(&lower, words) {
            feature_tags.push(tag.to_string());
        }
    }

    Some(MusicSamplePromptInfluence {
        prompt_hash,
        prompt_chars: prompt.chars().count(),
        genre,
        bpm_delta,
        key_bias_delta,
        meter_bias,
        feature_tags,
    })
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

pub fn generate_track_structure_plan(
    title: impl Into<String>,
    genre: MusicGenre,
    duration_seconds: f64,
    seed: u32,
) -> TrackStructurePlan {
    require_finite("duration_seconds", duration_seconds);
    assert!(duration_seconds > 0.0, "duration_seconds must be positive");
    let key_count = 1 + (seed as usize % 2);
    let meter_count = 2 + ((seed.rotate_left(7) as usize) % 2);
    let key_deltas = [3, -2, 5, -4, 2, -3, 6, -5];
    let meter_options = [
        (7, 8),
        (11, 8),
        (13, 16),
        (5, 4),
        (9, 8),
        (15, 16),
        (10, 8),
        (6, 8),
    ];
    let meter_positions = [
        0.18 + (seed % 7) as f64 * 0.004,
        0.46 + (seed.rotate_left(5) % 9) as f64 * 0.003,
        0.72 + (seed.rotate_left(11) % 7) as f64 * 0.004,
    ];
    let mut key_changes = Vec::new();
    let mut target = 0i32;
    for i in 0..key_count {
        let delta = key_deltas[((seed as usize / 3) + i * 2) % key_deltas.len()];
        target += delta;
        key_changes.push(KeyChangeEvent {
            time_seconds: duration_seconds * if i == 0 { 0.38 } else { 0.68 },
            scale_step_delta: delta,
            target_offset_steps: target,
            italian_feature: "modulazione".to_string(),
        });
    }

    let mut time_signature_changes = Vec::new();
    for i in 0..meter_count {
        let (numerator, denominator) =
            meter_options[((seed as usize / 11) + i * 3) % meter_options.len()];
        time_signature_changes.push(TimeSignatureChangeEvent {
            time_seconds: duration_seconds * meter_positions[i],
            numerator,
            denominator,
            italian_feature: "metro".to_string(),
        });
    }

    TrackStructurePlan {
        title: title.into(),
        genre,
        key_changes,
        time_signature_changes,
        pauses: vec![
            PauseEvent {
                start_seconds: duration_seconds * 0.315,
                duration_seconds: 0.38 + (seed % 5) as f64 * 0.045,
                depth: 0.08,
                italian_feature: "pausa".to_string(),
            },
            PauseEvent {
                start_seconds: duration_seconds * 0.705,
                duration_seconds: 0.28 + (seed % 7) as f64 * 0.035,
                depth: 0.12,
                italian_feature: "pausa".to_string(),
            },
        ],
        italian_features: all_italian_music_feature_terms(),
    }
}

pub fn derive_music_sample_seed_from_mp4(path: impl AsRef<Path>) -> io::Result<MusicSampleSeed> {
    derive_music_sample_seed_from_mp4_with_limit(path, MAX_MUSIC_SAMPLE_SEED_BYTES)
}

pub fn derive_music_sample_seed_from_mp4_with_limit(
    path: impl AsRef<Path>,
    max_bytes: u64,
) -> io::Result<MusicSampleSeed> {
    let path = path.as_ref();
    if max_bytes < 8 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "music-sample-seed byte limit is too small for an MP4 header",
        ));
    }
    let mut file = std::fs::File::open(path)?;
    let mut bytes = Vec::new();
    let mut limited = (&mut file).take(max_bytes.saturating_add(1));
    limited.read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("music-sample-seed file exceeds {max_bytes} bytes"),
        ));
    }
    let duration_seconds = parse_mp4_duration_seconds(&bytes).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "could not read mp4 mvhd duration for music-sample-seed",
        )
    })?;
    if !(10.0..=50.0).contains(&duration_seconds) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "music-sample-seed requires a 10-50 second mp4, got {:.2}s",
                duration_seconds
            ),
        ));
    }
    let seed = hash_bytes_to_seed(&bytes);
    let entropy = byte_entropy(&bytes);
    let suggested_genre = if entropy > 7.65 {
        MusicGenre::Breakbeat
    } else if entropy > 7.35 {
        MusicGenre::DrumAndBass
    } else if entropy > 6.95 {
        MusicGenre::TripHop
    } else {
        MusicGenre::Downtempo
    };
    let suggested_bpm = (suggested_genre.default_bpm() + (seed % 9) as f64 - 4.0).max(70.0);
    let key_bias_steps = ((seed >> 5) % 11) as i32 - 5;
    let meter_options = [(7, 8), (11, 8), (13, 16), (5, 4), (9, 8), (15, 16)];
    let meter_bias = meter_options[((seed >> 13) as usize) % meter_options.len()];
    let descriptors = vec![
        "music-sample-seed".to_string(),
        format!("duration={:.2}s", duration_seconds),
        format!("byte-entropy={:.3}", entropy),
        format!("suggested-genre={}", suggested_genre.as_str()),
    ];
    Ok(MusicSampleSeed {
        source_path: canonical_or_original(path.to_path_buf())
            .display()
            .to_string(),
        duration_seconds,
        seed,
        byte_entropy: entropy,
        suggested_genre,
        suggested_bpm,
        key_bias_steps,
        meter_bias,
        descriptors,
    })
}

pub fn derive_music_sample_seed_from_url(raw_url: &str) -> Result<MusicSampleSeed, String> {
    let spec = classify_music_source_url(raw_url)?;
    Ok(derive_music_sample_seed_from_url_source(&spec))
}

pub fn derive_music_sample_seed_from_url_source(spec: &MusicUrlSourceSpec) -> MusicSampleSeed {
    let seed_material = format!(
        "music-url-seed\0{}\0{}\0{}\0{}\0{}\0{}",
        spec.raw_url,
        spec.host,
        spec.path,
        spec.kind.as_str(),
        spec.downloader_hint,
        spec.direct_media_hint
    );
    let bytes = seed_material.as_bytes();
    let seed = hash_bytes_to_seed(bytes);
    let entropy = byte_entropy(bytes);
    let suggested_genre = music_url_genre_hint(spec.kind, seed, spec.direct_media_hint);
    let suggested_bpm =
        (suggested_genre.default_bpm() + ((seed >> 3) % 15) as f64 - 7.0).clamp(68.0, 188.0);
    let key_bias_steps = ((seed >> 9) % 13) as i32 - 6;
    let meter_options = [(4, 4), (7, 8), (9, 8), (11, 8), (13, 16), (15, 16)];
    let meter_bias = meter_options[((seed >> 17) as usize) % meter_options.len()];
    let duration_seconds = 10.0 + (seed % 4000) as f64 / 100.0;
    MusicSampleSeed {
        source_path: spec.raw_url.clone(),
        duration_seconds,
        seed,
        byte_entropy: entropy,
        suggested_genre,
        suggested_bpm,
        key_bias_steps,
        meter_bias,
        descriptors: vec![
            "music-url-seed".to_string(),
            format!("source-kind={}", spec.kind.as_str()),
            format!("host={}", spec.host),
            format!("downloader={}", spec.downloader_hint),
            format!("direct-media={}", spec.direct_media_hint),
            format!("url-entropy={:.3}", entropy),
        ],
    }
}

pub fn resolve_public_music_media_link(raw_url: &str) -> Result<ResolvedMusicMediaLink, String> {
    let spec = classify_music_source_url(raw_url)?;
    resolve_public_music_media_link_source(&spec)
}

pub fn resolve_public_music_media_link_source(
    spec: &MusicUrlSourceSpec,
) -> Result<ResolvedMusicMediaLink, String> {
    match spec.kind {
        MusicUrlSourceKind::YouTube => resolve_youtube_public_media_link(spec),
        MusicUrlSourceKind::S3
        | MusicUrlSourceKind::CloudFront
        | MusicUrlSourceKind::Cloudflare
        | MusicUrlSourceKind::StaticAssetHost
        | MusicUrlSourceKind::DirectAudio
        | MusicUrlSourceKind::DirectVideo
            if spec.direct_media_hint =>
        {
            Ok(ResolvedMusicMediaLink {
                source_url: spec.raw_url.clone(),
                media_url: spec.raw_url.clone(),
                source_kind: spec.kind,
                extractor: "direct-http".to_string(),
                mime_type: if looks_like_audio_download_path(&spec.path) {
                    "audio/*".to_string()
                } else {
                    "video/*".to_string()
                },
                bitrate: None,
                content_length: None,
                duration_seconds: None,
            })
        }
        _ => Err(format!(
            "Rust media resolver supports YouTube watch links and direct audio/video URLs; {} still needs a platform extractor",
            spec.kind.as_str()
        )),
    }
}

pub fn download_public_music_media_sample(
    raw_url: &str,
) -> Result<MusicDownloadedMediaSample, String> {
    let spec = classify_music_source_url(raw_url)?;
    download_public_music_media_sample_source(&spec)
}

pub fn download_public_music_media_sample_source(
    spec: &MusicUrlSourceSpec,
) -> Result<MusicDownloadedMediaSample, String> {
    let resolved_link = resolve_public_music_media_link_source(spec)?;
    let response = fetch_public_music_http_bytes(
        &resolved_link.media_url,
        MAX_MUSIC_PUBLIC_MEDIA_SEED_BYTES,
        Some((
            0,
            MAX_MUSIC_PUBLIC_MEDIA_SEED_BYTES.saturating_sub(1) as u64,
        )),
        true,
    )?;
    let content_type = music_http_header_value(&response.headers, "content-type");
    let advertised_type = content_type
        .as_deref()
        .unwrap_or(resolved_link.mime_type.as_str());
    if !is_audio_or_video_content_type(advertised_type) {
        return Err(format!(
            "direct HTTP resource is not advertised as audio/video (content-type {:?})",
            advertised_type
        ));
    }
    if response.body.is_empty() {
        return Err("direct HTTP media response was empty".to_string());
    }
    Ok(MusicDownloadedMediaSample {
        resolved_link,
        bytes: response.body,
        content_type,
    })
}

pub fn derive_music_sample_seed_from_public_media_link(
    raw_url: &str,
) -> Result<MusicSampleSeed, String> {
    let spec = classify_music_source_url(raw_url)?;
    derive_music_sample_seed_from_public_media_link_source(&spec)
}

pub fn derive_music_sample_seed_from_public_media_link_source(
    spec: &MusicUrlSourceSpec,
) -> Result<MusicSampleSeed, String> {
    let sample = download_public_music_media_sample_source(spec)?;
    Ok(music_sample_seed_from_downloaded_media(spec, &sample))
}

fn music_sample_seed_from_downloaded_media(
    spec: &MusicUrlSourceSpec,
    sample: &MusicDownloadedMediaSample,
) -> MusicSampleSeed {
    let mut seed_material = Vec::new();
    seed_material.extend_from_slice(b"music-url-media-seed\0");
    seed_material.extend_from_slice(spec.raw_url.as_bytes());
    seed_material.push(0);
    seed_material.extend_from_slice(sample.resolved_link.media_url.as_bytes());
    seed_material.push(0);
    seed_material.extend_from_slice(sample.resolved_link.mime_type.as_bytes());
    seed_material.push(0);
    seed_material.extend_from_slice(&sample.bytes);
    let seed = hash_bytes_to_seed(&seed_material);
    let entropy = byte_entropy(&sample.bytes);
    let suggested_genre = music_url_genre_hint(spec.kind, seed, true);
    let suggested_bpm =
        (suggested_genre.default_bpm() + ((seed >> 4) % 17) as f64 - 8.0).clamp(68.0, 188.0);
    let key_bias_steps = ((seed >> 10) % 13) as i32 - 6;
    let meter_options = [(4, 4), (7, 8), (9, 8), (11, 8), (13, 16), (15, 16)];
    let meter_bias = meter_options[((seed >> 18) as usize) % meter_options.len()];
    let byte_seconds = sample
        .resolved_link
        .bitrate
        .filter(|bitrate| *bitrate > 0)
        .map(|bitrate| sample.bytes.len() as f64 * 8.0 / bitrate as f64)
        .filter(|seconds| seconds.is_finite() && *seconds > 0.0);
    let duration_seconds = byte_seconds
        .or(sample.resolved_link.duration_seconds)
        .unwrap_or_else(|| 10.0 + (seed % 4000) as f64 / 100.0)
        .clamp(10.0, 50.0);
    let resolved_host = ParsedMusicUrl::parse(&sample.resolved_link.media_url)
        .map(|url| url.host)
        .unwrap_or_else(|_| "unknown-media-host".to_string());
    let advertised_type = sample
        .content_type
        .as_deref()
        .unwrap_or(sample.resolved_link.mime_type.as_str());
    MusicSampleSeed {
        source_path: spec.raw_url.clone(),
        duration_seconds,
        seed,
        byte_entropy: entropy,
        suggested_genre,
        suggested_bpm,
        key_bias_steps,
        meter_bias,
        descriptors: vec![
            "music-url-media-seed".to_string(),
            format!("source-kind={}", spec.kind.as_str()),
            format!("host={}", spec.host),
            format!("media-host={resolved_host}"),
            format!("extractor={}", sample.resolved_link.extractor),
            format!("media-mime={advertised_type}"),
            format!("downloaded-bytes={}", sample.bytes.len()),
            format!("byte-entropy={entropy:.3}"),
        ],
    }
}

fn resolve_youtube_public_media_link(
    spec: &MusicUrlSourceSpec,
) -> Result<ResolvedMusicMediaLink, String> {
    let watch_url = canonical_youtube_watch_url(spec)?;
    let html = fetch_public_music_http_text(&watch_url, MUSIC_MEDIA_HTTP_MAX_TEXT_BYTES)?;
    let player_json = extract_youtube_initial_player_response_json(&html)
        .ok_or_else(|| "YouTube watch page did not include ytInitialPlayerResponse".to_string())?;
    let response: serde_json::Value = serde_json::from_str(&player_json)
        .map_err(|err| format!("could not parse YouTube player response JSON: {err}"))?;
    select_youtube_public_media_link_from_player_response(spec, &response)
}

fn canonical_youtube_watch_url(spec: &MusicUrlSourceSpec) -> Result<String, String> {
    let video_id = youtube_video_id_from_spec(spec)
        .ok_or_else(|| "YouTube URL did not include a video id".to_string())?;
    Ok(format!("https://www.youtube.com/watch?v={video_id}"))
}

fn youtube_video_id_from_spec(spec: &MusicUrlSourceSpec) -> Option<String> {
    if host_matches(&spec.host, &["youtu.be"]) {
        let segment = spec
            .path
            .trim_start_matches('/')
            .split(['?', '#', '/'])
            .next()
            .unwrap_or_default();
        return valid_youtube_video_id(segment).then(|| segment.to_string());
    }
    if let Some(value) = query_value_from_url_path(&spec.path, "v") {
        if valid_youtube_video_id(&value) {
            return Some(value);
        }
    }
    for prefix in ["/shorts/", "/embed/", "/v/"] {
        if let Some(rest) = spec.path.strip_prefix(prefix) {
            let segment = rest
                .split(['?', '#', '/'])
                .next()
                .unwrap_or_default()
                .trim();
            if valid_youtube_video_id(segment) {
                return Some(segment.to_string());
            }
        }
    }
    None
}

fn valid_youtube_video_id(value: &str) -> bool {
    (6..=128).contains(&value.len())
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

fn query_value_from_url_path(path: &str, key: &str) -> Option<String> {
    let query = path
        .split_once('?')?
        .1
        .split('#')
        .next()
        .unwrap_or_default();
    for pair in query.split('&') {
        let (raw_key, raw_value) = pair.split_once('=').unwrap_or((pair, ""));
        if raw_key == key {
            return percent_decode_url_component(raw_value).ok();
        }
    }
    None
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn percent_decode_url_component(value: &str) -> Result<String, String> {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            if index + 2 >= bytes.len() {
                return Err("truncated percent escape in URL component".to_string());
            }
            let hi = hex_value(bytes[index + 1])
                .ok_or_else(|| "invalid percent escape in URL component".to_string())?;
            let lo = hex_value(bytes[index + 2])
                .ok_or_else(|| "invalid percent escape in URL component".to_string())?;
            out.push((hi << 4) | lo);
            index += 3;
        } else {
            out.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8(out).map_err(|_| "decoded URL component is not UTF-8".to_string())
}

fn extract_youtube_initial_player_response_json(html: &str) -> Option<String> {
    for marker in [
        "ytInitialPlayerResponse =",
        "ytInitialPlayerResponse=",
        "\"ytInitialPlayerResponse\":",
    ] {
        if let Some(marker_start) = html.find(marker) {
            let after_marker = marker_start + marker.len();
            if let Some(json_start) = html[after_marker..].find('{') {
                let absolute_start = after_marker + json_start;
                if let Some(json) = extract_balanced_json_object(html, absolute_start) {
                    return Some(json);
                }
            }
        }
    }
    None
}

fn extract_balanced_json_object(text: &str, start: usize) -> Option<String> {
    let bytes = text.as_bytes();
    if bytes.get(start).copied()? != b'{' {
        return None;
    }
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for index in start..bytes.len() {
        let byte = bytes[index];
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' => depth = depth.saturating_add(1),
            b'}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(text[start..=index].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

#[derive(Clone, Debug, PartialEq)]
struct YoutubeMediaCandidate {
    url: String,
    mime_type: String,
    bitrate: Option<u64>,
    content_length: Option<u64>,
    duration_seconds: Option<f64>,
    score: u64,
}

fn select_youtube_public_media_link_from_player_response(
    spec: &MusicUrlSourceSpec,
    response: &serde_json::Value,
) -> Result<ResolvedMusicMediaLink, String> {
    let mut candidates = Vec::new();
    let mut ciphered_media_count = 0usize;
    if let Some(streaming_data) = response.get("streamingData") {
        for key in ["formats", "adaptiveFormats"] {
            if let Some(formats) = streaming_data
                .get(key)
                .and_then(serde_json::Value::as_array)
            {
                for format in formats {
                    if !youtube_format_has_audio(format) {
                        continue;
                    }
                    if let Some(url) = format.get("url").and_then(serde_json::Value::as_str) {
                        if let Some(candidate) =
                            youtube_media_candidate_from_format(format, response, url)
                        {
                            candidates.push(candidate);
                        }
                    } else if format.get("signatureCipher").is_some()
                        || format.get("cipher").is_some()
                    {
                        ciphered_media_count += 1;
                    }
                }
            }
        }
    }
    candidates.sort_by(|left, right| right.score.cmp(&left.score));
    if let Some(candidate) = candidates.into_iter().next() {
        return Ok(ResolvedMusicMediaLink {
            source_url: spec.raw_url.clone(),
            media_url: candidate.url,
            source_kind: MusicUrlSourceKind::YouTube,
            extractor: "rust-youtube-player-response".to_string(),
            mime_type: candidate.mime_type,
            bitrate: candidate.bitrate,
            content_length: candidate.content_length,
            duration_seconds: candidate.duration_seconds,
        });
    }

    let playability = response
        .pointer("/playabilityStatus/reason")
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            response
                .pointer("/playabilityStatus/status")
                .and_then(serde_json::Value::as_str)
        });
    if ciphered_media_count > 0 {
        return Err(format!(
            "YouTube returned {ciphered_media_count} signature-protected media URL(s); Rust resolver needs player signature deciphering for this video"
        ));
    }
    if let Some(reason) = playability {
        Err(format!(
            "YouTube player response did not expose a direct media URL: {reason}"
        ))
    } else {
        Err("YouTube player response did not expose a direct audio/video URL".to_string())
    }
}

fn youtube_format_has_audio(format: &serde_json::Value) -> bool {
    let mime = format
        .get("mimeType")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_ascii_lowercase();
    mime.starts_with("audio/")
        || mime.contains("mp4a")
        || mime.contains("opus")
        || format.get("audioQuality").is_some()
}

fn youtube_media_candidate_from_format(
    format: &serde_json::Value,
    response: &serde_json::Value,
    url: &str,
) -> Option<YoutubeMediaCandidate> {
    ParsedMusicUrl::parse(url).ok()?;
    let mime_type = format
        .get("mimeType")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("application/octet-stream")
        .to_string();
    if !is_audio_or_video_content_type(&mime_type) {
        return None;
    }
    let lower = mime_type.to_ascii_lowercase();
    let audio_only = lower.starts_with("audio/");
    let progressive_with_audio =
        lower.starts_with("video/") && (lower.contains("mp4a") || lower.contains("opus"));
    if !audio_only && !progressive_with_audio {
        return None;
    }
    let bitrate = u64_json_field(format, "bitrate");
    let content_length = u64_json_field(format, "contentLength").or_else(|| {
        query_value_from_url_path(
            &format!("?{}", url.split('?').nth(1).unwrap_or_default()),
            "clen",
        )
        .and_then(|value| value.parse::<u64>().ok())
    });
    let duration_seconds = youtube_format_duration_seconds(format).or_else(|| {
        response
            .pointer("/videoDetails/lengthSeconds")
            .and_then(f64_from_json_value)
    });
    let container_bonus = if lower.contains("mp4") {
        80
    } else if lower.contains("webm") {
        50
    } else {
        10
    };
    let score =
        if audio_only { 10_000 } else { 5_000 } + container_bonus + bitrate.unwrap_or(0) / 1_000;
    Some(YoutubeMediaCandidate {
        url: url.to_string(),
        mime_type,
        bitrate,
        content_length,
        duration_seconds,
        score,
    })
}

fn youtube_format_duration_seconds(format: &serde_json::Value) -> Option<f64> {
    format
        .get("approxDurationMs")
        .and_then(f64_from_json_value)
        .map(|millis| millis / 1000.0)
}

fn u64_json_field(value: &serde_json::Value, key: &str) -> Option<u64> {
    value.get(key).and_then(|value| {
        value
            .as_u64()
            .or_else(|| value.as_str().and_then(|raw| raw.parse::<u64>().ok()))
    })
}

fn f64_from_json_value(value: &serde_json::Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|raw| raw.parse::<f64>().ok()))
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MusicHttpUrl {
    raw_url: String,
    scheme: String,
    host: String,
    port: u16,
    path: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MusicHttpResponse {
    final_url: String,
    status_code: u16,
    status_text: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    truncated: bool,
}

fn fetch_public_music_http_text(url: &str, max_bytes: usize) -> Result<String, String> {
    let response = fetch_public_music_http_bytes(url, max_bytes, None, false)?;
    if response.truncated {
        return Err(format!(
            "HTTP response from {} exceeded {} bytes",
            response.final_url, max_bytes
        ));
    }
    String::from_utf8(response.body).map_err(|_| {
        format!(
            "HTTP response from {} was not valid UTF-8 text",
            response.final_url
        )
    })
}

fn fetch_public_music_http_bytes(
    url: &str,
    max_body_bytes: usize,
    range: Option<(u64, u64)>,
    allow_truncated: bool,
) -> Result<MusicHttpResponse, String> {
    let mut current_url = url.trim().to_string();
    for _ in 0..=MUSIC_MEDIA_HTTP_REDIRECT_LIMIT {
        let response = fetch_public_music_http_once(&current_url, max_body_bytes, range)?;
        if (300..400).contains(&response.status_code) {
            let Some(location) = music_http_header_value(&response.headers, "location") else {
                return Err(format!(
                    "HTTP redirect from {} did not include Location",
                    response.final_url
                ));
            };
            current_url = resolve_music_http_redirect(&response.final_url, &location)?;
            continue;
        }
        if !(200..300).contains(&response.status_code) {
            return Err(format!(
                "HTTP GET {} returned {} {}",
                response.final_url, response.status_code, response.status_text
            ));
        }
        if response.truncated && !allow_truncated {
            return Err(format!(
                "HTTP response from {} exceeded {} bytes",
                response.final_url, max_body_bytes
            ));
        }
        return Ok(response);
    }
    Err(format!(
        "HTTP GET {url} exceeded {MUSIC_MEDIA_HTTP_REDIRECT_LIMIT} redirects"
    ))
}

fn fetch_public_music_http_once(
    url: &str,
    max_body_bytes: usize,
    range: Option<(u64, u64)>,
) -> Result<MusicHttpResponse, String> {
    let endpoint = parse_music_http_url(url)?;
    let timeout = Duration::from_millis(MUSIC_MEDIA_HTTP_TIMEOUT_MS);
    let address = (endpoint.host.as_str(), endpoint.port)
        .to_socket_addrs()
        .map_err(|err| format!("could not resolve {}: {err}", endpoint.host))?
        .next()
        .ok_or_else(|| format!("could not resolve {}", endpoint.host))?;
    let stream = TcpStream::connect_timeout(&address, timeout)
        .map_err(|err| format!("could not connect to {}: {err}", endpoint.raw_url))?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|err| format!("could not set HTTP read timeout: {err}"))?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|err| format!("could not set HTTP write timeout: {err}"))?;
    let request = music_http_get_request(&endpoint, range);
    let (status_code, status_text, headers, body, truncated) = if endpoint.scheme == "https" {
        let connector = TlsConnector::new()
            .map_err(|err| format!("could not initialize TLS connector: {err}"))?;
        let mut tls_stream = connector
            .connect(&endpoint.host, stream)
            .map_err(|err| format!("could not negotiate TLS with {}: {err}", endpoint.host))?;
        write_music_http_request_and_read_response(&mut tls_stream, &request, max_body_bytes)?
    } else {
        let mut stream = stream;
        write_music_http_request_and_read_response(&mut stream, &request, max_body_bytes)?
    };
    Ok(MusicHttpResponse {
        final_url: endpoint.raw_url,
        status_code,
        status_text,
        headers,
        body,
        truncated,
    })
}

fn parse_music_http_url(raw_url: &str) -> Result<MusicHttpUrl, String> {
    ParsedMusicUrl::parse(raw_url)?;
    let trimmed = raw_url.trim();
    let (scheme, rest) = trimmed
        .split_once("://")
        .ok_or_else(|| "HTTP URL must include a scheme".to_string())?;
    let scheme = scheme.to_ascii_lowercase();
    if !matches!(scheme.as_str(), "http" | "https") {
        return Err("HTTP URL must use http or https".to_string());
    }
    let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let authority = &rest[..authority_end];
    let host = extract_url_host(authority)?;
    let port = url_authority_port(authority).unwrap_or(if scheme == "https" { 443 } else { 80 });
    let raw_path = &rest[authority_end..];
    let path = if raw_path.is_empty() {
        "/".to_string()
    } else if raw_path.starts_with('/') {
        raw_path.split('#').next().unwrap_or(raw_path).to_string()
    } else {
        format!("/{}", raw_path.split('#').next().unwrap_or(raw_path))
    };
    Ok(MusicHttpUrl {
        raw_url: trimmed.to_string(),
        scheme,
        host,
        port,
        path,
    })
}

fn url_authority_port(authority: &str) -> Option<u16> {
    if let Some(rest) = authority.strip_prefix('[') {
        let (_, suffix) = rest.split_once(']')?;
        return suffix.strip_prefix(':')?.parse::<u16>().ok();
    }
    let (_, port) = authority.rsplit_once(':')?;
    if port.bytes().all(|byte| byte.is_ascii_digit()) {
        port.parse::<u16>().ok()
    } else {
        None
    }
}

fn music_http_get_request(endpoint: &MusicHttpUrl, range: Option<(u64, u64)>) -> String {
    let host_header = if (endpoint.scheme == "https" && endpoint.port == 443)
        || (endpoint.scheme == "http" && endpoint.port == 80)
    {
        endpoint.host.clone()
    } else {
        format!("{}:{}", endpoint.host, endpoint.port)
    };
    let range_header = range
        .map(|(start, end)| format!("Range: bytes={start}-{end}\r\n"))
        .unwrap_or_default();
    format!(
        "GET {} HTTP/1.1\r\n\
         Host: {}\r\n\
         User-Agent: {}\r\n\
         Accept: */*\r\n\
         Accept-Encoding: identity\r\n\
         {}\
         Connection: close\r\n\
         \r\n",
        endpoint.path, host_header, MUSIC_MEDIA_USER_AGENT, range_header
    )
}

type MusicHttpReadResult = (u16, String, Vec<(String, String)>, Vec<u8>, bool);

fn write_music_http_request_and_read_response<S: Read + Write>(
    stream: &mut S,
    request: &str,
    max_body_bytes: usize,
) -> Result<MusicHttpReadResult, String> {
    stream
        .write_all(request.as_bytes())
        .map_err(|err| format!("could not write HTTP request: {err}"))?;
    stream
        .flush()
        .map_err(|err| format!("could not flush HTTP request: {err}"))?;
    read_music_http_response(stream, max_body_bytes)
}

fn read_music_http_response<S: Read>(
    stream: &mut S,
    max_body_bytes: usize,
) -> Result<MusicHttpReadResult, String> {
    let mut data = Vec::new();
    let mut chunk = [0u8; 8192];
    let mut header_end = None;
    let mut truncated = false;
    loop {
        let n = stream
            .read(&mut chunk)
            .map_err(|err| format!("could not read HTTP response: {err}"))?;
        if n == 0 {
            break;
        }
        data.extend_from_slice(&chunk[..n]);
        if header_end.is_none() {
            header_end = find_music_http_header_end(&data);
            if header_end.is_none() && data.len() > 64 * 1024 {
                return Err("HTTP response headers exceeded 64 KiB".to_string());
            }
        }
        if let Some(end) = header_end {
            let body_len = data.len().saturating_sub(end);
            if body_len > max_body_bytes {
                truncated = true;
                break;
            }
        }
    }
    let header_end =
        header_end.ok_or_else(|| "HTTP response did not include complete headers".to_string())?;
    let (status_code, status_text, headers) =
        parse_music_http_response_headers(&data[..header_end])?;
    let raw_body = data[header_end..].to_vec();
    let chunked = music_http_header_value(&headers, "transfer-encoding")
        .map(|value| value.to_ascii_lowercase().contains("chunked"))
        .unwrap_or(false);
    let (mut body, chunk_truncated) = if chunked {
        decode_music_chunked_body(&raw_body, max_body_bytes)?
    } else {
        (raw_body, false)
    };
    if body.len() > max_body_bytes {
        body.truncate(max_body_bytes);
        truncated = true;
    }
    Ok((
        status_code,
        status_text,
        headers,
        body,
        truncated || chunk_truncated,
    ))
}

fn parse_music_http_response_headers(
    header_bytes: &[u8],
) -> Result<(u16, String, Vec<(String, String)>), String> {
    let header_text = String::from_utf8_lossy(header_bytes);
    let mut lines = header_text.lines();
    let status_line = lines
        .next()
        .ok_or_else(|| "HTTP response missing status line".to_string())?;
    let mut status_parts = status_line.splitn(3, ' ');
    let version = status_parts.next().unwrap_or_default();
    if !version.starts_with("HTTP/") {
        return Err("HTTP response status line did not start with HTTP/".to_string());
    }
    let status_code = status_parts
        .next()
        .ok_or_else(|| "HTTP response missing status code".to_string())?
        .parse::<u16>()
        .map_err(|_| "HTTP response status code was not a number".to_string())?;
    let status_text = status_parts.next().unwrap_or_default().trim().to_string();
    let headers = lines
        .filter_map(|line| {
            let (key, value) = line.split_once(':')?;
            Some((key.trim().to_ascii_lowercase(), value.trim().to_string()))
        })
        .collect();
    Ok((status_code, status_text, headers))
}

fn decode_music_chunked_body(
    raw_body: &[u8],
    max_body_bytes: usize,
) -> Result<(Vec<u8>, bool), String> {
    let mut body = Vec::new();
    let mut index = 0usize;
    let mut truncated = false;
    while index < raw_body.len() {
        let Some(line_end) = find_music_bytes(&raw_body[index..], b"\r\n")
            .or_else(|| find_music_bytes(&raw_body[index..], b"\n"))
        else {
            truncated = true;
            break;
        };
        let absolute_line_end = index + line_end;
        let size_text = String::from_utf8_lossy(&raw_body[index..absolute_line_end]);
        let size_hex = size_text.split(';').next().unwrap_or_default().trim();
        let size = usize::from_str_radix(size_hex, 16)
            .map_err(|_| "invalid chunked HTTP response size".to_string())?;
        let newline_len = if raw_body
            .get(absolute_line_end..absolute_line_end + 2)
            .map(|bytes| bytes == b"\r\n")
            .unwrap_or(false)
        {
            2
        } else {
            1
        };
        index = absolute_line_end + newline_len;
        if size == 0 {
            break;
        }
        let available = raw_body.len().saturating_sub(index);
        let take = available.min(size);
        let remaining_capacity = max_body_bytes.saturating_sub(body.len());
        body.extend_from_slice(&raw_body[index..index + take.min(remaining_capacity)]);
        if take < size || body.len() >= max_body_bytes {
            truncated = true;
            break;
        }
        index += size;
        if raw_body
            .get(index..index + 2)
            .map(|bytes| bytes == b"\r\n")
            .unwrap_or(false)
        {
            index += 2;
        } else if raw_body.get(index).copied() == Some(b'\n') {
            index += 1;
        }
    }
    Ok((body, truncated))
}

fn find_music_http_header_end(data: &[u8]) -> Option<usize> {
    find_music_bytes(data, b"\r\n\r\n")
        .map(|index| index + 4)
        .or_else(|| find_music_bytes(data, b"\n\n").map(|index| index + 2))
}

fn find_music_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn music_http_header_value(headers: &[(String, String)], name: &str) -> Option<String> {
    let name = name.to_ascii_lowercase();
    headers
        .iter()
        .find_map(|(key, value)| (key == &name).then(|| value.clone()))
}

fn resolve_music_http_redirect(current_url: &str, location: &str) -> Result<String, String> {
    let location = location.trim();
    if location.starts_with("http://") || location.starts_with("https://") {
        return Ok(location.to_string());
    }
    let endpoint = parse_music_http_url(current_url)?;
    if let Some(rest) = location.strip_prefix("//") {
        return Ok(format!("{}://{rest}", endpoint.scheme));
    }
    let authority = if (endpoint.scheme == "https" && endpoint.port == 443)
        || (endpoint.scheme == "http" && endpoint.port == 80)
    {
        endpoint.host
    } else {
        format!("{}:{}", endpoint.host, endpoint.port)
    };
    if location.starts_with('/') {
        Ok(format!("{}://{}{}", endpoint.scheme, authority, location))
    } else {
        let base = endpoint
            .path
            .rsplit_once('/')
            .map(|(base, _)| if base.is_empty() { "/" } else { base })
            .unwrap_or("/");
        Ok(format!(
            "{}://{}{}/{}",
            endpoint.scheme,
            authority,
            base.trim_end_matches('/'),
            location
        ))
    }
}

fn is_audio_or_video_content_type(content_type: &str) -> bool {
    let lower = content_type.to_ascii_lowercase();
    lower.starts_with("audio/")
        || lower.starts_with("video/")
        || lower.contains("audio/")
        || lower.contains("video/")
}

fn compact_music_descriptor_value(value: &str, max_chars: usize) -> String {
    let compact = value
        .split_whitespace()
        .map(|part| {
            if part.starts_with("http://") || part.starts_with("https://") {
                "[media-url]"
            } else {
                part
            }
        })
        .collect::<Vec<_>>()
        .join(" ");
    if compact.chars().count() <= max_chars {
        return compact;
    }
    let mut out = compact.chars().take(max_chars.saturating_sub(3)).collect::<String>();
    out.push_str("...");
    out
}

pub fn song_spec_from_music_url_source(
    source: &MusicUrlSourceSpec,
    title: impl Into<String>,
    duration_seconds: f64,
) -> SongSpec {
    song_spec_from_music_url_source_with_prompt(source, title, duration_seconds, None)
}

pub fn song_spec_from_music_url_source_with_prompt(
    source: &MusicUrlSourceSpec,
    title: impl Into<String>,
    duration_seconds: f64,
    prompt: Option<&str>,
) -> SongSpec {
    let sample = derive_music_sample_seed_from_url_source(source);
    song_spec_from_music_sample_seed_with_prompt(&sample, title, duration_seconds, prompt)
}

pub fn render_music_url_seed_wav(
    fields: &[(&str, &str)],
    out_path: impl AsRef<Path>,
) -> Result<MusicUrlSeedRenderResult, String> {
    let selected_source = select_music_url_input(fields)?
        .ok_or_else(|| "music URL seed form needs at least one source URL".to_string())?;
    let duration_seconds = music_form_duration_seconds(fields, DEFAULT_SONG_SECONDS)?;
    let title = nonempty_music_form_value(fields, "title")
        .or_else(|| nonempty_music_form_value(fields, "music_url_title"))
        .unwrap_or("music-url-source variation");
    let prompt = nonempty_music_form_value(fields, "prompt")
        .or_else(|| nonempty_music_form_value(fields, "music_url_prompt"));

    let sample_seed = if selected_source.spec.kind == MusicUrlSourceKind::YouTube {
        match derive_music_sample_seed_from_public_media_link_source(&selected_source.spec) {
            Ok(sample_seed) => sample_seed,
            Err(err) => {
                let mut fallback = derive_music_sample_seed_from_url_source(&selected_source.spec);
                fallback
                    .descriptors
                    .push("media-download-fallback=url-only".to_string());
                fallback.descriptors.push(format!(
                    "media-download-error={}",
                    compact_music_descriptor_value(&err, 180)
                ));
                fallback
            }
        }
    } else {
        derive_music_sample_seed_from_url_source(&selected_source.spec)
    };
    let spec =
        song_spec_from_music_sample_seed_with_prompt(&sample_seed, title, duration_seconds, prompt);
    let render = generate_microtonal_song(spec);
    let out_path = out_path.as_ref();
    if let Some(parent) = out_path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    render
        .audio
        .write_wav16(out_path)
        .map_err(|err| format!("failed to write {}: {err}", out_path.display()))?;
    let wav_bytes = std::fs::metadata(out_path)
        .map_err(|err| format!("failed to inspect {}: {err}", out_path.display()))?
        .len();
    Ok(MusicUrlSeedRenderResult {
        selected_source,
        sample_seed,
        output_path: canonical_or_original(out_path.to_path_buf()),
        wav_bytes,
        summary: render.summary,
    })
}

pub fn music_url_seed_render_result_json(
    result: &MusicUrlSeedRenderResult,
    wav_url: &str,
) -> String {
    format!(
        concat!(
            "{{",
            "\"ok\":true,",
            "\"wav_url\":\"{wav_url}\",",
            "\"source_url\":\"{source_url}\",",
            "\"source_input_field\":\"{source_input_field}\",",
            "\"submitted_field\":\"{submitted_field}\",",
            "\"source_kind\":\"{source_kind}\",",
            "\"host\":\"{host}\",",
            "\"downloader\":\"{downloader}\",",
            "\"direct_media_hint\":{direct_media_hint},",
            "\"seed\":{seed},",
            "\"title\":\"{title}\",",
            "\"genre\":\"{genre}\",",
            "\"duration_seconds\":{duration_seconds:.3},",
            "\"bpm\":{bpm:.3},",
            "\"wav_bytes\":{wav_bytes}",
            "}}"
        ),
        wav_url = json_escape(wav_url),
        source_url = json_escape(&result.selected_source.spec.raw_url),
        source_input_field = json_escape(&result.selected_source.spec.input_field_id),
        submitted_field = json_escape(&result.selected_source.submitted_field_id),
        source_kind = json_escape(result.selected_source.spec.kind.as_str()),
        host = json_escape(&result.selected_source.spec.host),
        downloader = json_escape(&result.selected_source.spec.downloader_hint),
        direct_media_hint = result.selected_source.spec.direct_media_hint,
        seed = result.sample_seed.seed,
        title = json_escape(&result.summary.title),
        genre = json_escape(result.summary.genre.as_str()),
        duration_seconds = result.summary.duration_seconds,
        bpm = result.summary.bpm,
        wav_bytes = result.wav_bytes,
    )
}

pub fn music_url_seed_error_json(error_code: &str, error: &str) -> String {
    format!(
        "{{\"ok\":false,\"error_code\":\"{}\",\"error\":\"{}\"}}",
        json_escape(error_code),
        json_escape(error)
    )
}

pub fn song_spec_from_music_sample_seed(
    sample: &MusicSampleSeed,
    title: impl Into<String>,
    duration_seconds: f64,
) -> SongSpec {
    song_spec_from_music_sample_seed_with_prompt(sample, title, duration_seconds, None)
}

pub fn song_spec_from_music_sample_seed_with_prompt(
    sample: &MusicSampleSeed,
    title: impl Into<String>,
    duration_seconds: f64,
    prompt: Option<&str>,
) -> SongSpec {
    let title = title.into();
    let prompt_influence = prompt.and_then(analyze_music_sample_prompt);
    let genre = prompt_influence
        .as_ref()
        .and_then(|influence| influence.genre)
        .unwrap_or(sample.suggested_genre);
    let seed = prompt_influence
        .as_ref()
        .map(|influence| sample.seed ^ influence.prompt_hash.rotate_left(9))
        .unwrap_or(sample.seed);
    let mut structure_plan =
        generate_track_structure_plan(title.clone(), genre, duration_seconds, seed);
    let mut key_bias_steps = sample.key_bias_steps;
    if let Some(influence) = &prompt_influence {
        key_bias_steps += influence.key_bias_delta;
    }
    for key_change in &mut structure_plan.key_changes {
        key_change.scale_step_delta += key_bias_steps.signum();
        key_change.target_offset_steps += key_bias_steps;
    }
    if let Some(first) = structure_plan.time_signature_changes.first_mut() {
        let meter_bias = prompt_influence
            .as_ref()
            .and_then(|influence| influence.meter_bias)
            .unwrap_or(sample.meter_bias);
        first.numerator = meter_bias.0;
        first.denominator = meter_bias.1;
    }
    push_unique(&mut structure_plan.italian_features, "campionamento");
    if let Some(influence) = &prompt_influence {
        push_unique(&mut structure_plan.italian_features, "direzione");
        for tag in &influence.feature_tags {
            push_unique(
                &mut structure_plan.italian_features,
                format!("prompt-{tag}"),
            );
        }
    }
    let bpm = if let Some(influence) = &prompt_influence {
        let source_offset = sample.suggested_bpm - sample.suggested_genre.default_bpm();
        (genre.default_bpm() + source_offset * 0.5 + influence.bpm_delta).clamp(60.0, 194.0)
    } else {
        sample.suggested_bpm
    };
    SongSpec {
        title,
        genre,
        duration_seconds,
        bpm,
        seed,
        structure_plan: Some(structure_plan),
        ..Default::default()
    }
}

pub fn generate_microtonal_song(spec: SongSpec) -> SongRender {
    require_finite("duration_seconds", spec.duration_seconds);
    require_finite("bpm", spec.bpm);
    assert!(
        spec.duration_seconds > 0.0,
        "duration_seconds must be positive"
    );
    assert!(spec.bpm > 0.0, "bpm must be positive");
    spec.sample_manifest
        .validate_legal_sources()
        .expect("sample manifest must declare legal/open sources");

    let mut rng = mulberry32(spec.seed);
    let mut audio = AudioBuffer::silence(spec.sample_rate, spec.duration_seconds);
    let beat = 60.0 / spec.bpm;
    let sections = build_sections(spec.duration_seconds);
    let structure_plan = spec.structure_plan.clone().unwrap_or_else(|| {
        generate_track_structure_plan(
            spec.title.clone(),
            spec.genre,
            spec.duration_seconds,
            spec.seed,
        )
    });
    let instrument_palette = spec.instrument_palette.clone().unwrap_or_else(|| {
        discover_instrument_palette(InstrumentDiscoverySpec {
            seed: spec.seed ^ 0x517a_5eed,
            sample_rate: spec.sample_rate.min(16_000),
            candidates_per_role: 12,
            ..Default::default()
        })
    });
    let mut bass = instrument_palette
        .for_role(InstrumentRole::Bass)
        .expect("instrument palette should include a bass role")
        .clone();
    let mut lead = instrument_palette
        .for_role(InstrumentRole::Lead)
        .expect("instrument palette should include a lead role")
        .clone();
    let mut pad = instrument_palette
        .for_role(InstrumentRole::Pad)
        .expect("instrument palette should include a pad role")
        .clone();
    let percussion = instrument_palette
        .for_role(InstrumentRole::Percussion)
        .expect("instrument palette should include a percussion role")
        .clone();
    let mut texture = instrument_palette
        .for_role(InstrumentRole::Texture)
        .expect("instrument palette should include a texture role")
        .clone();
    bass.output_gain *= 1.06;
    lead.output_gain *= 1.22;
    pad.output_gain *= 1.12;
    texture.output_gain *= 1.15;
    let mut parts = Vec::new();
    let mut drum_variation = DrumVariationSummary::default();
    let mut rendered_events = 0usize;

    let rhythm_events = render_breakbeat(
        &mut audio,
        &spec.scale,
        &percussion,
        &structure_plan,
        spec.genre,
        beat,
        spec.duration_seconds,
        &mut rng,
        &mut drum_variation,
    );
    rendered_events += rhythm_events;
    parts.push(SongPartSummary {
        name: "synthetic break lattice".to_string(),
        role: InstrumentRole::Percussion,
        instrument: percussion.display_name.clone(),
        events: rhythm_events,
    });

    let bass_events = render_bassline(
        &mut audio,
        &spec.scale,
        &bass,
        &structure_plan,
        beat,
        spec.duration_seconds,
        &mut rng,
    );
    rendered_events += bass_events;
    parts.push(SongPartSummary {
        name: "kinetic bass thread".to_string(),
        role: InstrumentRole::Bass,
        instrument: bass.display_name.clone(),
        events: bass_events,
    });

    let main_events = render_main_melody(
        &mut audio,
        &spec.scale,
        &lead,
        &structure_plan,
        beat,
        spec.duration_seconds,
        &mut rng,
    );
    rendered_events += main_events;
    parts.push(SongPartSummary {
        name: "main melody".to_string(),
        role: InstrumentRole::Lead,
        instrument: lead.display_name.clone(),
        events: main_events,
    });

    let secondary_a_events = render_secondary_counterline(
        &mut audio,
        &spec.scale,
        &texture,
        &structure_plan,
        beat,
        spec.duration_seconds,
        &mut rng,
    );
    rendered_events += secondary_a_events;
    parts.push(SongPartSummary {
        name: "secondary part A".to_string(),
        role: InstrumentRole::Texture,
        instrument: texture.display_name.clone(),
        events: secondary_a_events,
    });

    let secondary_b_events = render_secondary_pulse(
        &mut audio,
        &spec.scale,
        &pad,
        &structure_plan,
        beat,
        spec.duration_seconds,
        &mut rng,
    );
    rendered_events += secondary_b_events;
    parts.push(SongPartSummary {
        name: "secondary part B".to_string(),
        role: InstrumentRole::Pad,
        instrument: pad.display_name.clone(),
        events: secondary_b_events,
    });

    rendered_events += render_generated_vocal_chops(
        &mut audio,
        &spec.scale,
        &structure_plan,
        beat,
        spec.duration_seconds,
        &mut rng,
    );

    let lowpass = BiquadFilter::new(FilterMode::LowPass, spec.sample_rate, 11_000.0, 0.707);
    apply_filter(&mut audio, lowpass);
    apply_soft_clip(&mut audio, 1.8, 0.58);
    apply_feedback_delay(
        &mut audio,
        DelayEffect {
            delay_seconds: beat * 0.375,
            feedback: 0.28,
            mix: 0.14,
        },
    );
    apply_schroeder_reverb(
        &mut audio,
        ReverbEffect {
            room_size: 0.48,
            damping: 0.35,
            mix: 0.12,
        },
    );
    apply_pauses(&mut audio, &structure_plan.pauses);
    audio.normalize_peak(0.92);

    let fft_len = 2048usize.min(audio.samples.len().next_power_of_two());
    let start = audio.samples.len().saturating_sub(fft_len) / 2;
    let centroid = if fft_len.is_power_of_two() && start + fft_len <= audio.samples.len() {
        let bins = analyze_fft(
            &audio.samples[start..start + fft_len],
            audio.sample_rate,
            WindowFunction::Hann,
        );
        spectral_centroid(&bins)
    } else {
        0.0
    };

    let summary = ArrangementSummary {
        title: structure_plan.title.clone(),
        genre: structure_plan.genre,
        duration_seconds: audio.duration_seconds(),
        bpm: spec.bpm,
        scale_name: spec.scale.name.clone(),
        sections,
        parts,
        instruments: instrument_palette.names(),
        key_changes: structure_plan.key_changes.clone(),
        time_signature_changes: structure_plan.time_signature_changes.clone(),
        pauses: structure_plan.pauses.clone(),
        italian_features: structure_plan.italian_features.clone(),
        drum_variation,
        rendered_events,
        peak: audio.peak(),
        rms: audio.rms(),
        spectral_centroid_hz: centroid,
    };
    validate_song_structure(&summary).expect("generated song should satisfy structure constraints");

    SongRender {
        summary,
        sample_manifest: spec.sample_manifest,
        audio,
    }
}

pub fn render_song_wav(path: impl AsRef<Path>, spec: SongSpec) -> io::Result<ArrangementSummary> {
    let render = generate_microtonal_song(spec);
    render.audio.write_wav16(path)?;
    Ok(render.summary)
}

pub fn render_ten_song_album(
    out_dir: impl AsRef<Path>,
    seed: u32,
    duration_seconds: f64,
) -> io::Result<AlbumRenderSummary> {
    render_album_from_recipes(out_dir, seed, duration_seconds, &ten_song_recipes())
}

pub fn render_ten_more_song_album(
    out_dir: impl AsRef<Path>,
    seed: u32,
    duration_seconds: f64,
) -> io::Result<AlbumRenderSummary> {
    render_album_from_recipes(out_dir, seed, duration_seconds, &ten_more_song_recipes())
}

fn render_album_from_recipes(
    out_dir: impl AsRef<Path>,
    seed: u32,
    duration_seconds: f64,
    recipes: &[(&'static str, MusicGenre)],
) -> io::Result<AlbumRenderSummary> {
    let out_dir = out_dir.as_ref();
    std::fs::create_dir_all(out_dir)?;
    let mut tracks = Vec::new();
    for (i, (title, genre)) in recipes.iter().enumerate() {
        let track_seed = seed.wrapping_add((i as u32 + 1).wrapping_mul(7_919));
        let spec = SongSpec {
            title: (*title).to_string(),
            genre: *genre,
            duration_seconds,
            bpm: genre.default_bpm(),
            seed: track_seed,
            ..Default::default()
        };
        let render = generate_microtonal_song(spec);
        validate_song_structure(&render.summary)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let file_name = format!("{:02}-{}.wav", i + 1, slugify_ascii(title));
        let path = out_dir.join(file_name);
        render.audio.write_wav16(&path)?;
        let path = canonical_or_original(path);
        tracks.push(AlbumTrackSummary {
            index: i + 1,
            path: path.display().to_string(),
            summary: render.summary,
        });
    }

    let manifest_path = canonical_or_original(out_dir.join("album-manifest.json"));
    let out_dir = canonical_or_original(PathBuf::from(out_dir));
    let mut summary = AlbumRenderSummary {
        out_dir: out_dir.display().to_string(),
        manifest_path: manifest_path.display().to_string(),
        tracks,
    };
    std::fs::write(&summary.manifest_path, album_manifest_json(&summary))?;
    summary.manifest_path = canonical_or_original(PathBuf::from(&summary.manifest_path))
        .display()
        .to_string();
    Ok(summary)
}

pub fn validate_song_structure(summary: &ArrangementSummary) -> Result<(), String> {
    if summary.key_changes.is_empty() && summary.time_signature_changes.is_empty() {
        return Err(format!(
            "{} must include at least one key or time-signature change",
            summary.title
        ));
    }
    if summary.key_changes.len() > 2 {
        return Err(format!(
            "{} has {} key changes; maximum is 2",
            summary.title,
            summary.key_changes.len()
        ));
    }
    if summary.time_signature_changes.len() > 3 {
        return Err(format!(
            "{} has {} time-signature changes; maximum is 3",
            summary.title,
            summary.time_signature_changes.len()
        ));
    }
    if summary.pauses.is_empty() {
        return Err(format!("{} must include at least one pausa", summary.title));
    }
    if summary.drum_variation.pattern_names.len() < 2 {
        return Err(format!(
            "{} needs at least two drum patterns/variations",
            summary.title
        ));
    }
    if summary.drum_variation.fills == 0 && summary.drum_variation.syncopations == 0 {
        return Err(format!(
            "{} needs fills or syncopation in the drum part",
            summary.title
        ));
    }
    if summary.drum_variation.variation_ratio() < summary.drum_variation.repetition_reduction_target
    {
        return Err(format!(
            "{} drum variation ratio {:.3} is below target {:.3}",
            summary.title,
            summary.drum_variation.variation_ratio(),
            summary.drum_variation.repetition_reduction_target
        ));
    }
    for needed in ["pausa", "modulazione", "metro", "sincope", "spettro"] {
        if !summary
            .italian_features
            .iter()
            .any(|feature| feature == needed)
        {
            return Err(format!(
                "{} is missing Italian feature marker {needed}",
                summary.title
            ));
        }
    }
    validate_italian_music_feature_coverage(&summary.italian_features)
        .map_err(|err| format!("{} {err}", summary.title))?;
    Ok(())
}

fn ten_song_recipes() -> [(&'static str, MusicGenre); 10] {
    [
        ("Ritmo Delta", MusicGenre::DrumAndBass),
        ("Pausa Circuit", MusicGenre::House),
        ("Modulazione Glass", MusicGenre::Trance),
        ("Sincope Index", MusicGenre::Electronica),
        ("Spettro Steps", MusicGenre::Jazz),
        ("Misura Bloom", MusicGenre::Breakbeat),
        ("Metro Liquid", MusicGenre::LiquidFunk),
        ("Timbro Lattice", MusicGenre::Idm),
        ("Eco Dub Grid", MusicGenre::DubTechno),
        ("Contrappunto Garage", MusicGenre::FutureGarage),
    ]
}

fn ten_more_song_recipes() -> [(&'static str, MusicGenre); 10] {
    [
        ("Break Pressure Atlas", MusicGenre::Breakbeat),
        ("Rupture Melody Engine", MusicGenre::DrumAndBass),
        ("Frontier Amen Collage", MusicGenre::Jungle),
        ("Siren Bassline Drift", MusicGenre::LiquidFunk),
        ("Sub Harmonic Sprint", MusicGenre::Neurofunk),
        ("Stutter Signal Bloom", MusicGenre::Breakbeat),
        ("Massive Lattice Run", MusicGenre::DrumAndBass),
        ("Afterimage Break Grid", MusicGenre::FutureGarage),
        ("Cassette Velocity", MusicGenre::GlitchHop),
        ("Amber Dub Breaks", MusicGenre::Dubstep),
    ]
}

fn canonical_or_original(path: PathBuf) -> PathBuf {
    std::fs::canonicalize(&path).unwrap_or(path)
}

fn slugify_ascii(value: &str) -> String {
    let mut out = String::new();
    let mut last_dash = false;
    for c in value.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() {
        "track".to_string()
    } else {
        out
    }
}

fn album_manifest_json(album: &AlbumRenderSummary) -> String {
    let mut lines = Vec::new();
    lines.push("{".to_string());
    lines.push(format!(
        "  \"out_dir\": \"{}\",",
        json_escape(&album.out_dir)
    ));
    lines.push(format!(
        "  \"manifest_path\": \"{}\",",
        json_escape(&album.manifest_path)
    ));
    lines.push("  \"tracks\": [".to_string());
    for (i, track) in album.tracks.iter().enumerate() {
        let comma = if i + 1 == album.tracks.len() { "" } else { "," };
        lines.push("    {".to_string());
        lines.push(format!("      \"index\": {},", track.index));
        lines.push(format!("      \"path\": \"{}\",", json_escape(&track.path)));
        lines.push(format!(
            "      \"title\": \"{}\",",
            json_escape(&track.summary.title)
        ));
        lines.push(format!(
            "      \"genre\": \"{}\",",
            json_escape(track.summary.genre.as_str())
        ));
        lines.push(format!(
            "      \"duration_seconds\": {:.3},",
            track.summary.duration_seconds
        ));
        lines.push(format!("      \"bpm\": {:.3},", track.summary.bpm));
        lines.push(format!(
            "      \"key_changes\": {},",
            track.summary.key_changes.len()
        ));
        lines.push(format!(
            "      \"time_signature_changes\": {},",
            track.summary.time_signature_changes.len()
        ));
        lines.push(format!("      \"pauses\": {},", track.summary.pauses.len()));
        lines.push(format!(
            "      \"drum_patterns\": {},",
            track.summary.drum_variation.pattern_names.len()
        ));
        lines.push(format!(
            "      \"drum_fills\": {},",
            track.summary.drum_variation.fills
        ));
        lines.push(format!(
            "      \"drum_micro_variations\": {},",
            track.summary.drum_variation.micro_variations
        ));
        lines.push(format!(
            "      \"drum_variation_ratio\": {:.4},",
            track.summary.drum_variation.variation_ratio()
        ));
        lines.push(format!(
            "      \"percussion_gain\": {:.3},",
            track.summary.drum_variation.percussion_gain
        ));
        lines.push(format!(
            "      \"italian_features\": [{}]",
            track
                .summary
                .italian_features
                .iter()
                .map(|feature| format!("\"{}\"", json_escape(feature)))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        lines.push(format!("    }}{comma}"));
    }
    lines.push("  ]".to_string());
    lines.push("}".to_string());
    lines.join("\n")
}

fn json_escape(value: &str) -> String {
    let mut out = String::new();
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn parse_mp4_duration_seconds(bytes: &[u8]) -> Option<f64> {
    find_mvhd_duration_in_boxes(bytes, 0, bytes.len(), 0)
}

fn find_mvhd_duration_in_boxes(
    bytes: &[u8],
    start: usize,
    end: usize,
    depth: usize,
) -> Option<f64> {
    if depth > 8 || start > end || end > bytes.len() {
        return None;
    }
    let mut pos = start;
    while pos.checked_add(8)? <= end {
        let size32 = u32::from_be_bytes(bytes[pos..pos + 4].try_into().ok()?) as u64;
        let box_type: [u8; 4] = bytes[pos + 4..pos + 8].try_into().ok()?;
        let mut header_len = 8usize;
        let box_end = if size32 == 1 {
            if pos.checked_add(16)? > end {
                return None;
            }
            header_len = 16;
            let size64 = u64::from_be_bytes(bytes[pos + 8..pos + 16].try_into().ok()?);
            pos.checked_add(usize::try_from(size64).ok()?)?
        } else if size32 == 0 {
            end
        } else {
            pos.checked_add(usize::try_from(size32).ok()?)?
        };
        if box_end > end || box_end < pos + header_len {
            return None;
        }
        let content_start = pos + header_len;
        if &box_type == b"mvhd" {
            if let Some(duration) = parse_mvhd_duration_seconds(&bytes[content_start..box_end]) {
                return Some(duration);
            }
        } else if is_mp4_container_box(&box_type) {
            if let Some(duration) =
                find_mvhd_duration_in_boxes(bytes, content_start, box_end, depth + 1)
            {
                return Some(duration);
            }
        }
        if box_end == pos {
            return None;
        }
        pos = box_end;
    }
    None
}

fn is_mp4_container_box(box_type: &[u8; 4]) -> bool {
    matches!(
        box_type,
        b"moov"
            | b"trak"
            | b"mdia"
            | b"minf"
            | b"stbl"
            | b"edts"
            | b"meta"
            | b"udta"
            | b"moof"
            | b"traf"
    )
}

fn parse_mvhd_duration_seconds(content: &[u8]) -> Option<f64> {
    let version = *content.first()?;
    match version {
        0 => {
            if content.len() < 20 {
                return None;
            }
            let timescale = u32::from_be_bytes(content[12..16].try_into().ok()?);
            let duration = u32::from_be_bytes(content[16..20].try_into().ok()?);
            (timescale > 0).then_some(duration as f64 / timescale as f64)
        }
        1 => {
            if content.len() < 32 {
                return None;
            }
            let timescale = u32::from_be_bytes(content[20..24].try_into().ok()?);
            let duration = u64::from_be_bytes(content[24..32].try_into().ok()?);
            (timescale > 0).then_some(duration as f64 / timescale as f64)
        }
        _ => None,
    }
}

fn music_url_genre_hint(
    kind: MusicUrlSourceKind,
    seed: u32,
    direct_media_hint: bool,
) -> MusicGenre {
    let options: &[MusicGenre] = match kind {
        MusicUrlSourceKind::YouTube => &[
            MusicGenre::Breakbeat,
            MusicGenre::DrumAndBass,
            MusicGenre::FutureGarage,
            MusicGenre::TripHop,
        ],
        MusicUrlSourceKind::Facebook | MusicUrlSourceKind::Instagram => &[
            MusicGenre::Dance,
            MusicGenre::House,
            MusicGenre::Breakbeat,
            MusicGenre::GlitchHop,
        ],
        MusicUrlSourceKind::S3
        | MusicUrlSourceKind::CloudFront
        | MusicUrlSourceKind::Cloudflare
        | MusicUrlSourceKind::StaticAssetHost => {
            if direct_media_hint {
                &[
                    MusicGenre::DrumAndBass,
                    MusicGenre::Dubstep,
                    MusicGenre::FutureGarage,
                    MusicGenre::Breakbeat,
                ]
            } else {
                &[
                    MusicGenre::Electronica,
                    MusicGenre::AmbientTechno,
                    MusicGenre::Idm,
                    MusicGenre::Downtempo,
                ]
            }
        }
        MusicUrlSourceKind::DirectAudio => &[
            MusicGenre::Breakbeat,
            MusicGenre::DrumAndBass,
            MusicGenre::TripHop,
            MusicGenre::DubTechno,
        ],
        MusicUrlSourceKind::DirectVideo => &[
            MusicGenre::Jungle,
            MusicGenre::Breakcore,
            MusicGenre::GlitchHop,
            MusicGenre::FutureGarage,
        ],
        MusicUrlSourceKind::OtherUrl => &[
            MusicGenre::ExperimentalElectronic,
            MusicGenre::PostMinimalElectronic,
            MusicGenre::Soundscape,
            MusicGenre::Downtempo,
        ],
    };
    options[(seed as usize) % options.len()]
}

fn hash_bytes_to_seed(bytes: &[u8]) -> u32 {
    let mut hash = 0x811c_9dc5u32;
    let stride = (bytes.len() / 65_536).max(1);
    for &byte in bytes.iter().step_by(stride) {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

fn byte_entropy(bytes: &[u8]) -> f64 {
    if bytes.is_empty() {
        return 0.0;
    }
    let mut counts = [0usize; 256];
    let stride = (bytes.len() / 262_144).max(1);
    let mut n = 0usize;
    for &byte in bytes.iter().step_by(stride) {
        counts[byte as usize] += 1;
        n += 1;
    }
    counts
        .iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f64 / n as f64;
            -p * p.log2()
        })
        .sum()
}

fn build_sections(duration: f64) -> Vec<SongSection> {
    let marks = [
        ("intro", 0.0, 0.16),
        ("pressure", 0.16, 0.38),
        ("collage", 0.38, 0.66),
        ("swerve", 0.66, 0.86),
        ("outro", 0.86, 1.0),
    ];
    marks
        .iter()
        .map(|(name, a, b)| SongSection {
            name: (*name).to_string(),
            start_seconds: duration * a,
            end_seconds: duration * b,
        })
        .collect()
}

fn current_key_offset(plan: &TrackStructurePlan, time_seconds: f64) -> i32 {
    plan.key_changes
        .iter()
        .filter(|change| time_seconds >= change.time_seconds)
        .map(|change| change.target_offset_steps)
        .last()
        .unwrap_or(0)
}

fn transpose_degree(plan: &TrackStructurePlan, time_seconds: f64, degree: i32) -> i32 {
    degree + current_key_offset(plan, time_seconds)
}

fn current_time_signature(plan: &TrackStructurePlan, time_seconds: f64) -> (u8, u8) {
    plan.time_signature_changes
        .iter()
        .filter(|change| time_seconds >= change.time_seconds)
        .map(|change| (change.numerator, change.denominator))
        .last()
        .unwrap_or((4, 4))
}

fn meter_steps_per_bar(numerator: u8, denominator: u8) -> usize {
    let sixteenth_units_per_beat = (16 / denominator.max(1) as usize).max(1);
    (numerator as usize * sixteenth_units_per_beat).max(4)
}

fn push_unique(values: &mut Vec<String>, value: impl Into<String>) {
    let value = value.into();
    if !values.iter().any(|v| v == &value) {
        values.push(value);
    }
}

fn apply_pauses(buffer: &mut AudioBuffer, pauses: &[PauseEvent]) {
    let sample_rate = buffer.sample_rate as f64;
    for pause in pauses {
        let start = buffer.sample_index(pause.start_seconds);
        let end = buffer
            .sample_index(pause.start_seconds + pause.duration_seconds)
            .min(buffer.len());
        let fade = (0.025 * sample_rate).round() as usize;
        for i in start..end {
            let pos = i - start;
            let remaining = end - i;
            let edge = pos.min(remaining).min(fade) as f64 / fade.max(1) as f64;
            let depth = pause.depth.clamp(0.0, 1.0);
            let gain = depth + (1.0 - depth) * (1.0 - edge);
            buffer.samples[i] *= gain;
        }
    }
}

fn render_breakbeat(
    buffer: &mut AudioBuffer,
    scale: &MicrotonalScale,
    percussion: &SyntheticInstrument,
    plan: &TrackStructurePlan,
    genre: MusicGenre,
    beat: f64,
    duration: f64,
    rng: &mut impl RandomSource,
    summary: &mut DrumVariationSummary,
) -> usize {
    let step = beat / 4.0;
    let steps = (duration / step).ceil() as usize;
    summary.drum_steps = steps;
    summary.percussion_gain = genre.percussion_gain();
    let mut events = 0usize;
    for i in 0..steps {
        let t = i as f64 * step;
        let (num, den) = current_time_signature(plan, t);
        let bar_steps = meter_steps_per_bar(num, den);
        let pos = i % bar_steps;
        let bar = i / bar_steps;
        let density = genre.drum_density();
        let drum_gain = summary.percussion_gain;
        let variation_probability = (0.10 + density * 0.16).min(0.32);
        let phrase_turn = bar % 8 == 7;
        let pattern_name = format!("{}-{}-{}", genre.as_str(), num, den);
        push_unique(&mut summary.pattern_names, pattern_name);
        if (num, den) != (4, 4) {
            summary.meter_sensitive_variations += 1;
            if pos == 0 {
                summary.syncopations += 1;
            }
        }
        let backbeat_a = (bar_steps / 4).max(1);
        let backbeat_b = (3 * bar_steps / 4).max(backbeat_a + 1);
        let kick = pos == 0
            || pos == (bar_steps * 3 / 8).max(1)
            || (pos == (bar_steps * 5 / 8).max(1) && bar % 4 != 1)
            || (phrase_turn && pos + 2 >= bar_steps && rng.next_float() < density);
        if kick {
            render_kick(buffer, t, (0.92 + 0.08 * rng.next_float()) * drum_gain);
            let note = NoteEvent::new(t, beat * 0.22, transpose_degree(plan, t, -9), 0)
                .with_velocity(0.36 * drum_gain)
                .with_bend(PitchBendCurve::swoop(80.0, -130.0));
            percussion.render_note(buffer, scale, &note, rng);
            events += 1;
        }
        if pos == backbeat_a || pos == backbeat_b {
            render_snare(buffer, t, (0.82 + 0.12 * rng.next_float()) * drum_gain, rng);
            let note = NoteEvent::new(t, beat * 0.18, transpose_degree(plan, t, 2), 1)
                .with_velocity(0.42 * drum_gain)
                .with_bend(PitchBendCurve::swoop(35.0, -40.0));
            percussion.render_note(buffer, scale, &note, rng);
            events += 1;
        }
        if pos % 2 == 0 && rng.next_float() < 0.4 + density * 0.5 {
            render_hat(buffer, t, (0.18 + 0.08 * rng.next_float()) * drum_gain, rng);
            let note = NoteEvent::new(t, beat * 0.09, transpose_degree(plan, t, 12), 2)
                .with_velocity(0.16 * drum_gain);
            percussion.render_note(buffer, scale, &note, rng);
            events += 1;
        }
        if (pos % 4 == 3 || (phrase_turn && pos + 1 == bar_steps)) && rng.next_float() < density {
            render_snare(buffer, t, 0.22 * drum_gain, rng);
            summary.ghost_hits += 1;
            summary.syncopations += usize::from(pos % 4 != 0);
            if phrase_turn {
                summary.fills += 1;
            }
            events += 1;
        }
        if i % 9 == 0 || rng.next_float() < variation_probability {
            let offset = step * (0.35 + 0.3 * rng.next_float());
            if t + offset < duration {
                match (i + bar) % 4 {
                    0 => render_hat(buffer, t + offset, 0.10 * drum_gain, rng),
                    1 => render_snare(buffer, t + offset, 0.11 * drum_gain, rng),
                    2 => render_kick(buffer, t + offset, 0.12 * drum_gain),
                    _ => {
                        let note = NoteEvent::new(
                            t + offset,
                            beat * 0.07,
                            transpose_degree(plan, t, 14),
                            2,
                        )
                        .with_velocity(0.12 * drum_gain)
                        .with_bend(PitchBendCurve::swoop(20.0, -35.0));
                        percussion.render_note(buffer, scale, &note, rng);
                    }
                }
                summary.micro_variations += 1;
                summary.syncopations += 1;
                events += 1;
            }
        }
        if phrase_turn && pos + 4 >= bar_steps && rng.next_float() < 0.55 + density * 0.25 {
            for k in 1..=2 {
                let fill_t = t + step * k as f64 * 0.32;
                if fill_t < duration {
                    render_snare(buffer, fill_t, (0.16 + 0.03 * k as f64) * drum_gain, rng);
                    summary.fills += 1;
                    summary.micro_variations += 1;
                    events += 1;
                }
            }
        }
    }
    events
}

fn render_bassline(
    buffer: &mut AudioBuffer,
    scale: &MicrotonalScale,
    instrument: &SyntheticInstrument,
    plan: &TrackStructurePlan,
    beat: f64,
    duration: f64,
    rng: &mut impl RandomSource,
) -> usize {
    let phrase = [0, 0, -2, 0, 3, 0, 5, -1, 0, 7, 5, 3, -2, 0, 3, -4];
    let note_len = beat * 0.46;
    let start = beat * 4.0;
    let mut events = 0usize;
    let mut t = start;
    let mut i = 0usize;
    while t < duration {
        let degree = transpose_degree(plan, t, phrase[i % phrase.len()]);
        let bend = if i % 8 == 6 {
            PitchBendCurve::from_points(vec![
                PitchBendPoint {
                    position: 0.0,
                    cents: -70.0,
                },
                PitchBendPoint {
                    position: 0.35,
                    cents: 24.0,
                },
                PitchBendPoint {
                    position: 1.0,
                    cents: 0.0,
                },
            ])
        } else if i % 5 == 0 {
            PitchBendCurve::swoop(18.0, -8.0)
        } else {
            PitchBendCurve::flat()
        };
        let velocity = 0.78 + 0.18 * rng.next_float();
        let note = NoteEvent::new(t, note_len, degree, 0)
            .with_velocity(velocity)
            .with_bend(bend);
        instrument.render_note(buffer, scale, &note, rng);
        events += 1;
        t += beat * 0.5;
        i += 1;
    }
    events
}

fn render_main_melody(
    buffer: &mut AudioBuffer,
    scale: &MicrotonalScale,
    instrument: &SyntheticInstrument,
    plan: &TrackStructurePlan,
    beat: f64,
    duration: f64,
    rng: &mut impl RandomSource,
) -> usize {
    let phrase = [7, 9, 10, 14, 12, 15, 17, 10, 9, 5, 7, 12, 19, 17];
    let mut events = 0usize;
    let mut t = duration * 0.16;
    let mut i = 0usize;
    while t < duration * 0.9 {
        let degree = transpose_degree(
            plan,
            t,
            phrase[(i + (rng.next_int(0, 3) as usize)) % phrase.len()],
        );
        let bend = if i % 3 == 0 {
            PitchBendCurve::from_points(vec![
                PitchBendPoint {
                    position: 0.0,
                    cents: -32.0,
                },
                PitchBendPoint {
                    position: 0.72,
                    cents: 46.0,
                },
                PitchBendPoint {
                    position: 1.0,
                    cents: 12.0,
                },
            ])
        } else {
            PitchBendCurve::flat()
        };
        let note = NoteEvent::new(t, beat * 0.62, degree, 2)
            .with_velocity(0.58 + 0.24 * rng.next_float())
            .with_bend(bend);
        instrument.render_note(buffer, scale, &note, rng);
        events += 1;
        t += beat * if i % 4 == 3 { 1.0 } else { 0.5 };
        i += 1;
    }
    events
}

fn render_secondary_counterline(
    buffer: &mut AudioBuffer,
    scale: &MicrotonalScale,
    instrument: &SyntheticInstrument,
    plan: &TrackStructurePlan,
    beat: f64,
    duration: f64,
    rng: &mut impl RandomSource,
) -> usize {
    let phrase = [14, 10, 12, 8, 15, 17, 12, 9, 19, 15];
    let mut events = 0usize;
    let mut t = duration * 0.26;
    let mut i = 0usize;
    while t < duration * 0.86 {
        let degree = transpose_degree(plan, t, phrase[i % phrase.len()]);
        let note = NoteEvent::new(t, beat * 0.34, degree, 2)
            .with_velocity(0.26 + 0.16 * rng.next_float())
            .with_bend(if i % 4 == 1 {
                PitchBendCurve::swoop(44.0, -28.0)
            } else {
                PitchBendCurve::flat()
            });
        instrument.render_note(buffer, scale, &note, rng);
        events += 1;
        t += beat * if i % 5 == 2 { 0.75 } else { 0.5 };
        i += 1;
    }
    events
}

fn render_secondary_pulse(
    buffer: &mut AudioBuffer,
    scale: &MicrotonalScale,
    instrument: &SyntheticInstrument,
    plan: &TrackStructurePlan,
    beat: f64,
    duration: f64,
    rng: &mut impl RandomSource,
) -> usize {
    let chords = [[0, 5, 9], [3, 7, 12], [-2, 5, 10], [0, 8, 14]];
    let chord_len = beat * 8.0;
    let mut t = 0.0;
    let mut events = 0usize;
    let mut chord_idx = 0usize;
    while t < duration {
        for &degree in &chords[chord_idx % chords.len()] {
            let degree = transpose_degree(plan, t, degree);
            let note = NoteEvent::new(t, chord_len.min(duration - t), degree, 2)
                .with_velocity(0.42 + 0.12 * rng.next_float())
                .with_bend(PitchBendCurve::swoop(
                    (rng.next_float() - 0.5) * 18.0,
                    (rng.next_float() - 0.5) * 18.0,
                ));
            instrument.render_note(buffer, scale, &note, rng);
            events += 1;
        }
        t += chord_len * 0.75;
        chord_idx += 1;
    }
    events
}

fn render_generated_vocal_chops(
    buffer: &mut AudioBuffer,
    scale: &MicrotonalScale,
    plan: &TrackStructurePlan,
    beat: f64,
    duration: f64,
    rng: &mut impl RandomSource,
) -> usize {
    let mut events = 0usize;
    let bar = beat * 4.0;
    let degrees = [12, 14, 10, 17, 9, 15];
    let mut t = bar * 4.0;
    let mut phrase = 0usize;
    while t < duration * 0.92 {
        for offset in [0.0, beat * 0.75, beat * 1.5, beat * 2.75] {
            let chop_t = t + offset;
            if chop_t >= duration {
                break;
            }
            let degree = transpose_degree(plan, chop_t, degrees[(phrase + events) % degrees.len()]);
            let frequency = scale.degree_to_frequency(degree, 2);
            render_vocal_formant_chop(
                buffer,
                chop_t,
                beat * (0.18 + 0.12 * rng.next_float()),
                frequency,
                phrase % 4,
            );
            events += 1;
        }
        t += bar * 2.0;
        phrase += 1;
    }
    events
}

fn render_kick(buffer: &mut AudioBuffer, start_seconds: f64, velocity: f64) {
    let start = buffer.sample_index(start_seconds);
    let len = (0.42 * buffer.sample_rate as f64) as usize;
    let mut phase = 0.0;
    for i in 0..len {
        let idx = start + i;
        if idx >= buffer.len() {
            break;
        }
        let t = i as f64 / buffer.sample_rate as f64;
        let freq = 44.0 + 118.0 * (-t * 18.0).exp();
        phase = (phase + TAU * freq / buffer.sample_rate as f64).rem_euclid(TAU);
        let body = phase.sin() * (-t * 7.5).exp();
        let click = if t < 0.012 {
            (1.0 - t / 0.012) * (TAU * 2600.0 * t).sin() * 0.25
        } else {
            0.0
        };
        buffer.mix_sample(idx, (body + click) * velocity * 0.82);
    }
}

fn render_snare(
    buffer: &mut AudioBuffer,
    start_seconds: f64,
    velocity: f64,
    rng: &mut impl RandomSource,
) {
    let start = buffer.sample_index(start_seconds);
    let len = (0.24 * buffer.sample_rate as f64) as usize;
    let mut body_phase = 0.0;
    for i in 0..len {
        let idx = start + i;
        if idx >= buffer.len() {
            break;
        }
        let t = i as f64 / buffer.sample_rate as f64;
        body_phase = (body_phase + TAU * 185.0 / buffer.sample_rate as f64).rem_euclid(TAU);
        let noise = rng.next_float() * 2.0 - 1.0;
        let env = (-t * 16.0).exp();
        let body = body_phase.sin() * (-t * 9.0).exp() * 0.32;
        buffer.mix_sample(idx, (noise * env * 0.68 + body) * velocity);
    }
}

fn render_hat(
    buffer: &mut AudioBuffer,
    start_seconds: f64,
    velocity: f64,
    rng: &mut impl RandomSource,
) {
    let start = buffer.sample_index(start_seconds);
    let len = (0.055 * buffer.sample_rate as f64) as usize;
    let mut last = 0.0;
    for i in 0..len {
        let idx = start + i;
        if idx >= buffer.len() {
            break;
        }
        let t = i as f64 / buffer.sample_rate as f64;
        let noise = rng.next_float() * 2.0 - 1.0;
        let high = noise - last * 0.72;
        last = noise;
        buffer.mix_sample(idx, high * (-t * 55.0).exp() * velocity);
    }
}

fn render_vocal_formant_chop(
    buffer: &mut AudioBuffer,
    start_seconds: f64,
    duration_seconds: f64,
    pitch_hz: f64,
    vowel: usize,
) {
    let formants = match vowel % 4 {
        0 => [730.0, 1090.0, 2440.0],
        1 => [530.0, 1840.0, 2480.0],
        2 => [570.0, 840.0, 2410.0],
        _ => [300.0, 870.0, 2240.0],
    };
    let start = buffer.sample_index(start_seconds);
    let len = (duration_seconds * buffer.sample_rate as f64).round() as usize;
    let mut carrier = 0.0;
    let mut phases = [0.0; 3];
    for i in 0..len {
        let idx = start + i;
        if idx >= buffer.len() {
            break;
        }
        let t = i as f64 / buffer.sample_rate as f64;
        let position = if duration_seconds > 0.0 {
            t / duration_seconds
        } else {
            1.0
        };
        let env = if position < 0.08 {
            position / 0.08
        } else {
            (1.0 - position).max(0.0).powf(1.8)
        };
        carrier = (carrier + TAU * pitch_hz / buffer.sample_rate as f64).rem_euclid(TAU);
        let mut sample = carrier.sin() * 0.18;
        for (phase, formant) in phases.iter_mut().zip(formants) {
            *phase = (*phase + TAU * formant / buffer.sample_rate as f64).rem_euclid(TAU);
            sample += phase.sin() * 0.18;
        }
        buffer.mix_sample(idx, sample * env * 0.26);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * 1.0_f64.max(a.abs()).max(b.abs())
    }

    fn mp4_with_mvhd(duration: u32, timescale: u32) -> Vec<u8> {
        fn mp4_box(name: &[u8; 4], content: &[u8]) -> Vec<u8> {
            let size = (8 + content.len()) as u32;
            let mut out = Vec::with_capacity(size as usize);
            out.extend_from_slice(&size.to_be_bytes());
            out.extend_from_slice(name);
            out.extend_from_slice(content);
            out
        }

        let ftyp = mp4_box(b"ftyp", b"isom\0\0\0\x01isom");
        let mut mvhd_content = Vec::new();
        mvhd_content.extend_from_slice(&[0, 0, 0, 0]);
        mvhd_content.extend_from_slice(&0u32.to_be_bytes());
        mvhd_content.extend_from_slice(&0u32.to_be_bytes());
        mvhd_content.extend_from_slice(&timescale.to_be_bytes());
        mvhd_content.extend_from_slice(&duration.to_be_bytes());
        let mvhd = mp4_box(b"mvhd", &mvhd_content);
        let moov = mp4_box(b"moov", &mvhd);
        [ftyp, moov].concat()
    }

    #[test]
    fn microtonal_scale_reaches_octave() {
        let scale = MicrotonalScale::rave_collage_19_edo();
        assert!(close(
            scale.step_to_frequency(19),
            scale.base_frequency_hz * 2.0,
            1e-12
        ));
        assert_ne!(
            scale.degree_to_frequency(1, 0),
            scale.degree_to_frequency(2, 0)
        );
    }

    #[test]
    fn pitch_bend_interpolates() {
        let bend = PitchBendCurve::swoop(-100.0, 100.0);
        assert!(close(bend.cents_at(0.0), -100.0, 1e-12));
        assert!(close(bend.cents_at(0.5), 0.0, 1e-12));
        assert!(close(bend.cents_at(1.0), 100.0, 1e-12));
    }

    #[test]
    fn fft_finds_sine_peak() {
        let sample_rate = 44_100;
        let n = 2048usize;
        let freq = 440.0;
        let samples: Vec<f64> = (0..n)
            .map(|i| (TAU * freq * i as f64 / sample_rate as f64).sin())
            .collect();
        let bins = analyze_fft(&samples, sample_rate, WindowFunction::Hann);
        let peak = bins
            .iter()
            .max_by(|a, b| a.magnitude.partial_cmp(&b.magnitude).unwrap())
            .unwrap();
        let bin_width = sample_rate as f64 / n as f64;
        assert!((peak.frequency_hz - freq).abs() <= bin_width);
        assert!(spectral_centroid(&bins) > 0.0);
    }

    #[test]
    fn effects_keep_samples_finite() {
        let mut buffer = AudioBuffer::from_samples(8_000, vec![0.25; 1024]);
        apply_soft_clip(&mut buffer, 3.0, 1.0);
        apply_bitcrusher(
            &mut buffer,
            BitCrusherEffect {
                bits: 8,
                hold_samples: 4,
                mix: 0.5,
            },
        );
        apply_feedback_delay(
            &mut buffer,
            DelayEffect {
                delay_seconds: 0.01,
                feedback: 0.25,
                mix: 0.3,
            },
        );
        apply_schroeder_reverb(
            &mut buffer,
            ReverbEffect {
                room_size: 0.3,
                damping: 0.3,
                mix: 0.2,
            },
        );
        assert!(buffer.samples.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn discovery_avoids_acoustic_emulation_labels() {
        let palette = discover_instrument_palette(InstrumentDiscoverySpec {
            seed: 7,
            sample_rate: 8_000,
            candidates_per_role: 3,
            ..Default::default()
        });
        assert_eq!(palette.instruments.len(), 5);
        assert!(palette.validate_anti_mimicry().is_ok());
        assert!(palette.for_role(InstrumentRole::Lead).is_some());
        assert!(palette.for_role(InstrumentRole::Texture).is_some());
    }

    #[test]
    fn italian_music_feature_catalog_covers_requested_terms() {
        let requested = [
            "ritmo",
            "melodia",
            "armonia",
            "timbro",
            "tempo",
            "dinamica",
            "intensità",
            "altezza",
            "durata",
            "pausa",
            "battito",
            "misura",
            "accordo",
            "tonalità",
            "scala",
            "fraseggio",
            "articolazione",
            "legato",
            "staccato",
            "vibrato",
            "cadenza",
            "modulazione",
            "improvvisazione",
            "arrangiamento",
            "orchestrazione",
            "tessitura",
            "metro",
            "sincope",
            "accento",
            "groove",
            "andamento",
            "espressione",
            "interpretazione",
            "registrazione",
            "equalizzazione",
            "riverbero",
            "eco",
            "distorsione",
            "compressione",
            "campionamento",
            "loop",
            "sequenza",
            "armonizzazione",
            "contrappunto",
            "polifonia",
            "monodia",
            "modalità",
            "intonazione",
            "pulsazione",
            "spettro",
        ];

        assert_eq!(italian_music_feature_catalog().len(), 50);
        for term in requested {
            assert!(
                find_italian_music_feature(term).is_some(),
                "missing Italian music feature {term}"
            );
        }
        let coverage = music_feature_coverage(requested.iter().copied());
        assert!(coverage.is_complete(), "{coverage:?}");
        assert!(coverage.extras.is_empty(), "{coverage:?}");
    }

    #[test]
    fn generated_track_plan_covers_all_italian_music_features() {
        let plan =
            generate_track_structure_plan("feature coverage", MusicGenre::Electronica, 30.0, 9);
        let coverage = validate_italian_music_feature_coverage(&plan.italian_features)
            .expect("generated plan should cover the canonical feature catalog");
        assert_eq!(coverage.covered.len(), 50);
        assert!(plan.italian_features.contains(&"registrazione".to_string()));
        assert!(plan
            .italian_features
            .contains(&"equalizzazione".to_string()));
        assert!(plan.italian_features.contains(&"contrappunto".to_string()));
        assert!(plan.italian_features.contains(&"spettro".to_string()));
    }

    #[test]
    fn default_music_studio_board_validates_and_emits_visual_block_ir() {
        let board = default_music_studio_sound_board();
        let validation = board
            .validate()
            .expect("default music studio sound board should validate");
        assert_eq!(validation.feature_coverage.covered.len(), 50);
        assert!(validation.source_blocks.contains(&"transport".to_string()));
        assert!(validation.output_blocks.contains(&"master".to_string()));
        assert_eq!(validation.connection_count, board.connections.len());
        assert!(board
            .reference_apps
            .iter()
            .any(|app| app.software == "Reason"));

        let ir = board
            .to_visual_block_ir()
            .expect("valid board should convert to visual block IR");
        let JsonValue::Object(root) = &ir else {
            panic!("visual block IR should be a JSON object");
        };
        let blocks = root
            .get("blocks")
            .and_then(|value| match value {
                JsonValue::Array(blocks) => Some(blocks),
                _ => None,
            })
            .expect("IR has blocks");
        let connections = root
            .get("connections")
            .and_then(|value| match value {
                JsonValue::Array(connections) => Some(connections),
                _ => None,
            })
            .expect("IR has connections");
        assert_eq!(blocks.len(), board.blocks.len());
        assert_eq!(connections.len(), board.connections.len());
        let json = ir.to_json_string();
        assert!(json.contains("\"kind\":\"visual-block-graph\""), "{json}");
        assert!(json.contains("music-midi-sequencer"), "{json}");
        assert!(json.contains("italianFeatures"), "{json}");
        assert!(json.contains("midi_out"), "{json}");
        assert!(json.contains("audio_in"), "{json}");
    }

    #[test]
    fn music_studio_board_rejects_missing_features_and_bad_bus_connections() {
        let mut missing_feature = default_music_studio_sound_board();
        for block in &mut missing_feature.blocks {
            block.feature_terms.retain(|feature| feature != "spettro");
        }
        let err = missing_feature
            .validate()
            .expect_err("board missing spettro should reject");
        assert!(err.contains("spettro"), "{err}");

        let mut bad_bus = default_music_studio_sound_board();
        bad_bus.connections.push(MusicStudioBoardConnection::new(
            "bad-midi-audio",
            "synth",
            "audio_out",
            "mixer",
            "audio_in",
            MusicStudioBusKind::Midi,
        ));
        let err = bad_bus
            .validate()
            .expect_err("declared MIDI connection over audio ports should reject");
        assert!(err.contains("bus mismatch"), "{err}");
    }

    #[test]
    fn short_song_generation_is_deterministic_and_legal() {
        let spec = SongSpec {
            duration_seconds: 4.0,
            sample_rate: 8_000,
            bpm: 124.0,
            seed: 42,
            ..Default::default()
        };
        let a = generate_microtonal_song(spec.clone());
        let b = generate_microtonal_song(spec);
        assert_eq!(a.audio.samples, b.audio.samples);
        assert!(a.audio.peak() > 0.1);
        assert!(a.summary.rendered_events > 0);
        assert!(a.sample_manifest.validate_legal_sources().is_ok());
        assert!(a
            .summary
            .parts
            .iter()
            .any(|part| part.name == "main melody"));
        assert!(a
            .summary
            .parts
            .iter()
            .any(|part| part.name == "secondary part A"));
        assert!(a
            .summary
            .parts
            .iter()
            .any(|part| part.name == "secondary part B"));
        assert!(a.summary.instruments.iter().all(|name| {
            let lower = name.to_ascii_lowercase();
            !["harp", "accordion", "harmonica"]
                .iter()
                .any(|w| lower.contains(w))
        }));
        assert_eq!(a.audio.len(), 32_000);
    }

    #[test]
    fn mp4_duration_parser_uses_box_structure() {
        let bytes = mp4_with_mvhd(20_000, 1_000);
        assert_eq!(parse_mp4_duration_seconds(&bytes), Some(20.0));

        let stray = b"\0\0\0\x18ftypisomnot-a-box-mvhd\0\0\0\0\0".to_vec();
        assert_eq!(parse_mp4_duration_seconds(&stray), None);
    }

    #[test]
    fn sample_seed_read_respects_byte_limit() {
        let path = std::env::temp_dir().join(format!(
            "music-seed-limit-{}-{}.mp4",
            std::process::id(),
            42
        ));
        std::fs::write(&path, mp4_with_mvhd(20_000, 1_000)).expect("write mp4 fixture");
        let err = derive_music_sample_seed_from_mp4_with_limit(&path, 12)
            .expect_err("tiny byte limit should reject fixture");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        let sample = derive_music_sample_seed_from_mp4_with_limit(&path, 4096)
            .expect("fixture stays under larger byte limit");
        assert_eq!(sample.duration_seconds, 20.0);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn url_input_fields_cover_requested_source_classes() {
        let fields = music_url_input_fields();
        let ids: Vec<&str> = fields.iter().map(|field| field.id.as_str()).collect();
        for expected in [
            "youtube_url",
            "facebook_url",
            "instagram_url",
            "s3_url",
            "cloudfront_url",
            "cloudflare_url",
            "static_asset_url",
            "any_audio_url",
        ] {
            assert!(ids.contains(&expected), "missing input field {expected}");
        }

        let all_kinds: Vec<MusicUrlSourceKind> = fields
            .iter()
            .flat_map(|field| field.source_kinds.iter().copied())
            .collect();
        for expected in [
            MusicUrlSourceKind::YouTube,
            MusicUrlSourceKind::Facebook,
            MusicUrlSourceKind::Instagram,
            MusicUrlSourceKind::S3,
            MusicUrlSourceKind::CloudFront,
            MusicUrlSourceKind::Cloudflare,
            MusicUrlSourceKind::StaticAssetHost,
            MusicUrlSourceKind::DirectAudio,
            MusicUrlSourceKind::DirectVideo,
            MusicUrlSourceKind::OtherUrl,
        ] {
            assert!(all_kinds.contains(&expected), "missing kind {expected:?}");
        }
    }

    #[test]
    fn classify_music_urls_for_platform_storage_cdn_and_static_sources() {
        let cases = [
            (
                "https://www.youtube.com/watch?v=abc",
                MusicUrlSourceKind::YouTube,
                "youtube_url",
                "rust-youtube-player-response",
            ),
            (
                "https://fb.watch/example",
                MusicUrlSourceKind::Facebook,
                "facebook_url",
                "yt-dlp-or-platform-extractor",
            ),
            (
                "https://www.instagram.com/reel/example/",
                MusicUrlSourceKind::Instagram,
                "instagram_url",
                "yt-dlp-or-platform-extractor",
            ),
            (
                "https://bucket.s3.amazonaws.com/folder/loop.mp3",
                MusicUrlSourceKind::S3,
                "s3_url",
                "direct-http",
            ),
            (
                "https://d111111abcdef8.cloudfront.net/audio/seed.mp4",
                MusicUrlSourceKind::CloudFront,
                "cloudfront_url",
                "direct-http",
            ),
            (
                "https://example.r2.cloudflarestorage.com/sample.wav",
                MusicUrlSourceKind::Cloudflare,
                "cloudflare_url",
                "direct-http",
            ),
            (
                "https://static.example.com/music/loop.wav?download=1",
                MusicUrlSourceKind::StaticAssetHost,
                "static_asset_url",
                "direct-http",
            ),
            (
                "https://media.example.net/beat.flac",
                MusicUrlSourceKind::DirectAudio,
                "any_audio_url",
                "direct-http",
            ),
            (
                "https://media.example.net/clip.webm#part",
                MusicUrlSourceKind::DirectVideo,
                "any_audio_url",
                "direct-http",
            ),
            (
                "https://example.net/share/opaque-id",
                MusicUrlSourceKind::OtherUrl,
                "any_audio_url",
                "yt-dlp-or-platform-extractor",
            ),
        ];

        for (url, kind, field, downloader) in cases {
            let spec = classify_music_source_url(url).expect("URL should classify");
            assert_eq!(spec.kind, kind, "{url}");
            assert_eq!(spec.input_field_id, field, "{url}");
            assert_eq!(spec.downloader_hint, downloader, "{url}");
        }
    }

    #[test]
    fn youtube_video_id_parser_handles_common_url_shapes() {
        let cases = [
            (
                "https://www.youtube.com/watch?v=A4LAodkYjJg&feature=share",
                "A4LAodkYjJg",
            ),
            ("https://youtu.be/A4LAodkYjJg?t=30", "A4LAodkYjJg"),
            (
                "https://www.youtube.com/shorts/A4LAodkYjJg?si=abc",
                "A4LAodkYjJg",
            ),
            (
                "https://www.youtube-nocookie.com/embed/A4LAodkYjJg",
                "A4LAodkYjJg",
            ),
        ];

        for (url, expected) in cases {
            let spec = classify_music_source_url(url).expect("YouTube URL should classify");
            assert_eq!(youtube_video_id_from_spec(&spec).as_deref(), Some(expected));
            assert_eq!(
                canonical_youtube_watch_url(&spec).expect("canonical URL"),
                format!("https://www.youtube.com/watch?v={expected}")
            );
        }
    }

    #[test]
    fn youtube_player_response_fixture_selects_direct_progressive_media_url() {
        let html = r#"
            <html><script>
            var ytInitialPlayerResponse = {
              "streamingData": {
                "formats": [
                  {
                    "itag": 18,
                    "mimeType": "video/mp4; codecs=\"avc1.42001E, mp4a.40.2\"",
                    "bitrate": 442641,
                    "approxDurationMs": "31000",
                    "url": "https://rr1---sn-example.googlevideo.com/videoplayback?mime=video%2Fmp4&clen=123456&sig=test"
                  }
                ],
                "adaptiveFormats": [
                  {
                    "itag": 140,
                    "mimeType": "audio/mp4; codecs=\"mp4a.40.2\"",
                    "bitrate": 132000,
                    "approxDurationMs": "31000"
                  }
                ]
              },
              "videoDetails": { "lengthSeconds": "31" }
            };
            </script></html>
        "#;
        let player_json =
            extract_youtube_initial_player_response_json(html).expect("player JSON should parse");
        let response: serde_json::Value =
            serde_json::from_str(&player_json).expect("fixture JSON should parse");
        let spec = classify_music_source_url("https://www.youtube.com/watch?v=A4LAodkYjJg")
            .expect("URL should classify");
        let resolved = select_youtube_public_media_link_from_player_response(&spec, &response)
            .expect("fixture has direct progressive media URL");

        assert_eq!(resolved.source_kind, MusicUrlSourceKind::YouTube);
        assert_eq!(resolved.extractor, "rust-youtube-player-response");
        assert_eq!(
            resolved.media_url,
            "https://rr1---sn-example.googlevideo.com/videoplayback?mime=video%2Fmp4&clen=123456&sig=test"
        );
        assert_eq!(
            resolved.mime_type,
            "video/mp4; codecs=\"avc1.42001E, mp4a.40.2\""
        );
        assert_eq!(resolved.bitrate, Some(442641));
        assert_eq!(resolved.content_length, Some(123456));
        assert_eq!(resolved.duration_seconds, Some(31.0));
    }

    #[test]
    fn youtube_player_response_reports_signature_cipher_gap() {
        let response: serde_json::Value = serde_json::json!({
            "streamingData": {
                "formats": [{
                    "mimeType": "video/mp4; codecs=\"avc1.42001E, mp4a.40.2\"",
                    "signatureCipher": "url=https%3A%2F%2Fexample.test%2Fmedia&sp=sig&s=abc"
                }]
            }
        });
        let spec = classify_music_source_url("https://www.youtube.com/watch?v=A4LAodkYjJg")
            .expect("URL should classify");
        let err = select_youtube_public_media_link_from_player_response(&spec, &response)
            .expect_err("cipher-only fixture should reject");

        assert!(err.contains("signature-protected"), "{err}");
        assert!(err.contains("signature deciphering"), "{err}");
    }

    #[test]
    fn url_source_examples_match_classifier_and_selection() {
        let examples = music_url_source_examples();
        let covered_kinds: Vec<MusicUrlSourceKind> =
            examples.iter().map(|example| example.source_kind).collect();
        for expected in [
            MusicUrlSourceKind::YouTube,
            MusicUrlSourceKind::Facebook,
            MusicUrlSourceKind::Instagram,
            MusicUrlSourceKind::S3,
            MusicUrlSourceKind::CloudFront,
            MusicUrlSourceKind::Cloudflare,
            MusicUrlSourceKind::StaticAssetHost,
            MusicUrlSourceKind::DirectAudio,
            MusicUrlSourceKind::DirectVideo,
            MusicUrlSourceKind::OtherUrl,
        ] {
            assert!(
                covered_kinds.contains(&expected),
                "missing example {expected:?}"
            );
        }

        for example in examples {
            let spec =
                classify_music_source_url(example.source_url).expect("example URL should classify");
            assert_eq!(spec.kind, example.source_kind, "{}", example.source_url);
            assert_eq!(
                spec.input_field_id, example.field_id,
                "{}",
                example.source_url
            );

            let selected = select_music_url_input(&[(example.field_id, example.source_url)])
                .expect("example form should select")
                .expect("example form has a URL");
            assert_eq!(selected.submitted_field_id, example.field_id);
            assert_eq!(selected.spec.kind, example.source_kind);
        }
    }

    #[test]
    fn select_music_url_input_uses_ui_order_for_named_fields() {
        let selected = select_music_url_input(&[
            ("any_audio_url", "https://media.example.net/beat.flac"),
            ("youtube_url", "https://www.youtube.com/watch?v=abc"),
        ])
        .expect("selection should succeed")
        .expect("a source should be selected");

        assert_eq!(selected.submitted_field_id, "youtube_url");
        assert_eq!(selected.spec.kind, MusicUrlSourceKind::YouTube);
        assert_eq!(selected.spec.input_field_id, "youtube_url");
    }

    #[test]
    fn select_music_url_input_accepts_normalized_source_url_payload() {
        let selected = select_music_url_input(&[
            ("source_url", " https://media.example.net/beat.flac "),
            ("source_input_field", "any_audio_url"),
        ])
        .expect("selection should succeed")
        .expect("a source should be selected");

        assert_eq!(selected.submitted_field_id, "any_audio_url");
        assert_eq!(selected.spec.raw_url, "https://media.example.net/beat.flac");
        assert_eq!(selected.spec.kind, MusicUrlSourceKind::DirectAudio);
        assert_eq!(selected.spec.input_field_id, "any_audio_url");
    }

    #[test]
    fn select_music_url_input_ignores_unrelated_fields_without_url() {
        let selected =
            select_music_url_input(&[("prompt", "make it brighter"), ("duration_seconds", "180")])
                .expect("selection should not fail");

        assert_eq!(selected, None);
    }

    #[test]
    fn select_music_url_input_rejects_first_nonempty_known_field() {
        let err = select_music_url_input(&[
            ("facebook_url", "file:///tmp/song.mp3"),
            ("any_audio_url", "https://media.example.net/beat.flac"),
        ])
        .expect_err("invalid earlier UI field should reject the form");

        assert!(err.contains("facebook_url:"), "{err}");
        assert!(err.contains("http"), "{err}");
    }

    #[test]
    fn derive_music_sample_seed_from_url_source_is_deterministic() {
        let source = classify_music_source_url("https://media.example.net/audio/beat.flac")
            .expect("URL should classify");
        let first = derive_music_sample_seed_from_url_source(&source);
        let second = derive_music_sample_seed_from_url_source(&source);

        assert_eq!(first, second);
        assert_eq!(first.source_path, source.raw_url);
        assert!((10.0..=50.0).contains(&first.duration_seconds));
        assert!(first.byte_entropy > 0.0);
        assert!(
            first.descriptors.contains(&"music-url-seed".to_string()),
            "{:?}",
            first.descriptors
        );
        assert!(
            first
                .descriptors
                .contains(&"source-kind=direct-audio".to_string()),
            "{:?}",
            first.descriptors
        );
        assert!(
            first.descriptors.contains(&"direct-media=true".to_string()),
            "{:?}",
            first.descriptors
        );
    }

    #[test]
    fn url_source_song_spec_uses_url_seed_and_prompt_influence() {
        let source = classify_music_source_url("https://www.youtube.com/watch?v=abc")
            .expect("URL should classify");
        let sample = derive_music_sample_seed_from_url_source(&source);
        let spec = song_spec_from_music_url_source_with_prompt(
            &source,
            "url inspiration test",
            24.0,
            Some("faster jungle amen breaks with massive bass"),
        );

        assert_eq!(spec.title, "url inspiration test");
        assert_eq!(spec.duration_seconds, 24.0);
        assert_eq!(spec.genre, MusicGenre::Jungle);
        assert_ne!(spec.seed, sample.seed);
        let plan = spec.structure_plan.expect("URL spec has a plan");
        assert!(plan.italian_features.contains(&"campionamento".to_string()));
        assert!(plan.italian_features.contains(&"direzione".to_string()));
    }

    #[test]
    fn render_music_url_seed_wav_accepts_form_fields_and_writes_audio() {
        let path = std::env::temp_dir().join(format!(
            "music-url-seed-handler-{}-{}.wav",
            std::process::id(),
            7
        ));
        let result = render_music_url_seed_wav(
            &[
                ("s3_url", "https://bucket.s3.amazonaws.com/audio/loop.mp3"),
                ("duration_seconds", "4"),
                ("title", "URL handler render"),
                ("prompt", "ambient slower wide space"),
            ],
            &path,
        )
        .expect("URL seed form should render a wav");

        assert_eq!(result.selected_source.submitted_field_id, "s3_url");
        assert_eq!(result.selected_source.spec.kind, MusicUrlSourceKind::S3);
        assert_eq!(result.selected_source.spec.input_field_id, "s3_url");
        assert_eq!(result.summary.title, "URL handler render");
        assert_eq!(result.summary.genre, MusicGenre::Ambient);
        assert!((result.summary.duration_seconds - 4.0).abs() < 0.001);
        assert!(result.wav_bytes > 44);
        assert_eq!(
            std::fs::metadata(&path).expect("wav exists").len(),
            result.wav_bytes
        );

        let json = music_url_seed_render_result_json(&result, "/music/generated/url.wav");
        assert!(json.contains(r#""ok":true"#), "{json}");
        assert!(
            json.contains(r#""wav_url":"/music/generated/url.wav""#),
            "{json}"
        );
        assert!(json.contains(r#""source_kind":"s3""#), "{json}");
        assert!(json.contains(r#""submitted_field":"s3_url""#), "{json}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn music_url_seed_error_json_reports_endpoint_failures() {
        let json = music_url_seed_error_json(
            "validation_error",
            "music source URL must include \"http\" & a public host",
        );

        assert_eq!(
            json,
            r#"{"ok":false,"error_code":"validation_error","error":"music source URL must include \"http\" & a public host"}"#
        );
    }

    #[test]
    fn render_music_url_seed_wav_rejects_missing_or_invalid_form_input() {
        let path = std::env::temp_dir().join(format!(
            "music-url-seed-handler-error-{}-{}.wav",
            std::process::id(),
            11
        ));
        let missing = render_music_url_seed_wav(&[("prompt", "make it move")], &path)
            .expect_err("missing URL should reject");
        assert!(missing.contains("at least one source URL"), "{missing}");

        let invalid_duration = render_music_url_seed_wav(
            &[
                ("any_audio_url", "https://media.example.net/beat.flac"),
                ("duration_seconds", "-2"),
            ],
            &path,
        )
        .expect_err("invalid duration should reject");
        assert!(
            invalid_duration.contains("duration_seconds must be positive"),
            "{invalid_duration}"
        );
        assert!(!path.exists(), "invalid form should not write a wav");
    }

    #[test]
    fn rendered_url_seed_form_has_all_requested_inputs_and_posts_source_url() {
        let html = render_music_url_seed_form_html("music/sample-seed");
        for expected in [
            r#"id="youtube_url""#,
            r#"id="facebook_url""#,
            r#"id="instagram_url""#,
            r#"id="s3_url""#,
            r#"id="cloudfront_url""#,
            r#"id="cloudflare_url""#,
            r#"id="static_asset_url""#,
            r#"id="any_audio_url""#,
            r#"id="music_url_title""#,
            r#"id="music_url_seed_download""#,
            r#"role="status""#,
            "CloudFront URL",
            "Cloudflare URL",
            "Static asset URL",
            "Any audio or media URL",
            "direct-audio,direct-video,other-url",
        ] {
            assert!(
                html.contains(expected),
                "missing rendered UI token {expected}"
            );
        }
        for expected in [
            r#"fd.append("source_url", selected.value)"#,
            r#"fd.append("source_input_field", selected.id)"#,
            r#"fd.append("source_platform", selected.kinds.split(",")[0] || "auto")"#,
            r#"fd.append("prompt""#,
            r#"fd.append("duration_seconds""#,
            r#"fd.append("title", document.getElementById("music_url_title").value.trim()"#,
            "data.source_kind",
            "data.host",
            "data.genre",
            "data.bpm",
            "data.wavUrl",
            "download.href = wavUrl",
            "music/sample-seed",
        ] {
            assert!(
                html.contains(expected),
                "missing endpoint mapping token {expected}"
            );
        }
    }

    #[test]
    fn url_seed_endpoint_contract_covers_ui_and_handler_fields() {
        let contract = music_url_seed_endpoint_contract_json("sample-seed");
        for expected in [
            r#""schema": "des/music-url-seed-endpoint/v1""#,
            r#""endpoint": "sample-seed""#,
            r#""method": "POST""#,
            r#""content_type": "multipart/form-data""#,
            r#""host_policy": "public-http-only-no-credentials-no-localhost-private-or-internal""#,
            r#""examples""#,
            r#""id": "youtube_url""#,
            r#""id": "facebook_url""#,
            r#""id": "instagram_url""#,
            r#""id": "s3_url""#,
            r#""id": "cloudfront_url""#,
            r#""id": "cloudflare_url""#,
            r#""id": "static_asset_url""#,
            r#""id": "any_audio_url""#,
            r#""source_url""#,
            r#""source_input_field""#,
            r#""source_platform""#,
            r#""prompt""#,
            r#""duration_seconds""#,
            r#""title""#,
            r#""wav_url""#,
            r#""source_kind""#,
            r#""host""#,
            r#""downloader""#,
            r#""genre""#,
            r#""bpm""#,
            r#""wav_bytes""#,
            r#""error_code""#,
            r#""error""#,
            r#""source_url": "https://www.youtube.com/watch?v=abc123""#,
            r#""source_url": "https://example.r2.cloudflarestorage.com/sample.wav""#,
            r#""source_kind": "other-url""#,
        ] {
            assert!(
                contract.contains(expected),
                "missing contract token {expected}"
            );
        }
    }

    #[test]
    fn rendered_url_seed_form_escapes_endpoint_text() {
        let html = render_music_url_seed_form_html("music/sample-seed?x=\"<&");
        assert!(html.contains("music/sample-seed?x=&quot;&lt;&amp;"));
        assert!(!html.contains("music/sample-seed?x=\"<&"));
    }

    #[test]
    fn music_url_parser_rejects_unsupported_or_secret_bearing_urls() {
        for url in [
            "",
            "file:///tmp/seed.mp3",
            "ftp://example.com/seed.mp3",
            "https://user:pass@example.com/seed.mp3",
            "https:///missing-host.mp3",
            "http://localhost/seed.mp3",
            "http://127.0.0.1/seed.mp3",
            "http://10.0.0.5/seed.mp3",
            "http://172.16.4.2/seed.mp3",
            "http://192.168.1.10/seed.mp3",
            "http://169.254.1.2/seed.mp3",
            "http://100.64.1.2/seed.mp3",
            "http://[::1]/seed.mp3",
            "http://[fd00::1]/seed.mp3",
            "http://printer.local/seed.mp3",
            "http://intranet/seed.mp3",
        ] {
            assert!(
                classify_music_source_url(url).is_err(),
                "expected rejection for {url:?}"
            );
        }

        let public_ip = classify_music_source_url("https://93.184.216.34/seed.mp3")
            .expect("public IP media URL should classify");
        assert_eq!(public_ip.kind, MusicUrlSourceKind::DirectAudio);
    }

    #[test]
    fn prompt_guides_music_sample_seed_variation() {
        let sample = MusicSampleSeed {
            source_path: "seed.mp4".to_string(),
            duration_seconds: 20.0,
            seed: 1234,
            byte_entropy: 7.4,
            suggested_genre: MusicGenre::Downtempo,
            suggested_bpm: 92.0,
            key_bias_steps: 0,
            meter_bias: (7, 8),
            descriptors: vec!["music-sample-seed".to_string()],
        };
        let spec = song_spec_from_music_sample_seed_with_prompt(
            &sample,
            "prompted",
            30.0,
            Some("expand into faster jungle with massive synth melody and 13/16 stutter"),
        );
        let plan = spec.structure_plan.expect("prompted spec has a plan");
        assert_eq!(spec.genre, MusicGenre::Jungle);
        assert!(spec.bpm > sample.suggested_bpm);
        assert_ne!(spec.seed, sample.seed);
        assert_eq!(
            plan.time_signature_changes
                .first()
                .map(|meter| (meter.numerator, meter.denominator)),
            Some((13, 16))
        );
        assert!(plan
            .italian_features
            .iter()
            .any(|feature| feature == "prompt-directed" || feature == "prompt-expand"));
    }
}
