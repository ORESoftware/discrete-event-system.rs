//! Generative music production primitives.
//!
//! This module is intentionally dependency-free and deterministic. It provides
//! a small audio buffer, microtonal tuning, pitch-bend curves, simple synthesis,
//! audio effects, FFT-backed spectrum analysis, and a default three-minute
//! instrumental generator with generated vocal-chop textures. External samples
//! are represented through a license manifest so callers can keep provenance at
//! the audio boundary.

use std::f64::consts::TAU;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crate::des::general::prng::mulberry32;
use crate::des::general::signal_transforms::{run_fft_transform, FastFourierTransformParams};
use crate::des::shared::capabilities::RandomSource;

pub const DEFAULT_SAMPLE_RATE: u32 = 44_100;
pub const DEFAULT_SONG_SECONDS: f64 = 180.0;

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
                italian_feature: "respiro".to_string(),
            },
        ],
        italian_features: vec![
            "ritmo".to_string(),
            "melodia".to_string(),
            "armonia".to_string(),
            "timbro".to_string(),
            "pausa".to_string(),
            "misura".to_string(),
            "fraseggio".to_string(),
            "modulazione".to_string(),
            "metro".to_string(),
            "sincope".to_string(),
            "accento".to_string(),
            "contrappunto".to_string(),
            "polifonia".to_string(),
            "spettro".to_string(),
        ],
    }
}

pub fn derive_music_sample_seed_from_mp4(path: impl AsRef<Path>) -> io::Result<MusicSampleSeed> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)?;
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
    structure_plan
        .italian_features
        .push("campionamento".to_string());
    if let Some(influence) = &prompt_influence {
        structure_plan
            .italian_features
            .push("direzione".to_string());
        for tag in &influence.feature_tags {
            structure_plan
                .italian_features
                .push(format!("prompt-{tag}"));
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
    let mut search_from = 0usize;
    while search_from + 8 < bytes.len() {
        let rel = bytes[search_from..].windows(4).position(|w| w == b"mvhd")?;
        let typ = search_from + rel;
        if typ < 4 || typ + 24 >= bytes.len() {
            return None;
        }
        let box_start = typ - 4;
        let size = u32::from_be_bytes(bytes[box_start..box_start + 4].try_into().ok()?) as usize;
        let content = typ + 4;
        if size >= 24 && box_start + size <= bytes.len() {
            let version = bytes[content];
            if version == 1 {
                if content + 32 <= bytes.len() {
                    let timescale =
                        u32::from_be_bytes(bytes[content + 20..content + 24].try_into().ok()?);
                    let duration =
                        u64::from_be_bytes(bytes[content + 24..content + 32].try_into().ok()?);
                    if timescale > 0 {
                        return Some(duration as f64 / timescale as f64);
                    }
                }
            } else if content + 20 <= bytes.len() {
                let timescale =
                    u32::from_be_bytes(bytes[content + 12..content + 16].try_into().ok()?);
                let duration =
                    u32::from_be_bytes(bytes[content + 16..content + 20].try_into().ok()?);
                if timescale > 0 {
                    return Some(duration as f64 / timescale as f64);
                }
            }
        }
        search_from = typ + 4;
    }
    None
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
