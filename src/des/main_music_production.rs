//! Demo runner for the generative music production module.

use std::path::{Path, PathBuf};

use crate::des::general::music_production::{
    analyze_music_sample_prompt, derive_music_sample_seed_from_mp4, generate_microtonal_song,
    render_ten_more_song_album, render_ten_song_album,
    song_spec_from_music_sample_seed_with_prompt, AlbumRenderSummary, MusicGenre, SongSpec,
};

pub fn run() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| s.as_str()) == Some("--album-more") {
        let out_dir = args
            .get(2)
            .cloned()
            .unwrap_or_else(|| "out/music-production-ten-more-breaks".to_string());
        let seed = args
            .get(3)
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(0x6d75_2026);
        let duration_seconds = args
            .get(4)
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(180.0);
        let album = render_ten_more_song_album(&out_dir, seed, duration_seconds)
            .unwrap_or_else(|e| panic!("failed to render album {out_dir}: {e}"));
        print_album_summary(&album);
        return;
    }

    if args.get(1).map(|s| s.as_str()) == Some("--album") {
        let out_dir = args
            .get(2)
            .cloned()
            .unwrap_or_else(|| "out/music-production-ten-songs".to_string());
        let seed = args
            .get(3)
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(0x5150_2026);
        let duration_seconds = args
            .get(4)
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(180.0);
        let album = render_ten_song_album(&out_dir, seed, duration_seconds)
            .unwrap_or_else(|e| panic!("failed to render album {out_dir}: {e}"));
        print_album_summary(&album);
        return;
    }

    if args.get(1).map(|s| s.as_str()) == Some("--sample-seed") {
        let Some(mp4_path) = args.get(2) else {
            panic!("usage: main_music_production --sample-seed <10-50s.mp4> <out.wav> [duration_seconds] [--prompt <text> | --prompt-file <path>]");
        };
        let out_path = args
            .get(3)
            .cloned()
            .unwrap_or_else(|| "out/music-sample-seed-variation.wav".to_string());
        let duration_seconds = args
            .get(4)
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(180.0);
        let sample = derive_music_sample_seed_from_mp4(mp4_path)
            .unwrap_or_else(|e| panic!("failed to derive music-sample-seed: {e}"));
        let prompt = parse_sample_seed_prompt(&args, 5);
        let spec = song_spec_from_music_sample_seed_with_prompt(
            &sample,
            "music-sample-seed variation",
            duration_seconds,
            prompt.as_deref(),
        );
        let render = generate_microtonal_song(spec);
        let out = PathBuf::from(out_path);
        if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
            let _ = std::fs::create_dir_all(parent);
        }
        render
            .audio
            .write_wav16(&out)
            .unwrap_or_else(|e| panic!("failed to write {}: {e}", out.display()));
        let resolved = std::fs::canonicalize(&out).unwrap_or(out);
        println!("music-sample-seed source: {}", sample.source_path);
        println!("seed: {}", sample.seed);
        println!("source duration: {:.2}s", sample.duration_seconds);
        println!("byte entropy: {:.3}", sample.byte_entropy);
        println!("suggested genre: {}", sample.suggested_genre.as_str());
        println!("suggested bpm: {:.2}", sample.suggested_bpm);
        println!("descriptors: {}", sample.descriptors.join(", "));
        if let Some(influence) = prompt.as_deref().and_then(analyze_music_sample_prompt) {
            println!("prompt chars: {}", influence.prompt_chars);
            println!("prompt hash: {}", influence.prompt_hash);
            println!("prompt tags: {}", influence.feature_tags.join(", "));
            println!("prompt bpm delta: {:.1}", influence.bpm_delta);
        }
        print_summary(&resolved, &render.summary);
        return;
    }

    let out_path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "out/music-production-rave-collage.wav".to_string());
    let duration_seconds = args
        .get(2)
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(180.0);
    let seed = args
        .get(3)
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0x5150_1979);
    let genre = args
        .get(4)
        .and_then(|s| parse_genre(s))
        .unwrap_or(MusicGenre::Electronica);

    let render = generate_microtonal_song(SongSpec {
        title: "single generated study".to_string(),
        genre,
        duration_seconds,
        bpm: genre.default_bpm(),
        seed,
        ..Default::default()
    });
    let out = PathBuf::from(out_path);
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        let _ = std::fs::create_dir_all(parent);
    }
    render
        .audio
        .write_wav16(&out)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", out.display()));
    let resolved = std::fs::canonicalize(&out).unwrap_or(out);
    print_summary(&resolved, &render.summary);
}

fn parse_sample_seed_prompt(args: &[String], mut index: usize) -> Option<String> {
    let mut chunks = Vec::new();
    while index < args.len() {
        match args[index].as_str() {
            "--prompt" => {
                let Some(value) = args.get(index + 1) else {
                    panic!("--prompt requires a text value");
                };
                chunks.push(value.clone());
                index += 2;
            }
            "--prompt-file" => {
                let Some(path) = args.get(index + 1) else {
                    panic!("--prompt-file requires a path");
                };
                let text = std::fs::read_to_string(path)
                    .unwrap_or_else(|e| panic!("failed to read prompt file {path}: {e}"));
                chunks.push(text);
                index += 2;
            }
            value if value.starts_with("--") => {
                panic!("unknown sample-seed option {value}");
            }
            value => {
                chunks.push(value.to_string());
                index += 1;
            }
        }
    }
    let prompt = chunks.join("\n").trim().to_string();
    if prompt.is_empty() {
        None
    } else {
        Some(prompt)
    }
}

fn print_summary(path: &Path, summary: &crate::des::general::music_production::ArrangementSummary) {
    println!("Wrote {}", path.display());
    println!("title: {}", summary.title);
    println!("genre: {}", summary.genre.as_str());
    println!("duration: {:.2}s", summary.duration_seconds);
    println!("bpm: {:.2}", summary.bpm);
    println!("scale: {}", summary.scale_name);
    println!("key changes: {}", summary.key_changes.len());
    println!(
        "time signature changes: {}",
        summary.time_signature_changes.len()
    );
    println!("pauses: {}", summary.pauses.len());
    println!(
        "drum patterns: {} ({})",
        summary.drum_variation.pattern_names.len(),
        summary.drum_variation.pattern_names.join(", ")
    );
    println!(
        "drum variation: {:.1}% target {:.1}% (micro variations={}, percussion gain={:.2})",
        summary.drum_variation.variation_ratio() * 100.0,
        summary.drum_variation.repetition_reduction_target * 100.0,
        summary.drum_variation.micro_variations,
        summary.drum_variation.percussion_gain
    );
    println!("instruments: {}", summary.instruments.join(", "));
    println!("parts:");
    for part in &summary.parts {
        println!(
            "  - {}: {} via {} ({} events)",
            part.name,
            part.role.as_str(),
            part.instrument,
            part.events
        );
    }
    println!("events: {}", summary.rendered_events);
    println!("peak: {:.3}", summary.peak);
    println!("rms: {:.3}", summary.rms);
    println!("spectral centroid: {:.1} Hz", summary.spectral_centroid_hz);
}

fn print_album_summary(album: &AlbumRenderSummary) {
    println!("Wrote album to {}", album.out_dir);
    println!("Manifest {}", album.manifest_path);
    for track in &album.tracks {
        println!(
            "{:02}. {} [{}] -> {}",
            track.index,
            track.summary.title,
            track.summary.genre.as_str(),
            track.path
        );
        println!(
            "    changes: keys={} meters={} pauses={} drum_patterns={} fills={} syncopations={} micro_variations={} percussion_gain={:.2}",
            track.summary.key_changes.len(),
            track.summary.time_signature_changes.len(),
            track.summary.pauses.len(),
            track.summary.drum_variation.pattern_names.len(),
            track.summary.drum_variation.fills,
            track.summary.drum_variation.syncopations,
            track.summary.drum_variation.micro_variations,
            track.summary.drum_variation.percussion_gain
        );
    }
}

fn parse_genre(value: &str) -> Option<MusicGenre> {
    let key = value
        .to_ascii_lowercase()
        .replace([' ', '_', '/'], "-")
        .replace("--", "-");
    match key.as_str() {
        "drum-n-bass" | "drum-and-bass" | "dnb" => Some(MusicGenre::DrumAndBass),
        "house" => Some(MusicGenre::House),
        "trance" => Some(MusicGenre::Trance),
        "dance" => Some(MusicGenre::Dance),
        "electronica" => Some(MusicGenre::Electronica),
        "jazz" => Some(MusicGenre::Jazz),
        "ambient" => Some(MusicGenre::Ambient),
        "idm" => Some(MusicGenre::Idm),
        "breakbeat" => Some(MusicGenre::Breakbeat),
        "liquid-funk" => Some(MusicGenre::LiquidFunk),
        "dub-techno" => Some(MusicGenre::DubTechno),
        "future-garage" => Some(MusicGenre::FutureGarage),
        _ => None,
    }
}
