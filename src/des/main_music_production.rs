//! Demo runner for the generative music production module.

use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::des::general::music_production::{
    analyze_music_sample_prompt, classify_music_source_url, derive_music_sample_seed_from_mp4,
    generate_microtonal_song, music_url_input_fields, music_url_seed_endpoint_contract_json,
    music_url_seed_error_json, music_url_seed_render_result_json,
    render_music_url_seed_form_document_html, render_music_url_seed_wav,
    render_ten_more_song_album, render_ten_song_album,
    song_spec_from_music_sample_seed_with_prompt, write_music_url_seed_contract_json,
    write_music_url_seed_form_html, AlbumRenderSummary, MusicGenre, SongSpec,
};

pub fn run() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| s.as_str()) == Some("--url-source-fields") {
        for field in music_url_input_fields() {
            let kinds: Vec<&str> = field
                .source_kinds
                .iter()
                .map(|kind| kind.as_str())
                .collect();
            println!("{} | {} | {}", field.id, field.label, kinds.join(","));
            println!("  placeholder: {}", field.placeholder);
        }
        return;
    }

    if args.get(1).map(|s| s.as_str()) == Some("--classify-url") {
        let Some(url) = args.get(2) else {
            panic!("usage: main_music_production --classify-url <audio-or-media-url>");
        };
        let spec = classify_music_source_url(url)
            .unwrap_or_else(|e| panic!("failed to classify music source URL: {e}"));
        println!("url: {}", spec.raw_url);
        println!("host: {}", spec.host);
        println!("kind: {}", spec.kind.as_str());
        println!("input field: {}", spec.input_field_id);
        println!("downloader: {}", spec.downloader_hint);
        println!("direct media hint: {}", spec.direct_media_hint);
        return;
    }

    if args.get(1).map(|s| s.as_str()) == Some("--url-source-ui") {
        let out_path = args
            .get(2)
            .cloned()
            .unwrap_or_else(|| "out/music-url-source-inputs.html".to_string());
        let endpoint = args
            .get(3)
            .map(|value| value.as_str())
            .unwrap_or("music/sample-seed");
        let out = write_music_url_seed_form_html(&out_path, endpoint)
            .unwrap_or_else(|e| panic!("failed to write {out_path}: {e}"));
        let resolved = std::fs::canonicalize(&out).unwrap_or(out);
        println!("Wrote {}", resolved.display());
        return;
    }

    if args.get(1).map(|s| s.as_str()) == Some("--url-source-contract") {
        let out_path = args
            .get(2)
            .cloned()
            .unwrap_or_else(|| "out/music/url-source-contract.json".to_string());
        let endpoint = args
            .get(3)
            .map(|value| value.as_str())
            .unwrap_or("music/sample-seed");
        let out = write_music_url_seed_contract_json(&out_path, endpoint)
            .unwrap_or_else(|e| panic!("failed to write {out_path}: {e}"));
        let resolved = std::fs::canonicalize(&out).unwrap_or(out);
        println!("Wrote {}", resolved.display());
        return;
    }

    if args.get(1).map(|s| s.as_str()) == Some("--url-source-server") {
        let bind_addr = args
            .get(2)
            .map(|value| value.as_str())
            .unwrap_or("127.0.0.1:7878");
        let generated_dir = args
            .get(3)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("out/music/generated"));
        run_music_url_workbench_server(bind_addr, &generated_dir)
            .unwrap_or_else(|e| panic!("failed to run music URL workbench server: {e}"));
        return;
    }

    if args.get(1).map(|s| s.as_str()) == Some("--url-seed") {
        let Some(raw_url) = args.get(2) else {
            panic!("usage: main_music_production --url-seed <audio-or-media-url> <out.wav> [duration_seconds] [--prompt <text> | --prompt-file <path>]");
        };
        let out_path = args
            .get(3)
            .cloned()
            .unwrap_or_else(|| "out/music-url-seed-variation.wav".to_string());
        let duration_seconds = args.get(4).cloned().unwrap_or_else(|| "180".to_string());
        let prompt = parse_sample_seed_prompt(&args, 5);
        let mut fields = vec![
            ("source_url", raw_url.clone()),
            ("duration_seconds", duration_seconds),
            ("title", "music-url-source variation".to_string()),
        ];
        if let Some(prompt) = prompt.clone() {
            fields.push(("prompt", prompt));
        }
        let field_refs: Vec<(&str, &str)> = fields
            .iter()
            .map(|(key, value)| (*key, value.as_str()))
            .collect();
        let result = render_music_url_seed_wav(&field_refs, &out_path)
            .unwrap_or_else(|e| panic!("failed to render URL-inspired music: {e}"));
        let source = &result.selected_source.spec;
        let sample = &result.sample_seed;
        println!("music-url-source: {}", source.raw_url);
        println!("host: {}", source.host);
        println!("kind: {}", source.kind.as_str());
        println!("input field: {}", source.input_field_id);
        println!("downloader: {}", source.downloader_hint);
        println!("direct media hint: {}", source.direct_media_hint);
        println!("seed: {}", sample.seed);
        println!("url-derived duration hint: {:.2}s", sample.duration_seconds);
        println!("url entropy: {:.3}", sample.byte_entropy);
        println!("suggested genre: {}", sample.suggested_genre.as_str());
        println!("suggested bpm: {:.2}", sample.suggested_bpm);
        println!("descriptors: {}", sample.descriptors.join(", "));
        if let Some(influence) = prompt.as_deref().and_then(analyze_music_sample_prompt) {
            println!("prompt chars: {}", influence.prompt_chars);
            println!("prompt hash: {}", influence.prompt_hash);
            println!("prompt tags: {}", influence.feature_tags.join(", "));
            println!("prompt bpm delta: {:.1}", influence.bpm_delta);
        }
        print_summary(&result.output_path, &result.summary);
        return;
    }

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MusicUrlWorkbenchHttpResponse {
    pub status_code: u16,
    pub status_text: &'static str,
    pub content_type: &'static str,
    pub body: Vec<u8>,
}

impl MusicUrlWorkbenchHttpResponse {
    fn text(status_code: u16, status_text: &'static str, body: impl Into<String>) -> Self {
        MusicUrlWorkbenchHttpResponse {
            status_code,
            status_text,
            content_type: "text/plain; charset=utf-8",
            body: body.into().into_bytes(),
        }
    }

    fn html(body: impl Into<String>) -> Self {
        MusicUrlWorkbenchHttpResponse {
            status_code: 200,
            status_text: "OK",
            content_type: "text/html; charset=utf-8",
            body: body.into().into_bytes(),
        }
    }

    fn json(status_code: u16, status_text: &'static str, body: impl Into<String>) -> Self {
        MusicUrlWorkbenchHttpResponse {
            status_code,
            status_text,
            content_type: "application/json; charset=utf-8",
            body: body.into().into_bytes(),
        }
    }

    fn wav(body: Vec<u8>) -> Self {
        MusicUrlWorkbenchHttpResponse {
            status_code: 200,
            status_text: "OK",
            content_type: "audio/wav",
            body,
        }
    }
}

pub fn run_music_url_workbench_server(
    bind_addr: &str,
    generated_dir: impl AsRef<Path>,
) -> io::Result<()> {
    let listener = TcpListener::bind(bind_addr)?;
    let local_addr = listener.local_addr()?;
    let generated_dir = generated_dir.as_ref().to_path_buf();
    eprintln!(
        "Music URL Seed Workbench listening on http://{}/music/url-source-inputs.html",
        local_addr
    );
    eprintln!("Generated WAV files: {}", generated_dir.display());
    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                if let Err(err) = handle_music_url_workbench_stream(&mut stream, &generated_dir) {
                    eprintln!("music URL workbench request failed: {err}");
                }
            }
            Err(err) => eprintln!("music URL workbench connection failed: {err}"),
        }
    }
    Ok(())
}

pub fn music_url_workbench_http_response(
    method: &str,
    path: &str,
    content_type: Option<&str>,
    body: &[u8],
    generated_dir: impl AsRef<Path>,
) -> MusicUrlWorkbenchHttpResponse {
    let path = path.split('?').next().unwrap_or(path);
    if method == "GET"
        && matches!(
            path,
            "/" | "/music" | "/music/" | crate::des::service::MUSIC_URL_SEED_WORKBENCH_ROUTE
        )
    {
        return MusicUrlWorkbenchHttpResponse::html(render_music_url_seed_form_document_html(
            crate::des::service::MUSIC_URL_SEED_WORKBENCH_ENDPOINT,
        ));
    }
    if method == "GET" && path == crate::des::service::MUSIC_URL_SEED_CONTRACT_ROUTE {
        return MusicUrlWorkbenchHttpResponse::json(
            200,
            "OK",
            music_url_seed_endpoint_contract_json(
                crate::des::service::MUSIC_URL_SEED_WORKBENCH_ENDPOINT,
            ),
        );
    }
    if method == "GET" {
        if let Some(filename) = path.strip_prefix("/music/generated/") {
            if !is_safe_generated_wav_filename(filename) {
                return MusicUrlWorkbenchHttpResponse::text(404, "Not Found", "not found");
            }
            let wav_path = generated_dir.as_ref().join(filename);
            return match std::fs::read(&wav_path) {
                Ok(bytes) => MusicUrlWorkbenchHttpResponse::wav(bytes),
                Err(_) => MusicUrlWorkbenchHttpResponse::text(404, "Not Found", "not found"),
            };
        }
    }
    if method == "POST" && path == crate::des::service::MUSIC_URL_SEED_ROUTE {
        let Some(content_type) = content_type else {
            return MusicUrlWorkbenchHttpResponse::json(
                400,
                "Bad Request",
                music_url_seed_error_json("invalid_request", "missing Content-Type"),
            );
        };
        let fields = match parse_music_url_seed_form_fields(content_type, body) {
            Ok(fields) => fields,
            Err(err) => {
                return MusicUrlWorkbenchHttpResponse::json(
                    400,
                    "Bad Request",
                    music_url_seed_error_json("invalid_request", &err),
                )
            }
        };
        let field_refs: Vec<(&str, &str)> = fields
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
            .collect();
        let filename = unique_music_url_seed_wav_filename();
        let out_path = generated_dir.as_ref().join(&filename);
        return match render_music_url_seed_wav(&field_refs, &out_path) {
            Ok(result) => MusicUrlWorkbenchHttpResponse::json(
                200,
                "OK",
                music_url_seed_render_result_json(&result, &format!("/music/generated/{filename}")),
            ),
            Err(err) => MusicUrlWorkbenchHttpResponse::json(
                400,
                "Bad Request",
                music_url_seed_error_json("invalid_request", &err),
            ),
        };
    }
    MusicUrlWorkbenchHttpResponse::text(404, "Not Found", "not found")
}

struct BasicHttpRequest {
    method: String,
    path: String,
    content_type: Option<String>,
    body: Vec<u8>,
}

fn handle_music_url_workbench_stream(
    stream: &mut TcpStream,
    generated_dir: &Path,
) -> io::Result<()> {
    let request = read_basic_http_request(stream)?;
    let response = music_url_workbench_http_response(
        &request.method,
        &request.path,
        request.content_type.as_deref(),
        &request.body,
        generated_dir,
    );
    write_basic_http_response(stream, &response)
}

fn read_basic_http_request(stream: &mut TcpStream) -> io::Result<BasicHttpRequest> {
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    let mut data = Vec::new();
    let mut chunk = [0u8; 8192];
    let mut header_end = None;
    let mut content_length = 0usize;
    loop {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        data.extend_from_slice(&chunk[..n]);
        if header_end.is_none() {
            if let Some(end) = find_http_header_end(&data) {
                header_end = Some(end);
                content_length = parse_content_length(&data[..end]).unwrap_or(0);
            }
        }
        if let Some(end) = header_end {
            if data.len() >= end + content_length {
                break;
            }
        }
        if data.len() > 1_048_576 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "request body exceeds 1 MiB limit",
            ));
        }
    }
    let Some(header_end) = header_end else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "missing HTTP headers",
        ));
    };
    let header_text = String::from_utf8_lossy(&data[..header_end]);
    let mut lines = header_text.lines();
    let first_line = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing request line"))?;
    let mut request_parts = first_line.split_whitespace();
    let method = request_parts
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing method"))?
        .to_string();
    let path = request_parts
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing path"))?
        .to_string();
    let content_type = header_text.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        if key.eq_ignore_ascii_case("content-type") {
            Some(value.trim().to_string())
        } else {
            None
        }
    });
    let body_end = (header_end + content_length).min(data.len());
    Ok(BasicHttpRequest {
        method,
        path,
        content_type,
        body: data[header_end..body_end].to_vec(),
    })
}

fn write_basic_http_response(
    stream: &mut TcpStream,
    response: &MusicUrlWorkbenchHttpResponse,
) -> io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        response.status_code,
        response.status_text,
        response.content_type,
        response.body.len()
    )?;
    stream.write_all(&response.body)
}

fn find_http_header_end(data: &[u8]) -> Option<usize> {
    find_bytes(data, b"\r\n\r\n")
        .map(|index| index + 4)
        .or_else(|| find_bytes(data, b"\n\n").map(|index| index + 2))
}

fn parse_content_length(header: &[u8]) -> Option<usize> {
    String::from_utf8_lossy(header).lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        if key.eq_ignore_ascii_case("content-length") {
            value.trim().parse().ok()
        } else {
            None
        }
    })
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn parse_music_url_seed_form_fields(
    content_type: &str,
    body: &[u8],
) -> Result<Vec<(String, String)>, String> {
    let normalized = content_type.to_ascii_lowercase();
    if normalized.starts_with("multipart/form-data") {
        let boundary = multipart_boundary(content_type)
            .ok_or_else(|| "multipart form is missing a boundary".to_string())?;
        parse_multipart_text_fields(&boundary, body)
    } else if normalized.starts_with("application/x-www-form-urlencoded") {
        parse_urlencoded_text_fields(body)
    } else {
        Err(format!("unsupported form content type {content_type:?}"))
    }
}

fn multipart_boundary(content_type: &str) -> Option<String> {
    content_type.split(';').skip(1).find_map(|part| {
        let part = part.trim();
        let value = part.strip_prefix("boundary=")?;
        Some(value.trim_matches('"').to_string())
    })
}

fn parse_multipart_text_fields(
    boundary: &str,
    body: &[u8],
) -> Result<Vec<(String, String)>, String> {
    if boundary.is_empty() {
        return Err("multipart boundary is empty".to_string());
    }
    let text = String::from_utf8(body.to_vec())
        .map_err(|_| "multipart form fields must be UTF-8 text".to_string())?;
    let marker = format!("--{boundary}");
    let mut fields = Vec::new();
    for raw_part in text.split(&marker).skip(1) {
        let part = raw_part.trim_start_matches("\r\n").trim_start_matches('\n');
        if part.starts_with("--") || part.trim().is_empty() {
            continue;
        }
        let (headers, value) = part
            .split_once("\r\n\r\n")
            .or_else(|| part.split_once("\n\n"))
            .ok_or_else(|| "multipart field is missing headers".to_string())?;
        let Some(name) = multipart_field_name(headers) else {
            continue;
        };
        let value = value
            .trim_end_matches("\r\n")
            .trim_end_matches('\n')
            .trim_end_matches('\r')
            .to_string();
        fields.push((name, value));
    }
    Ok(fields)
}

fn multipart_field_name(headers: &str) -> Option<String> {
    for line in headers.lines() {
        let lower = line.to_ascii_lowercase();
        if !lower.starts_with("content-disposition:") || !lower.contains("form-data") {
            continue;
        }
        let marker = "name=\"";
        let start = line.find(marker)? + marker.len();
        let end = line[start..].find('"')? + start;
        return Some(line[start..end].to_string());
    }
    None
}

fn parse_urlencoded_text_fields(body: &[u8]) -> Result<Vec<(String, String)>, String> {
    let text = String::from_utf8(body.to_vec())
        .map_err(|_| "urlencoded form must be UTF-8 text".to_string())?;
    let mut fields = Vec::new();
    for pair in text.split('&').filter(|pair| !pair.is_empty()) {
        let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
        fields.push((percent_decode_form(key)?, percent_decode_form(value)?));
    }
    Ok(fields)
}

fn percent_decode_form(value: &str) -> Result<String, String> {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' => {
                if i + 2 >= bytes.len() {
                    return Err("truncated percent escape in form value".to_string());
                }
                let hi = hex_value(bytes[i + 1])
                    .ok_or_else(|| "invalid percent escape in form value".to_string())?;
                let lo = hex_value(bytes[i + 2])
                    .ok_or_else(|| "invalid percent escape in form value".to_string())?;
                out.push((hi << 4) | lo);
                i += 3;
            }
            other => {
                out.push(other);
                i += 1;
            }
        }
    }
    String::from_utf8(out).map_err(|_| "decoded form value is not UTF-8".to_string())
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn unique_music_url_seed_wav_filename() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("music-url-seed-{}-{millis}.wav", std::process::id())
}

fn is_safe_generated_wav_filename(filename: &str) -> bool {
    !filename.is_empty()
        && filename.len() <= 160
        && filename.ends_with(".wav")
        && filename
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b'_'))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn body_text(response: &MusicUrlWorkbenchHttpResponse) -> String {
        String::from_utf8(response.body.clone()).expect("response body should be text")
    }

    #[test]
    fn music_url_workbench_http_serves_form_and_contract() {
        let form = music_url_workbench_http_response(
            "GET",
            "/music/url-source-inputs.html",
            None,
            &[],
            "out/test-music-url-http",
        );
        assert_eq!(form.status_code, 200);
        assert_eq!(form.content_type, "text/html; charset=utf-8");
        let html = body_text(&form);
        for expected in [
            "youtube_url",
            "facebook_url",
            "instagram_url",
            "s3_url",
            "cloudfront_url",
            "cloudflare_url",
            "static_asset_url",
            "any_audio_url",
            r#"const MUSIC_URL_SEED_ENDPOINT = "sample-seed""#,
        ] {
            assert!(html.contains(expected), "missing form token {expected}");
        }

        let contract = music_url_workbench_http_response(
            "GET",
            "/music/url-source-contract.json",
            None,
            &[],
            "out/test-music-url-http",
        );
        assert_eq!(contract.status_code, 200);
        let json = body_text(&contract);
        assert!(json.contains(r#""endpoint": "sample-seed""#));
        assert!(json.contains(r#""id": "cloudflare_url""#));
    }

    #[test]
    fn music_url_workbench_http_renders_multipart_post_and_serves_wav() {
        let out_dir = PathBuf::from("out/test-music-url-http-render");
        let _ = std::fs::remove_dir_all(&out_dir);
        let boundary = "music-url-seed-test-boundary";
        let body = format!(
            concat!(
                "--{b}\r\nContent-Disposition: form-data; name=\"any_audio_url\"\r\n\r\nhttps://media.example.net/audio/beat.flac\r\n",
                "--{b}\r\nContent-Disposition: form-data; name=\"duration_seconds\"\r\n\r\n15\r\n",
                "--{b}\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nmake a new variation\r\n",
                "--{b}--\r\n"
            ),
            b = boundary
        );
        let response = music_url_workbench_http_response(
            "POST",
            "/music/sample-seed",
            Some(&format!("multipart/form-data; boundary={boundary}")),
            body.as_bytes(),
            &out_dir,
        );
        assert_eq!(response.status_code, 200);
        let json = body_text(&response);
        assert!(json.contains(r#""ok":true"#), "{json}");
        assert!(json.contains(r#""source_kind":"direct-audio""#), "{json}");
        assert!(
            json.contains(r#""wav_url":"/music/generated/music-url-seed-"#),
            "{json}"
        );

        let wav_name = std::fs::read_dir(&out_dir)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .find(|name| name.ends_with(".wav"))
            .expect("render should write a wav");
        let wav_response = music_url_workbench_http_response(
            "GET",
            &format!("/music/generated/{wav_name}"),
            None,
            &[],
            &out_dir,
        );
        assert_eq!(wav_response.status_code, 200);
        assert_eq!(wav_response.content_type, "audio/wav");
        assert!(wav_response.body.starts_with(b"RIFF"));
    }

    #[test]
    fn music_url_workbench_http_rejects_private_urlencoded_source() {
        let response = music_url_workbench_http_response(
            "POST",
            "/music/sample-seed",
            Some("application/x-www-form-urlencoded"),
            b"any_audio_url=http%3A%2F%2F127.0.0.1%2Fseed.mp3&duration_seconds=15",
            "out/test-music-url-http",
        );
        assert_eq!(response.status_code, 400);
        let json = body_text(&response);
        assert!(json.contains(r#""ok":false"#), "{json}");
        assert!(json.contains("localhost/private/internal"), "{json}");
    }
}
