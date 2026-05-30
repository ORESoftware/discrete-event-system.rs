//! Port of `src/des/animation/frame-recorder.ts`.
//!
//! Streams per-tick scene snapshots to a JSONL frames file (a header line, one
//! `animation-frame` line per recorded tick, and an optional trailing
//! `animation-charts` line), with an optional live stderr tick line and a
//! one-shot HTML render at [`FrameRecorder::finish`].
//!
//! ## Conversion notes
//!
//! * `fs.WriteStream` + `JSON.stringify(ev) + '\n'` → a [`BufWriter<File>`] and
//!   one [`JsonValue::to_string`] line per event (`writeln!` supplies the `\n`).
//! * The mapped-type `Required<Omit<..>> & Pick<..>` field collapses to plain
//!   resolved fields on the struct.
//! * `frame(t, tick, build)` takes the builder closure as `impl FnOnce() ->
//!   FrameParts`.
//! * `async finish(): Promise<Animation>` → synchronous `finish(&mut self) ->
//!   io::Result<Animation>` (the I/O here is blocking; no async runtime).
//! * `throw new Error(..)` (malformed JSONL / missing header) → `io::Error`
//!   with [`ErrorKind::InvalidData`].
//! * `process.stderr.isTTY` → [`std::io::IsTerminal`].

#![allow(dead_code)]

use std::fs::{self, File};
use std::io::{self, BufWriter, ErrorKind, IsTerminal, Write};
use std::path::Path;
use std::rc::Rc;

use crate::des::animation::html_player::build_html;
use crate::des::animation::types::{
    js_num, to_fixed, Animation, ChartSpec, Frame, FrameParts, Shape,
};
use crate::des::observability::logger::{parse_json, JsonValue};

// =============================================================================
// NOTE: `crate::des::general::des_base::visual_block` is ported, but it models
// `VisualBlockRenderable` as a concrete enum (`VisualBlock | VisualBlockSpec`)
// with a richer render context. The recorder only needs "anything that can emit
// `Vec<Shape>` for a frame", so it keeps this minimal `dyn`-object trait plus
// the `render_visual_blocks` helper — letting scenes supply lightweight ad-hoc
// renderables without constructing a full `VisualBlock`.
// =============================================================================

/// Context handed to a renderable block (`index` is filled per-block by
/// [`render_visual_blocks`]).
#[derive(Clone, Debug, Default)]
pub struct VisualBlockRenderContext {
    pub tick: Option<f64>,
    pub time: Option<f64>,
    pub index: Option<usize>,
    pub stage_width: Option<f64>,
    pub stage_height: Option<f64>,
}

/// Anything the recorder can append to every frame (a `VisualBlock` or a
/// pre-baked `VisualBlockSpec`).
pub trait VisualBlockRenderable {
    fn render_visual_block(&self, ctx: &VisualBlockRenderContext) -> Vec<Shape>;
}

/// `blocks.flatMap((block, index) => block.renderVisualBlock({...ctx, index}))`.
pub fn render_visual_blocks(
    blocks: &[Rc<dyn VisualBlockRenderable>],
    ctx: &VisualBlockRenderContext,
) -> Vec<Shape> {
    let mut out: Vec<Shape> = Vec::new();
    for (index, block) in blocks.iter().enumerate() {
        let mut c = ctx.clone();
        c.index = Some(index);
        out.extend(block.render_visual_block(&c));
    }
    out
}

// =============================================================================
// FrameRecorder options.
// =============================================================================

/// Constructor options. Optional (`?`) fields are `Option<T>`; defaults are
/// applied in [`FrameRecorder::new`].
#[derive(Clone, Default)]
pub struct FrameRecorderOpts {
    /// Path to write a JSONL frames file. Required.
    pub frames_path: String,
    /// Path to write the standalone HTML file at `finish()`. Optional.
    pub html_path: Option<String>,
    /// SVG stage width in pixels.
    pub width: f64,
    /// SVG stage height in pixels.
    pub height: f64,
    /// Playback rate (frames per second). Defaults to 30.
    pub fps: Option<f64>,
    /// Page title.
    pub title: Option<String>,
    /// One-line caption shown under the title.
    pub subtitle: Option<String>,
    /// Background color for the SVG stage. Defaults to '#fff'.
    pub background: Option<String>,
    /// If true, writes a one-line tick summary to stderr each frame.
    pub live_tick_line: Option<bool>,
    /// Record only every Nth tick (default 1).
    pub record_every_ticks: Option<f64>,
    /// Visual blocks appended to every HTML/animation frame.
    pub visual_blocks: Option<Vec<Rc<dyn VisualBlockRenderable>>>,
}

// =============================================================================
// FrameRecorder.
// =============================================================================

/// Emits per-tick scene snapshots to three sinks: a JSONL frames file
/// (always), an optional live stderr tick line, and an optional one-shot HTML
/// render at `finish()`.
pub struct FrameRecorder {
    // Resolved options.
    frames_path: String,
    html_path: Option<String>,
    width: f64,
    height: f64,
    fps: f64,
    title: String,
    subtitle: Option<String>,
    background: String,
    live_tick_line: bool,
    record_every_ticks: f64,
    visual_blocks: Vec<Rc<dyn VisualBlockRenderable>>,

    stream: Option<BufWriter<File>>,
    charts: Vec<ChartSpec>,
    frame_count: u64,
    last_live_line: String,
}

impl FrameRecorder {
    /// Create the recorder, make the frames file's parent directory, open the
    /// file (truncating), and write the `animation-header` line.
    pub fn new(opts: FrameRecorderOpts) -> io::Result<FrameRecorder> {
        let mut rec = FrameRecorder {
            frames_path: opts.frames_path.clone(),
            html_path: opts.html_path,
            width: opts.width,
            height: opts.height,
            fps: opts.fps.unwrap_or(30.0),
            title: opts.title.unwrap_or_else(|| "Simulation".to_string()),
            subtitle: opts.subtitle,
            background: opts.background.unwrap_or_else(|| "#ffffff".to_string()),
            live_tick_line: opts.live_tick_line.unwrap_or(false),
            record_every_ticks: opts.record_every_ticks.unwrap_or(1.0),
            visual_blocks: opts.visual_blocks.unwrap_or_default(),
            stream: None,
            charts: Vec::new(),
            frame_count: 0,
            last_live_line: String::new(),
        };

        if let Some(parent) = Path::new(&rec.frames_path).parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let mut writer = BufWriter::new(File::create(&rec.frames_path)?);

        // Header line: kind=animation-header.
        let mut header: Vec<(String, JsonValue)> = vec![
            (
                "kind".to_string(),
                JsonValue::String("animation-header".to_string()),
            ),
            ("width".to_string(), JsonValue::Number(rec.width)),
            ("height".to_string(), JsonValue::Number(rec.height)),
            ("fps".to_string(), JsonValue::Number(rec.fps)),
            ("title".to_string(), JsonValue::String(rec.title.clone())),
        ];
        if let Some(sub) = &rec.subtitle {
            header.push(("subtitle".to_string(), JsonValue::String(sub.clone())));
        }
        header.push((
            "background".to_string(),
            JsonValue::String(rec.background.clone()),
        ));
        writeln!(writer, "{}", JsonValue::Object(header))?;

        rec.stream = Some(writer);
        Ok(rec)
    }

    /// Record a frame. The builder is only invoked when this tick is eligible
    /// (`tick % record_every_ticks == 0`). The builder returns this tick's
    /// shapes (and optional caption); the recorder fills in `t` and `tick`.
    pub fn frame<F: FnOnce() -> FrameParts>(&mut self, t: f64, tick: f64, build: F) {
        if tick % self.record_every_ticks != 0.0 {
            return;
        }
        let built = build();
        let raw_caption = built.caption.clone();
        let visual_shapes = if !self.visual_blocks.is_empty() {
            render_visual_blocks(
                &self.visual_blocks,
                &VisualBlockRenderContext {
                    tick: Some(tick),
                    time: Some(t),
                    index: None,
                    stage_width: Some(self.width),
                    stage_height: Some(self.height),
                },
            )
        } else {
            Vec::new()
        };
        let mut shapes = built.shapes;
        shapes.extend(visual_shapes);
        // `...(built.caption ? {caption} : {})` — JS treats "" as falsy.
        let caption = built.caption.filter(|c| !c.is_empty());
        let f = Frame {
            t,
            tick,
            shapes,
            caption,
        };

        if let Some(w) = self.stream.as_mut() {
            let _ = writeln!(w, "{}", frame_event(&f));
        }
        self.frame_count += 1;
        if self.live_tick_line {
            self.write_live_line(t, tick, raw_caption.as_deref());
        }
    }

    /// Set the global time-series chart panels. May be called any time before
    /// `finish()`.
    pub fn set_charts(&mut self, charts: Vec<ChartSpec>) {
        self.charts = charts;
    }

    /// Add a single chart panel.
    pub fn add_chart(&mut self, c: ChartSpec) {
        self.charts.push(c);
    }

    /// Number of frames recorded so far.
    pub fn get_frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Flush the JSONL stream, write the HTML output (if configured), and
    /// return the in-memory [`Animation`].
    pub fn finish(&mut self) -> io::Result<Animation> {
        if let Some(w) = self.stream.as_mut() {
            if !self.charts.is_empty() {
                let charts_json =
                    JsonValue::Array(self.charts.iter().map(ChartSpec::to_json).collect());
                let event = JsonValue::Object(vec![
                    (
                        "kind".to_string(),
                        JsonValue::String("animation-charts".to_string()),
                    ),
                    ("charts".to_string(), charts_json),
                ]);
                let _ = writeln!(w, "{event}");
            }
        }
        // `await stream.end()` — flush and close.
        if let Some(mut w) = self.stream.take() {
            w.flush()?;
        }

        let anim = read_animation(&self.frames_path)?;
        if let Some(html_path) = &self.html_path {
            if let Some(parent) = Path::new(html_path).parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent)?;
                }
            }
            fs::write(html_path, build_html(&anim))?;
        }

        if self.live_tick_line {
            // End the live line with a newline so later stderr output is clean.
            eprintln!();
        }
        Ok(anim)
    }

    fn write_live_line(&mut self, t: f64, tick: f64, caption: Option<&str>) {
        if !io::stderr().is_terminal() {
            return;
        }
        let cap = match caption {
            Some(c) if !c.is_empty() => format!("  {c}"),
            _ => String::new(),
        };
        let line = format!(
            "[anim] t={}  tick={}  frames={}{}",
            to_fixed(t, 2),
            js_num(tick),
            self.frame_count,
            cap
        );
        // Pad to overwrite previous line if shorter.
        let padding = self.last_live_line.len().saturating_sub(line.len());
        eprint!("\r{}{}", line, " ".repeat(padding));
        let _ = io::stderr().flush();
        self.last_live_line = line;
    }
}

/// `{kind: 'animation-frame', ...f}` — `kind` first, then the frame fields.
fn frame_event(f: &Frame) -> JsonValue {
    let mut entries: Vec<(String, JsonValue)> = vec![(
        "kind".to_string(),
        JsonValue::String("animation-frame".to_string()),
    )];
    if let JsonValue::Object(fields) = f.to_json() {
        entries.extend(fields);
    }
    JsonValue::Object(entries)
}

/// Reconstruct an [`Animation`] from a JSONL frames file. Tolerant of unknown
/// event kinds (so simulations may interleave other observability events into
/// the same file).
pub fn read_animation(frames_path: &str) -> io::Result<Animation> {
    let raw = fs::read_to_string(frames_path)?;
    let mut header: Option<JsonValue> = None;
    let mut frames: Vec<Frame> = Vec::new();
    let mut charts: Option<Vec<ChartSpec>> = None;
    for line in raw.split('\n') {
        if line.is_empty() {
            continue;
        }
        let ev = parse_json(line).map_err(|e| {
            io::Error::new(
                ErrorKind::InvalidData,
                format!("malformed JSONL in {frames_path}: {e}"),
            )
        })?;
        match ev.get("kind").and_then(|k| k.as_str()) {
            Some("animation-header") => header = Some(ev),
            Some("animation-frame") => frames.push(Frame::from_json(&ev)),
            Some("animation-charts") => {
                charts = ev
                    .get("charts")
                    .and_then(|c| c.as_array())
                    .map(|arr| arr.iter().map(ChartSpec::from_json).collect());
            }
            _ => {}
        }
    }
    let header = header.ok_or_else(|| {
        io::Error::new(
            ErrorKind::InvalidData,
            format!("{frames_path} contains no animation-header event"),
        )
    })?;
    Ok(Animation {
        width: header.get("width").and_then(|v| v.as_f64()).unwrap_or(0.0),
        height: header.get("height").and_then(|v| v.as_f64()).unwrap_or(0.0),
        fps: header.get("fps").and_then(|v| v.as_f64()).unwrap_or(0.0),
        title: header
            .get("title")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        subtitle: header
            .get("subtitle")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        background: header
            .get("background")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        frames,
        charts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::animation::types::{Shape, TextShape};

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("des_anim_{}_{}", std::process::id(), name));
        p
    }

    #[test]
    fn records_and_reads_back_an_animation() {
        let path = temp_path("rec.frames.jsonl");
        let p = path.to_str().unwrap().to_string();
        {
            let mut rec = FrameRecorder::new(FrameRecorderOpts {
                frames_path: p.clone(),
                width: 320.0,
                height: 240.0,
                fps: Some(24.0),
                title: Some("Test".to_string()),
                ..Default::default()
            })
            .expect("create recorder");
            for tick in 0..3 {
                rec.frame(tick as f64 * 0.5, tick as f64, || {
                    FrameParts::with_caption(
                        vec![Shape::Text(TextShape {
                            x: 1.0,
                            y: 2.0,
                            text: format!("t{tick}"),
                            ..Default::default()
                        })],
                        format!("frame {tick}"),
                    )
                });
            }
            rec.set_charts(vec![ChartSpec {
                w: 10.0,
                ..Default::default()
            }]);
            let anim = rec.finish().expect("finish");
            assert_eq!(anim.frames.len(), 3);
            assert_eq!(anim.width, 320.0);
            assert_eq!(anim.fps, 24.0);
            assert_eq!(anim.title.as_deref(), Some("Test"));
            assert!(anim.charts.is_some());
            assert_eq!(anim.frames[1].caption.as_deref(), Some("frame 1"));
        }
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn record_every_ticks_filters() {
        let path = temp_path("every.frames.jsonl");
        let p = path.to_str().unwrap().to_string();
        let mut rec = FrameRecorder::new(FrameRecorderOpts {
            frames_path: p.clone(),
            width: 10.0,
            height: 10.0,
            record_every_ticks: Some(2.0),
            ..Default::default()
        })
        .expect("create recorder");
        for tick in 0..5 {
            rec.frame(tick as f64, tick as f64, || FrameParts::new(vec![]));
        }
        // ticks 0, 2, 4 recorded.
        assert_eq!(rec.get_frame_count(), 3);
        let _ = rec.finish();
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn missing_header_errors() {
        let path = temp_path("noheader.frames.jsonl");
        let p = path.to_str().unwrap().to_string();
        fs::write(
            &p,
            "{\"kind\":\"animation-frame\",\"t\":0,\"tick\":0,\"shapes\":[]}\n",
        )
        .unwrap();
        let err = read_animation(&p).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);
        let _ = fs::remove_file(&p);
    }
}
