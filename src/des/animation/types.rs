//! Port of `src/des/animation/types.ts`.
//!
//! The wire/data schema for the animation player (frames, shapes, charts) —
//! pure DTOs, plus the small number-formatting helpers the scene builders use.
//!
//! ## Rust shape
//!
//! TypeScript's discriminated union `Shape = Circle | Rect | Line | Text |
//! Path` (tagged on the string field `kind`) becomes an [`enum Shape`] whose
//! variants wrap one struct each. Optional (`?`) fields become `Option<T>`.
//!
//! ## JSON
//!
//! `serde` is deliberately **not** a dependency of this crate (the engine
//! hand-rolls JSON via [`crate::des::observability::logger`]). So instead of
//! `#[derive(Serialize, Deserialize)]` each type carries an explicit
//! `to_json` / `from_json` against [`JsonValue`]. Keys are emitted in the
//! TypeScript object-literal order, camelCased to match the embedded player
//! JS (`strokeWidth`, `visualBlockId`, `fontSize`, …), and `None`s are skipped
//! exactly as `JSON.stringify` drops `undefined` fields — so the produced JSON
//! matches the TS output.

#![allow(dead_code)]

use crate::des::observability::logger::JsonValue;

// =============================================================================
// Number-formatting helpers (the JS `Number` methods the scenes lean on).
// =============================================================================

/// `String(n)` for a `number` — finite values use the shortest round-tripping
/// decimal (Rust `{}` shares JS's algorithm: `3.0 -> "3"`, `0.1 -> "0.1"`);
/// non-finite values mirror JS (`Infinity` / `-Infinity` / `NaN`).
pub fn js_num(n: f64) -> String {
    if n.is_nan() {
        "NaN".to_string()
    } else if n == f64::INFINITY {
        "Infinity".to_string()
    } else if n == f64::NEG_INFINITY {
        "-Infinity".to_string()
    } else {
        format!("{n}")
    }
}

/// `n.toFixed(digits)` — fixed-point string with `digits` decimals. Rust's
/// formatter is used directly (it agrees with V8 for all but exact-half ties,
/// which the binary `f64`s these scenes format essentially never hit). Negative
/// zero is normalised to `0` to match `(-0).toFixed(d)`.
pub fn to_fixed(n: f64, digits: usize) -> String {
    if !n.is_finite() {
        return js_num(n);
    }
    let n = if n == 0.0 { 0.0 } else { n };
    format!("{n:.digits$}")
}

/// `n.toExponential(digits)` — exponential notation with `digits` mantissa
/// decimals and a signed exponent (`1.234e-3`, `1.200e+0`), matching JS.
pub fn to_exponential(n: f64, digits: usize) -> String {
    if !n.is_finite() {
        return js_num(n);
    }
    if n == 0.0 {
        return format!("{:.*}e+0", digits, 0.0);
    }
    let s = format!("{n:.digits$e}");
    match s.split_once('e') {
        Some((mant, exp)) => {
            let e: i32 = exp.parse().unwrap_or(0);
            let sign = if e < 0 { '-' } else { '+' };
            format!("{mant}e{sign}{}", e.abs())
        }
        None => s,
    }
}

// =============================================================================
// Enumerated string-literal unions.
// =============================================================================

/// `'start' | 'middle' | 'end'` text anchor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Anchor {
    Start,
    Middle,
    End,
}

impl Anchor {
    pub fn as_str(self) -> &'static str {
        match self {
            Anchor::Start => "start",
            Anchor::Middle => "middle",
            Anchor::End => "end",
        }
    }

    pub fn from_str(s: &str) -> Option<Anchor> {
        match s {
            "start" => Some(Anchor::Start),
            "middle" => Some(Anchor::Middle),
            "end" => Some(Anchor::End),
            _ => None,
        }
    }
}

/// `'normal' | 'bold'` font weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FontWeight {
    Normal,
    Bold,
}

impl FontWeight {
    pub fn as_str(self) -> &'static str {
        match self {
            FontWeight::Normal => "normal",
            FontWeight::Bold => "bold",
        }
    }

    pub fn from_str(s: &str) -> Option<FontWeight> {
        match s {
            "normal" => Some(FontWeight::Normal),
            "bold" => Some(FontWeight::Bold),
            _ => None,
        }
    }
}

// =============================================================================
// Shapes.
// =============================================================================

#[derive(Clone, Debug, Default, PartialEq)]
pub struct CircleShape {
    pub x: f64,
    pub y: f64,
    pub r: f64,
    pub fill: String,
    pub stroke: Option<String>,
    pub stroke_width: Option<f64>,
    pub opacity: Option<f64>,
    pub label: Option<String>,
    /// Optional SVG-style title (hover text).
    pub title: Option<String>,
    /// `VisualBlock` id when this shape is part of an always-rendered block.
    pub visual_block_id: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RectShape {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
    pub fill: String,
    pub stroke: Option<String>,
    pub stroke_width: Option<f64>,
    pub opacity: Option<f64>,
    pub label: Option<String>,
    /// Optional rounded-corner radius.
    pub rx: Option<f64>,
    pub title: Option<String>,
    pub visual_block_id: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LineShape {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
    pub stroke: String,
    pub stroke_width: Option<f64>,
    pub opacity: Option<f64>,
    /// "5,3" for dashed, omit for solid.
    pub dasharray: Option<String>,
    pub visual_block_id: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TextShape {
    pub x: f64,
    pub y: f64,
    pub text: String,
    pub font_size: Option<f64>,
    pub fill: Option<String>,
    pub anchor: Option<Anchor>,
    pub font_weight: Option<FontWeight>,
    pub font_family: Option<String>,
    pub visual_block_id: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PathShape {
    /// SVG path data.
    pub d: String,
    pub stroke: Option<String>,
    pub stroke_width: Option<f64>,
    pub fill: Option<String>,
    pub opacity: Option<f64>,
    pub visual_block_id: Option<String>,
}

/// `type Shape = CircleShape | RectShape | LineShape | TextShape | PathShape`,
/// tagged on `kind`.
#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    Circle(CircleShape),
    Rect(RectShape),
    Line(LineShape),
    Text(TextShape),
    Path(PathShape),
}

// --- small JSON object-builder helpers (skip `None`, matching `JSON.stringify`).

fn num(o: &mut Vec<(String, JsonValue)>, k: &str, v: f64) {
    o.push((k.to_string(), JsonValue::Number(v)));
}
fn string(o: &mut Vec<(String, JsonValue)>, k: &str, v: &str) {
    o.push((k.to_string(), JsonValue::String(v.to_string())));
}
fn opt_num(o: &mut Vec<(String, JsonValue)>, k: &str, v: Option<f64>) {
    if let Some(x) = v {
        o.push((k.to_string(), JsonValue::Number(x)));
    }
}
fn opt_str(o: &mut Vec<(String, JsonValue)>, k: &str, v: &Option<String>) {
    if let Some(x) = v {
        o.push((k.to_string(), JsonValue::String(x.clone())));
    }
}

fn jget_f64(o: &JsonValue, k: &str) -> Option<f64> {
    o.get(k).and_then(|v| v.as_f64())
}
fn jget_str(o: &JsonValue, k: &str) -> Option<String> {
    o.get(k).and_then(|v| v.as_str()).map(str::to_string)
}

impl Shape {
    pub fn to_json(&self) -> JsonValue {
        let mut o: Vec<(String, JsonValue)> = Vec::new();
        match self {
            Shape::Circle(s) => {
                string(&mut o, "kind", "circle");
                num(&mut o, "x", s.x);
                num(&mut o, "y", s.y);
                num(&mut o, "r", s.r);
                string(&mut o, "fill", &s.fill);
                opt_str(&mut o, "stroke", &s.stroke);
                opt_num(&mut o, "strokeWidth", s.stroke_width);
                opt_num(&mut o, "opacity", s.opacity);
                opt_str(&mut o, "label", &s.label);
                opt_str(&mut o, "title", &s.title);
                opt_str(&mut o, "visualBlockId", &s.visual_block_id);
            }
            Shape::Rect(s) => {
                string(&mut o, "kind", "rect");
                num(&mut o, "x", s.x);
                num(&mut o, "y", s.y);
                num(&mut o, "w", s.w);
                num(&mut o, "h", s.h);
                string(&mut o, "fill", &s.fill);
                opt_str(&mut o, "stroke", &s.stroke);
                opt_num(&mut o, "strokeWidth", s.stroke_width);
                opt_num(&mut o, "opacity", s.opacity);
                opt_str(&mut o, "label", &s.label);
                opt_num(&mut o, "rx", s.rx);
                opt_str(&mut o, "title", &s.title);
                opt_str(&mut o, "visualBlockId", &s.visual_block_id);
            }
            Shape::Line(s) => {
                string(&mut o, "kind", "line");
                num(&mut o, "x1", s.x1);
                num(&mut o, "y1", s.y1);
                num(&mut o, "x2", s.x2);
                num(&mut o, "y2", s.y2);
                string(&mut o, "stroke", &s.stroke);
                opt_num(&mut o, "strokeWidth", s.stroke_width);
                opt_num(&mut o, "opacity", s.opacity);
                opt_str(&mut o, "dasharray", &s.dasharray);
                opt_str(&mut o, "visualBlockId", &s.visual_block_id);
            }
            Shape::Text(s) => {
                string(&mut o, "kind", "text");
                num(&mut o, "x", s.x);
                num(&mut o, "y", s.y);
                string(&mut o, "text", &s.text);
                opt_num(&mut o, "fontSize", s.font_size);
                opt_str(&mut o, "fill", &s.fill);
                if let Some(a) = s.anchor {
                    string(&mut o, "anchor", a.as_str());
                }
                if let Some(w) = s.font_weight {
                    string(&mut o, "fontWeight", w.as_str());
                }
                opt_str(&mut o, "fontFamily", &s.font_family);
                opt_str(&mut o, "visualBlockId", &s.visual_block_id);
            }
            Shape::Path(s) => {
                string(&mut o, "kind", "path");
                string(&mut o, "d", &s.d);
                opt_str(&mut o, "stroke", &s.stroke);
                opt_num(&mut o, "strokeWidth", s.stroke_width);
                opt_str(&mut o, "fill", &s.fill);
                opt_num(&mut o, "opacity", s.opacity);
                opt_str(&mut o, "visualBlockId", &s.visual_block_id);
            }
        }
        JsonValue::Object(o)
    }

    pub fn from_json(v: &JsonValue) -> Option<Shape> {
        let kind = v.get("kind").and_then(|k| k.as_str())?;
        let visual_block_id = jget_str(v, "visualBlockId");
        match kind {
            "circle" => Some(Shape::Circle(CircleShape {
                x: jget_f64(v, "x").unwrap_or(0.0),
                y: jget_f64(v, "y").unwrap_or(0.0),
                r: jget_f64(v, "r").unwrap_or(0.0),
                fill: jget_str(v, "fill").unwrap_or_default(),
                stroke: jget_str(v, "stroke"),
                stroke_width: jget_f64(v, "strokeWidth"),
                opacity: jget_f64(v, "opacity"),
                label: jget_str(v, "label"),
                title: jget_str(v, "title"),
                visual_block_id,
            })),
            "rect" => Some(Shape::Rect(RectShape {
                x: jget_f64(v, "x").unwrap_or(0.0),
                y: jget_f64(v, "y").unwrap_or(0.0),
                w: jget_f64(v, "w").unwrap_or(0.0),
                h: jget_f64(v, "h").unwrap_or(0.0),
                fill: jget_str(v, "fill").unwrap_or_default(),
                stroke: jget_str(v, "stroke"),
                stroke_width: jget_f64(v, "strokeWidth"),
                opacity: jget_f64(v, "opacity"),
                label: jget_str(v, "label"),
                rx: jget_f64(v, "rx"),
                title: jget_str(v, "title"),
                visual_block_id,
            })),
            "line" => Some(Shape::Line(LineShape {
                x1: jget_f64(v, "x1").unwrap_or(0.0),
                y1: jget_f64(v, "y1").unwrap_or(0.0),
                x2: jget_f64(v, "x2").unwrap_or(0.0),
                y2: jget_f64(v, "y2").unwrap_or(0.0),
                stroke: jget_str(v, "stroke").unwrap_or_default(),
                stroke_width: jget_f64(v, "strokeWidth"),
                opacity: jget_f64(v, "opacity"),
                dasharray: jget_str(v, "dasharray"),
                visual_block_id,
            })),
            "text" => Some(Shape::Text(TextShape {
                x: jget_f64(v, "x").unwrap_or(0.0),
                y: jget_f64(v, "y").unwrap_or(0.0),
                text: jget_str(v, "text").unwrap_or_default(),
                font_size: jget_f64(v, "fontSize"),
                fill: jget_str(v, "fill"),
                anchor: jget_str(v, "anchor").as_deref().and_then(Anchor::from_str),
                font_weight: jget_str(v, "fontWeight").as_deref().and_then(FontWeight::from_str),
                font_family: jget_str(v, "fontFamily"),
                visual_block_id,
            })),
            "path" => Some(Shape::Path(PathShape {
                d: jget_str(v, "d").unwrap_or_default(),
                stroke: jget_str(v, "stroke"),
                stroke_width: jget_f64(v, "strokeWidth"),
                fill: jget_str(v, "fill"),
                opacity: jget_f64(v, "opacity"),
                visual_block_id,
            })),
            _ => None,
        }
    }
}

// =============================================================================
// Frames.
// =============================================================================

/// Per-tick scene snapshot: `{ t, tick, shapes, caption? }`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Frame {
    /// Simulation time.
    pub t: f64,
    /// Tick index (integer, kept as `f64` to mirror the JSON `number`).
    pub tick: f64,
    /// SVG shapes that make up this frame.
    pub shapes: Vec<Shape>,
    /// Optional caption shown beneath the stage for this frame.
    pub caption: Option<String>,
}

/// `Omit<Frame, 't' | 'tick'>` — what scene builders return for a single tick
/// (the recorder fills in `t` and `tick`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FrameParts {
    pub shapes: Vec<Shape>,
    pub caption: Option<String>,
}

impl FrameParts {
    pub fn new(shapes: Vec<Shape>) -> Self {
        FrameParts { shapes, caption: None }
    }

    pub fn with_caption(shapes: Vec<Shape>, caption: impl Into<String>) -> Self {
        FrameParts { shapes, caption: Some(caption.into()) }
    }

    /// Promote to a full [`Frame`] by attaching `t` and `tick`.
    pub fn into_frame(self, t: f64, tick: f64) -> Frame {
        Frame { t, tick, shapes: self.shapes, caption: self.caption }
    }
}

impl Frame {
    /// `{ t, tick, shapes, caption? }`.
    pub fn to_json(&self) -> JsonValue {
        let mut o: Vec<(String, JsonValue)> = Vec::new();
        num(&mut o, "t", self.t);
        num(&mut o, "tick", self.tick);
        o.push((
            "shapes".to_string(),
            JsonValue::Array(self.shapes.iter().map(Shape::to_json).collect()),
        ));
        opt_str(&mut o, "caption", &self.caption);
        JsonValue::Object(o)
    }

    pub fn from_json(v: &JsonValue) -> Frame {
        let shapes = v
            .get("shapes")
            .and_then(|s| s.as_array())
            .map(|arr| arr.iter().filter_map(Shape::from_json).collect())
            .unwrap_or_default();
        Frame {
            t: jget_f64(v, "t").unwrap_or(0.0),
            tick: jget_f64(v, "tick").unwrap_or(0.0),
            shapes,
            caption: jget_str(v, "caption"),
        }
    }
}

// =============================================================================
// Charts.
// =============================================================================

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChartSeries {
    pub label: String,
    pub color: String,
    /// Parallel arrays. `t` and `y` must have the same length.
    pub t: Vec<f64>,
    pub y: Vec<f64>,
}

impl ChartSeries {
    pub fn to_json(&self) -> JsonValue {
        let mut o: Vec<(String, JsonValue)> = Vec::new();
        string(&mut o, "label", &self.label);
        string(&mut o, "color", &self.color);
        o.push((
            "t".to_string(),
            JsonValue::Array(self.t.iter().map(|&x| JsonValue::Number(x)).collect()),
        ));
        o.push((
            "y".to_string(),
            JsonValue::Array(self.y.iter().map(|&x| JsonValue::Number(x)).collect()),
        ));
        JsonValue::Object(o)
    }

    pub fn from_json(v: &JsonValue) -> ChartSeries {
        let arr = |k: &str| -> Vec<f64> {
            v.get(k)
                .and_then(|a| a.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
                .unwrap_or_default()
        };
        ChartSeries {
            label: jget_str(v, "label").unwrap_or_default(),
            color: jget_str(v, "color").unwrap_or_default(),
            t: arr("t"),
            y: arr("y"),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChartSpec {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
    pub title: Option<String>,
    pub y_min: Option<f64>,
    pub y_max: Option<f64>,
    /// Display name for the y-axis.
    pub y_label: Option<String>,
    pub series: Vec<ChartSeries>,
    /// If `Some(false)`, suppresses the vertical "current time" cursor.
    pub cursor: Option<bool>,
}

impl ChartSpec {
    pub fn to_json(&self) -> JsonValue {
        let mut o: Vec<(String, JsonValue)> = Vec::new();
        num(&mut o, "x", self.x);
        num(&mut o, "y", self.y);
        num(&mut o, "w", self.w);
        num(&mut o, "h", self.h);
        opt_str(&mut o, "title", &self.title);
        opt_num(&mut o, "yMin", self.y_min);
        opt_num(&mut o, "yMax", self.y_max);
        opt_str(&mut o, "yLabel", &self.y_label);
        o.push((
            "series".to_string(),
            JsonValue::Array(self.series.iter().map(ChartSeries::to_json).collect()),
        ));
        if let Some(c) = self.cursor {
            o.push(("cursor".to_string(), JsonValue::Bool(c)));
        }
        JsonValue::Object(o)
    }

    pub fn from_json(v: &JsonValue) -> ChartSpec {
        let series = v
            .get("series")
            .and_then(|s| s.as_array())
            .map(|arr| arr.iter().map(ChartSeries::from_json).collect())
            .unwrap_or_default();
        ChartSpec {
            x: jget_f64(v, "x").unwrap_or(0.0),
            y: jget_f64(v, "y").unwrap_or(0.0),
            w: jget_f64(v, "w").unwrap_or(0.0),
            h: jget_f64(v, "h").unwrap_or(0.0),
            title: jget_str(v, "title"),
            y_min: jget_f64(v, "yMin"),
            y_max: jget_f64(v, "yMax"),
            y_label: jget_str(v, "yLabel"),
            series,
            cursor: v.get("cursor").and_then(|c| c.as_bool()),
        }
    }
}

// =============================================================================
// Animation (the top-level document).
// =============================================================================

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Animation {
    /// Pixel width of the SVG stage.
    pub width: f64,
    /// Pixel height of the SVG stage.
    pub height: f64,
    /// Default playback rate (frames per second).
    pub fps: f64,
    /// Optional title displayed in the page header.
    pub title: Option<String>,
    /// Optional one-line caption shown beneath the title.
    pub subtitle: Option<String>,
    /// Per-tick scene snapshots.
    pub frames: Vec<Frame>,
    /// Optional global time series panels rendered alongside frames.
    pub charts: Option<Vec<ChartSpec>>,
    /// Optional CSS background color for the stage. Defaults to white.
    pub background: Option<String>,
}

impl Animation {
    /// Serialize the way `JSON.stringify(anim)` would (drops `None` fields).
    pub fn to_json(&self) -> JsonValue {
        let mut o: Vec<(String, JsonValue)> = Vec::new();
        num(&mut o, "width", self.width);
        num(&mut o, "height", self.height);
        num(&mut o, "fps", self.fps);
        opt_str(&mut o, "title", &self.title);
        opt_str(&mut o, "subtitle", &self.subtitle);
        o.push((
            "frames".to_string(),
            JsonValue::Array(self.frames.iter().map(Frame::to_json).collect()),
        ));
        if let Some(charts) = &self.charts {
            o.push((
                "charts".to_string(),
                JsonValue::Array(charts.iter().map(ChartSpec::to_json).collect()),
            ));
        }
        opt_str(&mut o, "background", &self.background);
        JsonValue::Object(o)
    }
}
