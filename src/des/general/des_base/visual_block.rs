//! Port of `src/des/general/des-base/visual-block.ts`
//! (module `des::general::des_base::visual_block`).
//!
//! A visual-editor wrapper over [`CompositeDESStation`]: typed in/out ports,
//! visual connections, layout/style metadata, and SVG-[`Shape`] rendering.
//!
//! ## Conversion notes (faithful to the TS shape)
//!
//!   * The string-union types become enums: [`VisualBlockRole`],
//!     [`VisualPortDirection`]. `VisualPortInput = string | VisualPortOptions` →
//!     the [`VisualPortInput`] enum.
//!   * `class VisualBlock extends CompositeDESStation` → a struct composing a
//!     [`CompositeDESStation`] and `impl DESStation` by delegation; the
//!     overriding `addSubstation` calls the inner method then records the member.
//!   * `Required<VisualBlockLayout>` / `Required<VisualBlockStyle>` (all fields
//!     filled) → the resolved [`ResolvedLayout`] / [`ResolvedStyle`] structs.
//!   * `metadata?: Record<string, unknown>` → `Option<JsonObject>` (the engine's
//!     ordered JSON object from `des_spec`, the faithful `Record<…, unknown>`).
//!   * `throw new Error(...)` invariant checks → `panic!`.
//!
//! PORT NOTE: `memberKind` used `member.constructor?.name`, which has no Rust
//! analogue. Substation members record `std::any::type_name::<S>()` as their
//! `kind`; explicitly-added members carry a caller-supplied `kind`.
//!
//! PORT NOTE: `connectTo`'s `wireDES` branch called `this.pipe(target, …)`,
//! which needs a shared `StationRef` handle to the target. A `&mut VisualBlock`
//! cannot yield an `Rc<RefCell<dyn DESStation>>` to itself, so the DES pipe is
//! recorded as intent (the visual `VisualBlockConnectionSpec` is always built)
//! but not executed here; wire the underlying stations explicitly via the shared
//! handles when needed.
//!
//! PORT NOTE: `isVisualBlock(value: unknown)` was a structural/duck-typed guard.
//! Rust uses the [`VisualBlockRenderable`] enum tag instead; the ported
//! [`is_visual_block`] reports whether a renderable is block-like (both arms are,
//! since every spec carries `alwaysRenderInHtml: true`).

#![allow(dead_code)]

use std::any::Any;

use crate::des::animation::types::{Anchor, CircleShape, FontWeight, RectShape, Shape, TextShape};
use crate::des::general::des_base::composite_station::CompositeDESStation;
use crate::des::general::des_base::station::{DESStation, StationCore};
use crate::des::general::des_spec::JsonObject;

/// `Record<string, unknown>` opaque metadata bag (insertion-ordered JSON object).
pub type Metadata = JsonObject;

// =============================================================================
// String-union enums.
// =============================================================================

/// `type VisualBlockRole`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisualBlockRole {
    Source,
    Sink,
    Transform,
    Station,
    Composite,
    Observer,
}

impl VisualBlockRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            VisualBlockRole::Source => "source",
            VisualBlockRole::Sink => "sink",
            VisualBlockRole::Transform => "transform",
            VisualBlockRole::Station => "station",
            VisualBlockRole::Composite => "composite",
            VisualBlockRole::Observer => "observer",
        }
    }
}

/// `type VisualPortDirection`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisualPortDirection {
    In,
    Out,
}

impl VisualPortDirection {
    pub fn as_str(&self) -> &'static str {
        match self {
            VisualPortDirection::In => "in",
            VisualPortDirection::Out => "out",
        }
    }
}

// =============================================================================
// Port / layout / style / connection structs.
// =============================================================================

/// `interface VisualPortOptions`.
#[derive(Clone, Debug, Default)]
pub struct VisualPortOptions {
    pub id: String,
    pub kind: Option<String>,
    pub label: Option<String>,
    pub data_type: Option<String>,
    pub required: Option<bool>,
    pub capacity: Option<i64>,
    pub metadata: Option<Metadata>,
}

/// `type VisualPortInput = string | VisualPortOptions`.
#[derive(Clone, Debug)]
pub enum VisualPortInput {
    Name(String),
    Opts(VisualPortOptions),
}

impl From<&str> for VisualPortInput {
    fn from(s: &str) -> Self {
        VisualPortInput::Name(s.to_string())
    }
}

/// `interface VisualBlockPort`.
#[derive(Clone, Debug)]
pub struct VisualBlockPort {
    pub id: String,
    pub direction: VisualPortDirection,
    pub kind: String,
    pub label: String,
    pub data_type: Option<String>,
    pub required: bool,
    pub capacity: Option<i64>,
    pub metadata: Option<Metadata>,
}

/// `interface VisualBlockPortSpec` (optionals default to empty lists).
#[derive(Clone, Debug, Default)]
pub struct VisualBlockPortSpec {
    pub inputs: Vec<VisualPortInput>,
    pub outputs: Vec<VisualPortInput>,
}

/// `interface VisualBlockLayout` (all optional overrides).
#[derive(Clone, Copy, Debug, Default)]
pub struct VisualBlockLayout {
    pub x: Option<f64>,
    pub y: Option<f64>,
    pub w: Option<f64>,
    pub h: Option<f64>,
}

/// `Required<VisualBlockLayout>` — every field filled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedLayout {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

impl ResolvedLayout {
    /// `{...self, ...override}`.
    fn merged(self, o: &VisualBlockLayout) -> ResolvedLayout {
        ResolvedLayout {
            x: o.x.unwrap_or(self.x),
            y: o.y.unwrap_or(self.y),
            w: o.w.unwrap_or(self.w),
            h: o.h.unwrap_or(self.h),
        }
    }
}

/// `interface VisualBlockStyle` (all optional overrides).
#[derive(Clone, Debug, Default)]
pub struct VisualBlockStyle {
    pub fill: Option<String>,
    pub stroke: Option<String>,
    pub text: Option<String>,
}

/// `Required<VisualBlockStyle>`.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedStyle {
    pub fill: String,
    pub stroke: String,
    pub text: String,
}

impl ResolvedStyle {
    fn merged(&self, o: &VisualBlockStyle) -> ResolvedStyle {
        ResolvedStyle {
            fill: o.fill.clone().unwrap_or_else(|| self.fill.clone()),
            stroke: o.stroke.clone().unwrap_or_else(|| self.stroke.clone()),
            text: o.text.clone().unwrap_or_else(|| self.text.clone()),
        }
    }
}

/// `{ blockId, portId }`.
#[derive(Clone, Debug)]
pub struct ConnectionEndpoint {
    pub block_id: String,
    pub port_id: String,
}

/// `interface VisualBlockConnectionSpec`.
#[derive(Clone, Debug)]
pub struct VisualBlockConnectionSpec {
    pub id: String,
    pub kind: String,
    pub from: ConnectionEndpoint,
    pub to: ConnectionEndpoint,
    pub metadata: Option<Metadata>,
}

/// `interface VisualBlockConnectionOptions`.
#[derive(Clone, Debug, Default)]
pub struct VisualBlockConnectionOptions {
    pub id: Option<String>,
    pub from_port: Option<String>,
    pub to_port: Option<String>,
    pub kind: Option<String>,
    pub metadata: Option<Metadata>,
    pub wire_des: Option<bool>,
}

/// `interface VisualBlockRenderContext`.
#[derive(Clone, Copy, Debug, Default)]
pub struct VisualBlockRenderContext {
    pub tick: Option<f64>,
    pub time: Option<f64>,
    pub index: Option<usize>,
    pub stage_width: Option<f64>,
    pub stage_height: Option<f64>,
}

/// `contains` entry of a [`VisualBlockSpec`].
#[derive(Clone, Debug)]
pub struct VisualBlockContains {
    pub id: Option<String>,
    pub kind: String,
}

/// Resolved port pair for a spec.
#[derive(Clone, Debug, Default)]
pub struct VisualBlockSpecPorts {
    pub inputs: Vec<VisualBlockPort>,
    pub outputs: Vec<VisualBlockPort>,
}

/// `interface VisualBlockSpec`.
#[derive(Clone, Debug)]
pub struct VisualBlockSpec {
    pub id: String,
    pub kind: String,
    pub role: VisualBlockRole,
    pub label: String,
    /// `alwaysRenderInHtml: true`.
    pub always_render_in_html: bool,
    pub layout: ResolvedLayout,
    pub ports: VisualBlockSpecPorts,
    pub connections_in: Vec<VisualBlockConnectionSpec>,
    pub connections_out: Vec<VisualBlockConnectionSpec>,
    pub contains: Vec<VisualBlockContains>,
    pub style: ResolvedStyle,
    pub metadata: Option<Metadata>,
}

/// `interface VisualBlockOptions`.
#[derive(Clone, Debug, Default)]
pub struct VisualBlockOptions {
    pub kind: Option<String>,
    pub role: Option<VisualBlockRole>,
    pub label: Option<String>,
    pub layout: Option<VisualBlockLayout>,
    pub ports: Option<VisualBlockPortSpec>,
    pub style: Option<VisualBlockStyle>,
    pub metadata: Option<Metadata>,
}

/// `interface VisualBlockInternalOptions` (resolved).
struct VisualBlockInternalOptions {
    kind: String,
    role: VisualBlockRole,
    label: String,
    layout: ResolvedLayout,
    inputs: Vec<VisualBlockPort>,
    outputs: Vec<VisualBlockPort>,
    style: ResolvedStyle,
    metadata: Option<Metadata>,
}

/// `type VisualBlockMember` — only the id + kind the spec extracts are kept.
#[derive(Clone, Debug, PartialEq)]
pub struct VisualBlockMember {
    pub id: Option<String>,
    pub kind: String,
}

const DEFAULT_LAYOUT: ResolvedLayout = ResolvedLayout {
    x: 24.0,
    y: 24.0,
    w: 180.0,
    h: 64.0,
};

fn default_style() -> ResolvedStyle {
    ResolvedStyle {
        fill: "#eef2ff".to_string(),
        stroke: "#4f46e5".to_string(),
        text: "#111827".to_string(),
    }
}

// =============================================================================
// VisualBlock
// =============================================================================

/// `class VisualBlock extends CompositeDESStation`.
pub struct VisualBlock {
    composite: CompositeDESStation,
    pub always_render_in_html: bool,
    visual_members: Vec<VisualBlockMember>,
    visual_connections_in: Vec<VisualBlockConnectionSpec>,
    visual_connections_out: Vec<VisualBlockConnectionSpec>,
    visual_options: VisualBlockInternalOptions,
}

impl VisualBlock {
    /// `constructor(id, opts)`.
    pub fn new(id: &str, opts: VisualBlockOptions) -> Self {
        let role = opts.role.unwrap_or(VisualBlockRole::Station);
        let ports = opts.ports.unwrap_or_default();
        let layout = DEFAULT_LAYOUT.merged(&opts.layout.unwrap_or_default());
        let style = default_style().merged(&opts.style.unwrap_or_default());
        let visual_options = VisualBlockInternalOptions {
            kind: opts.kind.unwrap_or_else(|| "visual-block".to_string()),
            role,
            label: opts.label.unwrap_or_else(|| id.to_string()),
            layout,
            inputs: normalize_ports(&ports.inputs, VisualPortDirection::In),
            outputs: normalize_ports(&ports.outputs, VisualPortDirection::Out),
            style,
            metadata: opts.metadata,
        };
        let vb = VisualBlock {
            composite: CompositeDESStation::new(id),
            always_render_in_html: true,
            visual_members: Vec::new(),
            visual_connections_in: Vec::new(),
            visual_connections_out: Vec::new(),
            visual_options,
        };
        vb.assert_role_ports();
        vb
    }

    /// `static source(id, outputs, opts)`.
    pub fn source(id: &str, outputs: Vec<VisualPortInput>, mut opts: VisualBlockOptions) -> Self {
        opts.role = Some(VisualBlockRole::Source);
        opts.ports = Some(VisualBlockPortSpec {
            inputs: Vec::new(),
            outputs,
        });
        VisualBlock::new(id, opts)
    }

    /// `static sink(id, inputs, opts)`.
    pub fn sink(id: &str, inputs: Vec<VisualPortInput>, mut opts: VisualBlockOptions) -> Self {
        opts.role = Some(VisualBlockRole::Sink);
        opts.ports = Some(VisualBlockPortSpec {
            inputs,
            outputs: Vec::new(),
        });
        VisualBlock::new(id, opts)
    }

    /// This block's id (from the composed station core).
    pub fn id(&self) -> String {
        self.composite.core().id.clone()
    }

    /// `override addSubstation(station)` — add to the composite then record it.
    pub fn add_substation<S: DESStation + 'static>(
        &mut self,
        station: std::rc::Rc<std::cell::RefCell<S>>,
    ) -> std::rc::Rc<std::cell::RefCell<S>> {
        let child = self.composite.add_substation(station);
        let child_id = {
            let b = child.borrow();
            b.id().to_string()
        };
        self.add_visual_member(VisualBlockMember {
            id: Some(child_id),
            // PORT NOTE: closest analogue to `member.constructor.name`.
            kind: std::any::type_name::<S>().to_string(),
        });
        child
    }

    /// `addVisualMember(member)` — dedup by value.
    pub fn add_visual_member(&mut self, member: VisualBlockMember) -> VisualBlockMember {
        if !self.visual_members.contains(&member) {
            self.visual_members.push(member.clone());
        }
        member
    }

    pub fn contained_visual_members(&self) -> &[VisualBlockMember] {
        &self.visual_members
    }

    pub fn visual_input_ports(&self) -> &[VisualBlockPort] {
        &self.visual_options.inputs
    }

    pub fn visual_output_ports(&self) -> &[VisualBlockPort] {
        &self.visual_options.outputs
    }

    /// `addInputPort(port)`.
    pub fn add_input_port(&mut self, port: VisualPortInput) -> VisualBlockPort {
        if self.visual_options.role == VisualBlockRole::Source {
            panic!(
                "VisualBlock({}): source blocks cannot have input ports",
                self.id()
            );
        }
        let normalized = normalize_port(&port, VisualPortDirection::In);
        assert_unique_port(
            &self.visual_options.inputs,
            &normalized.id,
            &self.id(),
            "input",
        );
        self.visual_options.inputs.push(normalized.clone());
        normalized
    }

    /// `addOutputPort(port)`.
    pub fn add_output_port(&mut self, port: VisualPortInput) -> VisualBlockPort {
        if self.visual_options.role == VisualBlockRole::Sink {
            panic!(
                "VisualBlock({}): sink blocks cannot have output ports",
                self.id()
            );
        }
        let normalized = normalize_port(&port, VisualPortDirection::Out);
        assert_unique_port(
            &self.visual_options.outputs,
            &normalized.id,
            &self.id(),
            "output",
        );
        self.visual_options.outputs.push(normalized.clone());
        normalized
    }

    /// `connectTo(target, opts)` — build the visual connection (and record it on
    /// both blocks). See module PORT NOTE re: the `wireDES` DES pipe.
    pub fn connect_to(
        &mut self,
        target: &mut VisualBlock,
        opts: VisualBlockConnectionOptions,
    ) -> VisualBlockConnectionSpec {
        let output = self
            .resolve_port(VisualPortDirection::Out, opts.from_port.as_deref())
            .clone();
        let input = target
            .resolve_port(VisualPortDirection::In, opts.to_port.as_deref())
            .clone();
        let kind = opts.kind.clone().unwrap_or_else(|| output.kind.clone());
        if output.kind != kind {
            panic!(
                "VisualBlock({}): output port \"{}\" has kind \"{}\", not \"{}\"",
                self.id(),
                output.id,
                output.kind,
                kind
            );
        }
        if input.kind != kind {
            panic!(
                "VisualBlock({}): input port \"{}\" has kind \"{}\", not \"{}\"",
                target.id(),
                input.id,
                input.kind,
                kind
            );
        }
        let connection = VisualBlockConnectionSpec {
            id: opts.id.clone().unwrap_or_else(|| {
                format!("{}:{}->{}:{}", self.id(), output.id, target.id(), input.id)
            }),
            kind,
            from: ConnectionEndpoint {
                block_id: self.id(),
                port_id: output.id.clone(),
            },
            to: ConnectionEndpoint {
                block_id: target.id(),
                port_id: input.id.clone(),
            },
            metadata: opts.metadata.clone(),
        };
        self.visual_connections_out.push(connection.clone());
        target.receive_visual_connection(connection.clone());
        // PORT NOTE: `if (opts.wireDES ?? true) this.pipe(target, ...)` — deferred
        // (no shared station handle for `target`); the visual spec is still built.
        let _wire_des = opts.wire_des.unwrap_or(true);
        connection
    }

    pub fn visual_connections_incoming(&self) -> &[VisualBlockConnectionSpec] {
        &self.visual_connections_in
    }

    pub fn visual_connections_outgoing(&self) -> &[VisualBlockConnectionSpec] {
        &self.visual_connections_out
    }

    /// `setVisualLayout(layout): this`.
    pub fn set_visual_layout(&mut self, layout: VisualBlockLayout) -> &mut Self {
        self.visual_options.layout = self.visual_options.layout.merged(&layout);
        self
    }

    /// `visualBlockSpec(overrides)`.
    pub fn visual_block_spec(&self, layout_override: Option<VisualBlockLayout>) -> VisualBlockSpec {
        let layout = match layout_override {
            Some(o) => self.visual_options.layout.merged(&o),
            None => self.visual_options.layout,
        };
        VisualBlockSpec {
            id: self.id(),
            kind: self.visual_options.kind.clone(),
            role: self.visual_options.role,
            label: self.visual_options.label.clone(),
            always_render_in_html: true,
            layout,
            ports: VisualBlockSpecPorts {
                inputs: clone_ports(&self.visual_options.inputs),
                outputs: clone_ports(&self.visual_options.outputs),
            },
            connections_in: self
                .visual_connections_in
                .iter()
                .map(clone_connection)
                .collect(),
            connections_out: self
                .visual_connections_out
                .iter()
                .map(clone_connection)
                .collect(),
            contains: self
                .visual_members
                .iter()
                .map(|m| VisualBlockContains {
                    id: member_id(m),
                    kind: member_kind(m),
                })
                .collect(),
            style: self.visual_options.style.clone(),
            metadata: self.visual_options.metadata.clone(),
        }
    }

    /// `renderVisualBlock(ctx)`.
    pub fn render_visual_block(&self, ctx: VisualBlockRenderContext) -> Vec<Shape> {
        let layout = default_layout_for(ctx.index.unwrap_or(0), self.visual_options.layout);
        render_visual_block_spec(&self.visual_block_spec(Some(VisualBlockLayout {
            x: Some(layout.x),
            y: Some(layout.y),
            w: Some(layout.w),
            h: Some(layout.h),
        })))
    }

    fn receive_visual_connection(&mut self, connection: VisualBlockConnectionSpec) {
        self.visual_connections_in.push(connection);
    }

    fn resolve_port(
        &self,
        direction: VisualPortDirection,
        port_id: Option<&str>,
    ) -> &VisualBlockPort {
        let ports = match direction {
            VisualPortDirection::In => &self.visual_options.inputs,
            VisualPortDirection::Out => &self.visual_options.outputs,
        };
        if let Some(pid) = port_id {
            return ports.iter().find(|p| p.id == pid).unwrap_or_else(|| {
                panic!(
                    "VisualBlock({}): unknown {} port \"{}\"",
                    self.id(),
                    direction.as_str(),
                    pid
                )
            });
        }
        if ports.len() != 1 {
            let hint = if direction == VisualPortDirection::In {
                "toPort"
            } else {
                "fromPort"
            };
            panic!(
                "VisualBlock({}): expected exactly one {} port, found {}; pass {}",
                self.id(),
                direction.as_str(),
                ports.len(),
                hint
            );
        }
        &ports[0]
    }

    fn assert_role_ports(&self) {
        if self.visual_options.role == VisualBlockRole::Source
            && !self.visual_options.inputs.is_empty()
        {
            panic!(
                "VisualBlock({}): source blocks can only define output ports",
                self.id()
            );
        }
        if self.visual_options.role == VisualBlockRole::Sink
            && !self.visual_options.outputs.is_empty()
        {
            panic!(
                "VisualBlock({}): sink blocks can only define input ports",
                self.id()
            );
        }
    }
}

/// `VisualBlock` is a `CompositeDESStation`, hence a [`DESStation`] (delegating
/// to the composed core).
impl DESStation for VisualBlock {
    fn core(&self) -> &StationCore {
        self.composite.core()
    }
    fn core_mut(&mut self) -> &mut StationCore {
        self.composite.core_mut()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn run_time_step(&mut self) {
        self.composite.run_time_step();
    }
}

// =============================================================================
// Free functions.
// =============================================================================

/// `type VisualBlockRenderable = VisualBlock | VisualBlockSpec`.
pub enum VisualBlockRenderable<'a> {
    Block(&'a VisualBlock),
    Spec(&'a VisualBlockSpec),
}

/// `isVisualBlock(value)` — see module PORT NOTE. Both renderable arms are
/// block-like (every spec carries `alwaysRenderInHtml: true`).
pub fn is_visual_block(value: &VisualBlockRenderable) -> bool {
    match value {
        VisualBlockRenderable::Block(_) => true,
        VisualBlockRenderable::Spec(s) => s.always_render_in_html,
    }
}

/// `renderVisualBlocks(blocks, ctx)`.
pub fn render_visual_blocks(
    blocks: &[VisualBlockRenderable],
    ctx: VisualBlockRenderContext,
) -> Vec<Shape> {
    let mut out = Vec::new();
    for (index, block) in blocks.iter().enumerate() {
        match block {
            VisualBlockRenderable::Block(b) => {
                let mut c = ctx;
                c.index = Some(index);
                out.extend(b.render_visual_block(c));
            }
            VisualBlockRenderable::Spec(s) => {
                let layout = default_layout_for(index, s.layout);
                let mut spec = (*s).clone();
                spec.layout = layout;
                out.extend(render_visual_block_spec(&spec));
            }
        }
    }
    out
}

/// `visualBlockSpecs(blocks)`.
pub fn visual_block_specs(blocks: &[&VisualBlock]) -> Vec<VisualBlockSpec> {
    blocks
        .iter()
        .enumerate()
        .map(|(index, block)| {
            let base_layout = block.visual_block_spec(None).layout;
            let layout = default_layout_for(index, base_layout);
            block.visual_block_spec(Some(VisualBlockLayout {
                x: Some(layout.x),
                y: Some(layout.y),
                w: Some(layout.w),
                h: Some(layout.h),
            }))
        })
        .collect()
}

/// `renderVisualBlockSpec(spec)` — pure data transform to SVG [`Shape`]s.
pub fn render_visual_block_spec(spec: &VisualBlockSpec) -> Vec<Shape> {
    let ResolvedLayout { x, y, w, h } = spec.layout;
    let port_y = y + h / 2.0;
    let mut shapes: Vec<Shape> = vec![
        Shape::Rect(RectShape {
            x,
            y,
            w,
            h,
            rx: Some(6.0),
            fill: spec.style.fill.clone(),
            stroke: Some(spec.style.stroke.clone()),
            stroke_width: Some(2.0),
            title: Some(format!("{}: {}", spec.kind, spec.id)),
            visual_block_id: Some(spec.id.clone()),
            opacity: None,
            label: None,
        }),
        Shape::Text(TextShape {
            x: x + w / 2.0,
            y: y + 23.0,
            text: spec.label.clone(),
            font_size: Some(13.0),
            anchor: Some(Anchor::Middle),
            font_weight: Some(FontWeight::Bold),
            fill: Some(spec.style.text.clone()),
            visual_block_id: Some(spec.id.clone()),
            font_family: None,
        }),
        Shape::Text(TextShape {
            x: x + w / 2.0,
            y: y + 43.0,
            text: if !spec.contains.is_empty() {
                format!("{} ({})", spec.kind, spec.contains.len())
            } else {
                spec.kind.clone()
            },
            font_size: Some(11.0),
            anchor: Some(Anchor::Middle),
            fill: Some("#475569".to_string()),
            visual_block_id: Some(spec.id.clone()),
            font_weight: None,
            font_family: None,
        }),
    ];

    let n_in = spec.ports.inputs.len();
    for (i, port) in spec.ports.inputs.iter().enumerate() {
        shapes.push(Shape::Circle(CircleShape {
            x,
            y: port_y - (n_in as f64 - 1.0) * 5.0 + i as f64 * 10.0,
            r: 4.0,
            fill: "#ffffff".to_string(),
            stroke: Some(spec.style.stroke.clone()),
            stroke_width: Some(1.5),
            title: Some(format!("{}: {}", port.id, port.kind)),
            visual_block_id: Some(spec.id.clone()),
            opacity: None,
            label: None,
        }));
    }
    let n_out = spec.ports.outputs.len();
    for (i, port) in spec.ports.outputs.iter().enumerate() {
        shapes.push(Shape::Circle(CircleShape {
            x: x + w,
            y: port_y - (n_out as f64 - 1.0) * 5.0 + i as f64 * 10.0,
            r: 4.0,
            fill: spec.style.stroke.clone(),
            stroke: Some("#ffffff".to_string()),
            stroke_width: Some(1.5),
            title: Some(format!("{}: {}", port.id, port.kind)),
            visual_block_id: Some(spec.id.clone()),
            opacity: None,
            label: None,
        }));
    }

    shapes
}

fn default_layout_for(index: usize, layout: ResolvedLayout) -> ResolvedLayout {
    let explicit = layout.x != DEFAULT_LAYOUT.x || layout.y != DEFAULT_LAYOUT.y;
    if explicit {
        return layout;
    }
    ResolvedLayout {
        x: DEFAULT_LAYOUT.x,
        y: DEFAULT_LAYOUT.y + index as f64 * (layout.h + 12.0),
        ..layout
    }
}

fn normalize_ports(
    ports: &[VisualPortInput],
    direction: VisualPortDirection,
) -> Vec<VisualBlockPort> {
    let normalized: Vec<VisualBlockPort> =
        ports.iter().map(|p| normalize_port(p, direction)).collect();
    let mut seen = std::collections::HashSet::new();
    for port in &normalized {
        if !seen.insert(port.id.clone()) {
            panic!(
                "VisualBlock: duplicate {} port \"{}\"",
                direction.as_str(),
                port.id
            );
        }
    }
    normalized
}

fn normalize_port(port: &VisualPortInput, direction: VisualPortDirection) -> VisualBlockPort {
    let raw: VisualPortOptions = match port {
        VisualPortInput::Name(s) => VisualPortOptions {
            id: s.clone(),
            ..Default::default()
        },
        VisualPortInput::Opts(o) => o.clone(),
    };
    if raw.id.trim().is_empty() {
        panic!("VisualBlock ports require non-empty ids");
    }
    // PORT NOTE: capacity is `i64`, so the TS `Number.isInteger` check is implicit;
    // only the non-negativity guard remains.
    if let Some(cap) = raw.capacity {
        if cap < 0 {
            panic!(
                "VisualBlock port \"{}\" capacity must be a non-negative integer",
                raw.id
            );
        }
    }
    let kind = raw.kind.clone().unwrap_or_else(|| "token".to_string());
    VisualBlockPort {
        id: raw.id.clone(),
        direction,
        kind,
        label: raw.label.clone().unwrap_or_else(|| raw.id.clone()),
        data_type: raw.data_type.clone(),
        required: raw.required.unwrap_or(false),
        capacity: raw.capacity,
        metadata: raw.metadata.clone(),
    }
}

fn assert_unique_port(ports: &[VisualBlockPort], id: &str, block_id: &str, direction_label: &str) {
    if ports.iter().any(|p| p.id == id) {
        panic!("VisualBlock({block_id}): duplicate {direction_label} port \"{id}\"");
    }
}

fn clone_ports(ports: &[VisualBlockPort]) -> Vec<VisualBlockPort> {
    ports.to_vec()
}

fn clone_connection(connection: &VisualBlockConnectionSpec) -> VisualBlockConnectionSpec {
    connection.clone()
}

fn member_id(member: &VisualBlockMember) -> Option<String> {
    member.id.clone()
}

fn member_kind(member: &VisualBlockMember) -> String {
    member.kind.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_sink_role_constraints() {
        let src = VisualBlock::source("src", vec!["out".into()], VisualBlockOptions::default());
        assert_eq!(src.visual_output_ports().len(), 1);
        assert!(src.visual_input_ports().is_empty());

        let sink = VisualBlock::sink("snk", vec!["in".into()], VisualBlockOptions::default());
        assert_eq!(sink.visual_input_ports().len(), 1);
    }

    #[test]
    #[should_panic(expected = "source blocks cannot have input ports")]
    fn source_rejects_input_port() {
        let mut src = VisualBlock::source("src", vec!["out".into()], VisualBlockOptions::default());
        src.add_input_port("bad".into());
    }

    #[test]
    fn connect_builds_spec_and_records_both_sides() {
        let mut a = VisualBlock::source("a", vec!["out".into()], VisualBlockOptions::default());
        let mut b = VisualBlock::sink("b", vec!["in".into()], VisualBlockOptions::default());
        let conn = a.connect_to(&mut b, VisualBlockConnectionOptions::default());
        assert_eq!(conn.from.block_id, "a");
        assert_eq!(conn.to.block_id, "b");
        assert_eq!(a.visual_connections_outgoing().len(), 1);
        assert_eq!(b.visual_connections_incoming().len(), 1);
    }

    #[test]
    fn render_emits_box_labels_and_ports() {
        let block = VisualBlock::source(
            "gen",
            vec!["a".into(), "b".into()],
            VisualBlockOptions::default(),
        );
        let shapes = block.render_visual_block(VisualBlockRenderContext::default());
        // rect + 2 text + 2 output port circles.
        assert_eq!(shapes.len(), 5);
        assert!(matches!(shapes[0], Shape::Rect(_)));
    }

    #[test]
    fn duplicate_ports_panic() {
        let result = std::panic::catch_unwind(|| {
            VisualBlock::new(
                "x",
                VisualBlockOptions {
                    ports: Some(VisualBlockPortSpec {
                        inputs: vec!["p".into(), "p".into()],
                        outputs: Vec::new(),
                    }),
                    ..Default::default()
                },
            )
        });
        assert!(result.is_err());
    }
}
