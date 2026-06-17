//! `two-bathrooms` — the same household bathroom question as
//! [`crate::des::bathrooms`], but expressed through the engine's reusable
//! entity + animation **frameworks** rather than a hand-rolled event loop and a
//! bespoke HTML player.
//!
//! What is reused here that `bathrooms` does by hand:
//!   * **Moving entities** — each person is a real
//!     [`ProcessableMovingEntity`] that flows between stations, accruing
//!     `time_in_system` and station-visit counts through the framework.
//!   * **Stationary entities** — each bathroom is a [`BathroomStation`]
//!     implementing [`StationaryEntity`], with the framework's busy/idle
//!     server-time accounting driven by `do_time_step`.
//!   * **Visual blocks** — the lobby, the two bathrooms and the exit each
//!     implement [`VisualBlockRenderable`] and draw themselves every frame.
//!   * **Animation framework** — frames are assembled as [`Animation`] shapes
//!     and rendered with the shared [`build_html`] player (the same one the
//!     elevator/traffic/dc-motor scenes use), including a time-series
//!     [`ChartSpec`].
//!
//! The *physics* is identical to `bathrooms` — a finite-source, 2-server loss
//! system — so the steady-state occupancy numbers match. The aggregate
//! statistics panel is taken straight from [`run_bathroom_montecarlo`]; only the
//! animated sample window is re-simulated here so the **stations themselves**
//! make the accept/reject decision as the movables arrive.

use std::rc::Rc;

use crate::des::animation::frame_recorder::{
    render_visual_blocks, VisualBlockRenderContext, VisualBlockRenderable,
};
use crate::des::animation::html_player::build_html;
use crate::des::animation::types::{
    Anchor, Animation, ChartSeries, ChartSpec, CircleShape, FontWeight, Frame, FrameParts,
    RectShape, Shape, TextShape,
};
use crate::des::bathrooms::{run_bathroom_montecarlo, BathroomConfig, BathroomReport};
use crate::des::entity_moving::moving::{MovingEntity, ProcessableMovingEntity};
use crate::des::r#abstract::interfaces::{EntityGraphData, HasEntityValidation};
use crate::des::r#abstract::r#abstract::{Entity, EntityCore, StationaryEntity};
use crate::des::shared::capabilities::{RandomSource, SeededRandom};
use crate::des::shared::precision::{bgn, to_f64, Decimal};

const MINUTES_PER_DAY: f64 = 1440.0;

// ---- stage layout ----------------------------------------------------------
const STAGE_W: f64 = 900.0;
const STAGE_H: f64 = 470.0;
const COL_BG: &str = "#0e1116";
const COL_FREE: &str = "#3fb950";
const COL_BUSY: &str = "#f85149";
const COL_REJECT: &str = "#d29922";
const COL_INK: &str = "#e6edf3";
const COL_MUTED: &str = "#8b97a7";
const LOBBY_Y: f64 = 360.0;
const DOOR_Y: f64 = 70.0;

// =============================================================================
// Stationary entity: one bathroom (a single-server loss station).
// =============================================================================

/// A bathroom modelled as a framework [`StationaryEntity`]: a single server
/// that is either free or holds one occupant for the fixed service time. The
/// framework's `run_time_step` accrues busy/idle server time.
struct BathroomStation {
    core: EntityCore,
    /// Display index (1-based) and door centre on the stage.
    number: usize,
    cx: f64,
    /// Person id currently inside, if any.
    occupant: Option<usize>,
    /// Minutes of service remaining for the current occupant.
    remaining: f64,
    busy_time: Decimal,
    idle_time: Decimal,
    /// Total occupants served (completed visits).
    served: u64,
}

impl BathroomStation {
    fn new(number: usize, cx: f64) -> Self {
        BathroomStation {
            core: EntityCore::new(format!("bathroom-{number}")),
            number,
            cx,
            occupant: None,
            remaining: 0.0,
            busy_time: bgn(0.0),
            idle_time: bgn(0.0),
            served: 0,
        }
    }

    fn is_free(&self) -> bool {
        self.occupant.is_none()
    }

    /// Server utilisation = busy / (busy + idle).
    fn utilization(&self) -> f64 {
        let total = self.busy_time + self.idle_time;
        if to_f64(total) > 0.0 {
            to_f64(self.busy_time / total)
        } else {
            0.0
        }
    }

    /// Admit a person for `service` minutes (caller guarantees the station is
    /// free).
    fn admit(&mut self, person: usize, service: f64) {
        self.occupant = Some(person);
        self.remaining = service;
    }

    /// Advance the occupant's clock; returns the person who just finished, if
    /// any.
    fn tick_service(&mut self, step: f64) -> Option<usize> {
        if let Some(p) = self.occupant {
            self.remaining -= step;
            if self.remaining <= 1e-9 {
                self.occupant = None;
                self.remaining = 0.0;
                self.served += 1;
                return Some(p);
            }
        }
        None
    }
}

impl Entity for BathroomStation {
    fn core(&self) -> &EntityCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut EntityCore {
        &mut self.core
    }
    fn do_validation(&mut self) {}
    fn do_validation_before_run(&mut self) -> bool {
        true
    }
    fn get_graph_data(&self) -> EntityGraphData {
        EntityGraphData::new()
            .with("occupied", if self.occupant.is_some() { 1.0 } else { 0.0 })
            .with("serverUtilization", self.utilization())
            .with("served", self.served as f64)
    }
    /// Framework server-time accounting: busy while occupied, idle otherwise.
    fn run_time_step(&mut self, step_size: Decimal) {
        if self.occupant.is_some() {
            self.busy_time += step_size;
        } else {
            self.idle_time += step_size;
        }
    }
}

impl HasEntityValidation for BathroomStation {
    fn validate(&self) -> bool {
        true
    }
}

impl StationaryEntity for BathroomStation {
    fn do_setup_after_input_conn(&mut self) -> bool {
        true
    }
    fn do_setup_after_output_conn(&mut self) -> bool {
        true
    }
}

impl VisualBlockRenderable for BathroomStation {
    fn render_visual_block(&self, _ctx: &VisualBlockRenderContext) -> Vec<Shape> {
        let busy = self.occupant.is_some();
        let door_col = if busy { COL_BUSY } else { COL_FREE };
        let block_id = self.core.id.clone();
        let mut shapes = vec![
            // door frame
            Shape::Rect(RectShape {
                x: self.cx - 46.0,
                y: DOOR_Y - 44.0,
                w: 92.0,
                h: 104.0,
                fill: "#1c232d".to_string(),
                stroke: Some("#283142".to_string()),
                stroke_width: Some(1.5),
                rx: Some(10.0),
                visual_block_id: Some(block_id.clone()),
                ..Default::default()
            }),
            // door leaf (colour = state)
            Shape::Rect(RectShape {
                x: self.cx - 36.0,
                y: DOOR_Y - 36.0,
                w: 72.0,
                h: 88.0,
                fill: door_col.to_string(),
                stroke: Some(if busy { "#911" } else { "#0b3" }.to_string()),
                stroke_width: Some(2.0),
                rx: Some(6.0),
                visual_block_id: Some(block_id.clone()),
                ..Default::default()
            }),
            Shape::Text(TextShape {
                x: self.cx,
                y: DOOR_Y + 74.0,
                text: format!("BATH {}", self.number),
                font_size: Some(13.0),
                fill: Some(COL_INK.to_string()),
                anchor: Some(Anchor::Middle),
                font_weight: Some(FontWeight::Bold),
                visual_block_id: Some(block_id.clone()),
                ..Default::default()
            }),
            Shape::Text(TextShape {
                x: self.cx,
                y: DOOR_Y - 4.0,
                text: if busy {
                    format!("IN USE · #{}", self.occupant.unwrap() + 1)
                } else {
                    "VACANT".to_string()
                },
                font_size: Some(12.0),
                fill: Some(if busy { "#fff" } else { "#062" }.to_string()),
                anchor: Some(Anchor::Middle),
                font_weight: Some(FontWeight::Bold),
                visual_block_id: Some(block_id.clone()),
                ..Default::default()
            }),
        ];
        // utilisation readout under the label
        shapes.push(Shape::Text(TextShape {
            x: self.cx,
            y: DOOR_Y + 90.0,
            text: format!("util {:.0}%", 100.0 * self.utilization()),
            font_size: Some(10.0),
            fill: Some(COL_MUTED.to_string()),
            anchor: Some(Anchor::Middle),
            visual_block_id: Some(block_id),
            ..Default::default()
        }));
        shapes
    }
}

// =============================================================================
// Lightweight stationary visual blocks for the lobby and the exit.
// =============================================================================

struct LabelledZone {
    id: String,
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    label: String,
}

impl VisualBlockRenderable for LabelledZone {
    fn render_visual_block(&self, _ctx: &VisualBlockRenderContext) -> Vec<Shape> {
        vec![
            Shape::Rect(RectShape {
                x: self.x,
                y: self.y,
                w: self.w,
                h: self.h,
                fill: "#11161d".to_string(),
                stroke: Some("#283142".to_string()),
                stroke_width: Some(1.0),
                rx: Some(8.0),
                visual_block_id: Some(self.id.clone()),
                ..Default::default()
            }),
            Shape::Text(TextShape {
                x: self.x + 10.0,
                y: self.y + 18.0,
                text: self.label.clone(),
                font_size: Some(11.0),
                fill: Some(COL_MUTED.to_string()),
                anchor: Some(Anchor::Start),
                visual_block_id: Some(self.id.clone()),
                ..Default::default()
            }),
        ]
    }
}

// =============================================================================
// People as moving entities + their on-stage rendering state.
// =============================================================================

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Lobby,
    InBath(usize),
}

struct PersonView {
    x: f64,
    y: f64,
    tx: f64,
    ty: f64,
    seat_x: f64,
    seat_y: f64,
    mode: Mode,
    /// Frames remaining to flash a "turned away" bubble.
    reject_flash: f64,
}

const PALETTE: [&str; 8] = [
    "#58a6ff", "#f778ba", "#3fb950", "#d29922", "#a371f7", "#39c5cf", "#ff7b72", "#e3b341",
];

fn person_color(i: usize) -> &'static str {
    PALETTE[i % PALETTE.len()]
}

// =============================================================================
// Arrival generation (finite-source, self-non-overlapping) — same rule as
// `bathrooms`, re-derived locally for the animated window.
// =============================================================================

fn generate_arrivals(cfg: &BathroomConfig, days: usize, seed: u32) -> Vec<(f64, usize)> {
    let mut rng = SeededRandom::new(seed);
    let mut all: Vec<(f64, usize)> = Vec::new();
    for person in 0..cfg.people {
        let mut starts: Vec<f64> = Vec::new();
        for d in 0..days {
            let base = d as f64 * MINUTES_PER_DAY;
            for _ in 0..cfg.visits_per_day {
                starts.push(base + rng.next_float() * MINUTES_PER_DAY);
            }
        }
        starts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut busy_until = f64::NEG_INFINITY;
        for s in starts {
            if s >= busy_until {
                all.push((s, person));
                busy_until = s + cfg.service_min;
            }
        }
    }
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    all
}

// =============================================================================
// Build the animation by stepping the station network.
// =============================================================================

/// Simulate the animated sample window through the framework stations and emit
/// an [`Animation`]. `report` supplies the headline statistics shown alongside.
fn build_animation(cfg: &BathroomConfig) -> Animation {
    // The animated window is intentionally short (2 days) and frame-budgeted:
    // the player embeds every frame's shapes, so we record a snapshot only every
    // few minutes to keep the page small. The long-run statistics come from the
    // full Monte-Carlo, not from this window.
    let days = cfg.sample_days.min(2).max(1);
    let span = days as f64 * MINUTES_PER_DAY;
    let dt = 1.0_f64; // 1-minute integration step
    let record_every = (span / 360.0).max(1.0); // ~360 frames total
    let service = cfg.service_min;

    // Two bathroom stations, evenly placed across the top.
    let mut stations: Vec<BathroomStation> = (0..cfg.bathrooms)
        .map(|s| {
            let cx = STAGE_W * (s as f64 + 1.0) / (cfg.bathrooms as f64 + 1.0);
            BathroomStation::new(s + 1, cx)
        })
        .collect();

    // People as real moving entities + their stage views.
    let mut people: Vec<ProcessableMovingEntity> = (0..cfg.people)
        .map(|_| {
            let mut m = ProcessableMovingEntity::new();
            m.init();
            m
        })
        .collect();
    let cols = (cfg.people as f64 / 2.0).ceil().max(1.0);
    let mut views: Vec<PersonView> = (0..cfg.people)
        .map(|i| {
            let col = (i as f64) % cols;
            let row = ((i as f64) / cols).floor();
            let sx = 80.0 + col * ((STAGE_W - 160.0) / (cols - 1.0).max(1.0));
            let sy = LOBBY_Y + 28.0 + row * 56.0;
            PersonView {
                x: sx,
                y: sy,
                tx: sx,
                ty: sy,
                seat_x: sx,
                seat_y: sy,
                mode: Mode::Lobby,
                reject_flash: 0.0,
            }
        })
        .collect();

    // Static visual blocks (lobby + exit zones).
    let static_blocks: Vec<Rc<dyn VisualBlockRenderable>> = vec![
        Rc::new(LabelledZone {
            id: "lobby".to_string(),
            x: 30.0,
            y: LOBBY_Y,
            w: STAGE_W - 60.0,
            h: STAGE_H - LOBBY_Y - 20.0,
            label: "living area (sources / sink of arrivals)".to_string(),
        }),
        Rc::new(LabelledZone {
            id: "hall".to_string(),
            x: 30.0,
            y: DOOR_Y + 110.0,
            w: STAGE_W - 60.0,
            h: 2.0,
            label: String::new(),
        }),
    ];

    let mut arrivals = generate_arrivals(cfg, days, cfg.seed ^ 0x7A11_E72);
    arrivals.reverse(); // pop from the back in time order

    let mut frames: Vec<Frame> = Vec::new();
    // Occupancy time-series for the chart.
    let (mut chart_t, mut chart_occ) = (Vec::<f64>::new(), Vec::<f64>::new());

    let mut t = 0.0_f64;
    let mut tick: f64 = 0.0;
    let mut next_record = 0.0_f64;
    let step_dec = bgn(dt);

    while t < span + 1e-6 {
        // 1) advance service clocks; finished occupants walk home.
        for st in stations.iter_mut() {
            if let Some(done) = st.tick_service(dt) {
                views[done].mode = Mode::Lobby;
            }
        }
        // 2) admit arrivals scheduled in [t, t+dt).
        while let Some(&(at, person)) = arrivals.last() {
            if at >= t + dt {
                break;
            }
            arrivals.pop();
            // A person already inside cannot start a second visit.
            if matches!(views[person].mode, Mode::InBath(_)) {
                continue;
            }
            match stations.iter_mut().position(|s| s.is_free()) {
                Some(idx) => {
                    stations[idx].admit(person, service);
                    views[person].mode = Mode::InBath(idx);
                    people[person].start_new_station(&format!("bathroom-{}", idx + 1));
                    people[person].add_visited_station(&format!("bathroom-{}", idx + 1));
                }
                None => {
                    // Both busy → rejected (lost demand).
                    people[person].add_visited_station("rejected");
                    views[person].reject_flash = 6.0;
                }
            }
        }
        // 3) framework server-time accounting + occupant time-in-system.
        for st in stations.iter_mut() {
            st.do_time_step(step_dec);
        }
        for (i, v) in views.iter().enumerate() {
            if matches!(v.mode, Mode::InBath(_)) {
                people[i].bump_time_in_system(step_dec);
            }
        }

        // 4) update stage targets for each person.
        for (i, v) in views.iter_mut().enumerate() {
            match v.mode {
                Mode::InBath(idx) => {
                    v.tx = stations[idx].cx;
                    v.ty = DOOR_Y + 40.0;
                }
                Mode::Lobby => {
                    v.tx = v.seat_x;
                    v.ty = v.seat_y;
                }
            }
            // ease toward target
            v.x += (v.tx - v.x) * 0.35;
            v.y += (v.ty - v.y) * 0.35;
            if v.reject_flash > 0.0 {
                v.reject_flash -= 1.0;
            }
            let _ = i;
        }

        // 5) record a frame every `record_every` minutes.
        if t + 1e-9 >= next_record {
            next_record += record_every;
            let occ_now = stations.iter().filter(|s| !s.is_free()).count();
            chart_t.push(t);
            chart_occ.push(occ_now as f64);
            frames.push(render_frame(&stations, &views, &static_blocks, t, tick, occ_now));
            tick += 1.0;
        }
        t += dt;
    }

    let occ_chart = ChartSpec {
        x: 20.0,
        y: STAGE_H + 16.0,
        w: STAGE_W - 40.0,
        h: 120.0,
        title: Some("bathrooms occupied over the sampled window".to_string()),
        y_min: Some(0.0),
        y_max: Some(cfg.bathrooms as f64 + 0.2),
        y_label: Some("occupied".to_string()),
        series: vec![ChartSeries {
            label: "occupied".to_string(),
            color: "#58a6ff".to_string(),
            t: chart_t,
            y: chart_occ,
        }],
        cursor: Some(true),
    };

    Animation {
        width: STAGE_W,
        height: STAGE_H,
        fps: 30.0,
        title: Some("Two bathrooms — entity/visual-framework Monte-Carlo".to_string()),
        subtitle: Some(occupancy_subtitle(cfg)),
        frames,
        charts: Some(vec![occ_chart]),
        background: Some(COL_BG.to_string()),
    }
}

fn occupancy_subtitle(cfg: &BathroomConfig) -> String {
    format!(
        "{} people · {} bathrooms · {} visits/day × {} min — people are MovingEntities, \
         bathrooms are StationaryEntities, drawn as visual blocks",
        cfg.people, cfg.bathrooms, cfg.visits_per_day, cfg.service_min as i64
    )
}

/// Assemble one frame's shapes: static blocks, station blocks, people, caption.
fn render_frame(
    stations: &[BathroomStation],
    views: &[PersonView],
    static_blocks: &[Rc<dyn VisualBlockRenderable>],
    t: f64,
    tick: f64,
    occ_now: usize,
) -> Frame {
    let ctx = VisualBlockRenderContext {
        tick: Some(tick),
        time: Some(t),
        index: None,
        stage_width: Some(STAGE_W),
        stage_height: Some(STAGE_H),
    };
    let mut shapes: Vec<Shape> = Vec::new();
    // static stationary visual blocks (lobby/hall)
    shapes.extend(render_visual_blocks(static_blocks, &ctx));
    // bathroom stationary entities render themselves
    for st in stations {
        shapes.extend(st.render_visual_block(&ctx));
    }
    // people (moving entities) as coloured tokens
    for (i, v) in views.iter().enumerate() {
        shapes.push(Shape::Circle(CircleShape {
            x: v.x,
            y: v.y,
            r: 13.0,
            fill: person_color(i).to_string(),
            stroke: Some("#0008".to_string()),
            stroke_width: Some(2.0),
            visual_block_id: Some(format!("person-{i}")),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: v.x,
            y: v.y + 4.0,
            text: (i + 1).to_string(),
            font_size: Some(12.0),
            fill: Some("#06121f".to_string()),
            anchor: Some(Anchor::Middle),
            font_weight: Some(FontWeight::Bold),
            visual_block_id: Some(format!("person-{i}")),
            ..Default::default()
        }));
        if v.reject_flash > 0.0 {
            shapes.push(Shape::Text(TextShape {
                x: v.x,
                y: v.y - 20.0,
                text: "occupied!".to_string(),
                font_size: Some(11.0),
                fill: Some(COL_REJECT.to_string()),
                anchor: Some(Anchor::Middle),
                font_weight: Some(FontWeight::Bold),
                visual_block_id: Some(format!("person-{i}")),
                ..Default::default()
            }));
        }
    }

    let day = (t / MINUTES_PER_DAY).floor() as i64 + 1;
    let mins = (t % MINUTES_PER_DAY) as i64;
    let caption = format!(
        "Day {} · {:02}:{:02} — {} / {} bathrooms occupied",
        day,
        mins / 60,
        mins % 60,
        occ_now,
        stations.len()
    );
    FrameParts::with_caption(shapes, caption).into_frame(t, tick)
}

// =============================================================================
// Public API.
// =============================================================================

/// Run the study and render the framework-based animation, with a statistics
/// banner (occupancy probabilities vs. the binomial reference) injected into the
/// shared animation player page.
pub fn render_two_bathrooms_html(cfg: &BathroomConfig) -> String {
    let report = run_bathroom_montecarlo(cfg);
    let anim = build_animation(cfg);
    let page = build_html(&anim);
    inject_stats_panel(page, &report)
}

/// Default-config convenience used by the server route.
pub fn render_default_two_bathrooms_html() -> String {
    render_two_bathrooms_html(&BathroomConfig::default())
}

/// Splice a compact results panel into the shared player page (before `</body>`).
fn inject_stats_panel(page: String, report: &BathroomReport) -> String {
    let occ = &report.occupancy_fraction;
    let a = &report.analytic;
    let cfg = &report.config;
    let pct = |x: f64| format!("{:.1}%", 100.0 * x);
    let two = occ.get(2).copied().unwrap_or(0.0);
    let any = 1.0 - occ[0];
    let cond = if any > 0.0 { two / any } else { 0.0 };
    let panel = format!(
        r#"<div style="max-width:900px;margin:18px auto;font:14px -apple-system,Segoe UI,Roboto,sans-serif;color:#e6edf3">
  <h3 style="margin:0 0 8px">Steady-state occupancy — Monte-Carlo vs. binomial</h3>
  <p style="color:#8b97a7;margin:0 0 10px">
    {reps} replications × {days} days (warm-up discarded), {people} people sharing {baths} bathrooms.
    The animation above is the framework station network; these numbers are the long-run statistics.
  </p>
  <table style="width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums">
    <thead><tr>
      <th style="text-align:left;border-bottom:1px solid #283142;padding:6px">state</th>
      <th style="text-align:right;border-bottom:1px solid #283142;padding:6px">simulated</th>
      <th style="text-align:right;border-bottom:1px solid #283142;padding:6px">binomial</th>
    </tr></thead>
    <tbody>
      <tr><td style="padding:6px">no bathroom occupied</td><td style="text-align:right;padding:6px">{s0}</td><td style="text-align:right;padding:6px">{a0}</td></tr>
      <tr><td style="padding:6px">exactly one occupied</td><td style="text-align:right;padding:6px">{s1}</td><td style="text-align:right;padding:6px">{a1}</td></tr>
      <tr><td style="padding:6px">both occupied</td><td style="text-align:right;padding:6px">{s2}</td><td style="text-align:right;padding:6px">{a2}</td></tr>
      <tr><td style="padding:6px">P(both | at least one)</td><td style="text-align:right;padding:6px">{sc}</td><td style="text-align:right;padding:6px">{ac}</td></tr>
      <tr><td style="padding:6px">requests turned away</td><td style="text-align:right;padding:6px">{blk}</td><td style="text-align:right;padding:6px">—</td></tr>
    </tbody>
  </table>
  <p style="color:#8b97a7;margin:10px 0 0">Per person, P(in a bathroom at a random instant) = {pp}. Built on
    <code>entity_moving</code> (MovingEntity), <code>StationaryEntity</code> bathrooms, and the
    <code>animation</code> visual-block player.</p>
</div>"#,
        reps = cfg.replications,
        days = cfg.days_per_rep,
        people = cfg.people,
        baths = cfg.bathrooms,
        s0 = pct(occ[0]),
        s1 = pct(occ[1]),
        s2 = pct(two),
        a0 = pct(a.p0),
        a1 = pct(a.p1),
        a2 = pct(a.p2_plus),
        sc = pct(cond),
        ac = pct(a.p_both_given_any),
        blk = pct(report.blocked_fraction),
        pp = format!("{:.2}%", 100.0 * a.p_person),
    );
    match page.rfind("</body>") {
        Some(idx) => {
            let mut out = String::with_capacity(page.len() + panel.len());
            out.push_str(&page[..idx]);
            out.push_str(&panel);
            out.push_str(&page[idx..]);
            out
        }
        None => page + &panel,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn animation_has_frames_and_chart() {
        let cfg = BathroomConfig {
            sample_days: 2,
            ..BathroomConfig::default()
        };
        let anim = build_animation(&cfg);
        assert!(anim.frames.len() > 100, "frames = {}", anim.frames.len());
        assert!(anim.charts.as_ref().map(|c| !c.is_empty()).unwrap_or(false));
        // Every frame carries the static + station + people shapes.
        assert!(anim.frames.iter().all(|f| f.shapes.len() >= 8));
    }

    #[test]
    fn page_embeds_animation_and_stats() {
        let cfg = BathroomConfig {
            replications: 40,
            days_per_rep: 20,
            sample_days: 1,
            ..BathroomConfig::default()
        };
        let html = render_two_bathrooms_html(&cfg);
        assert!(html.contains("<html"));
        assert!(html.contains("Monte-Carlo vs. binomial"));
        assert!(html.contains("both occupied"));
    }
}
