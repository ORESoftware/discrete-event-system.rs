//! Port of `src/des/animation/scenes/elevator-scene.ts`.
//!
//! Builds frames + a system-occupancy chart for the elevator-dispatch
//! animation: floors as horizontal lanes, elevators as tall rectangles whose
//! vertical position tracks the (continuous) current floor, with per-floor
//! up/down queues, a metrics panel, per-elevator status, and a legend.
//!
//! ## Conversion notes
//!
//! * `dirColor(dir, state)` — both string-literal unions become enums
//!   ([`Direction`], [`ElevatorState`]); the body is a `match`.
//! * PORT NOTE: only the subset of `crate::des::main_elevator::Building` the
//!   scene reads is mirrored locally below.

#![allow(dead_code)]

use crate::des::animation::types::{
    js_num, to_fixed, Anchor, ChartSeries, ChartSpec, CircleShape, FontWeight, FrameParts,
    LineShape, RectShape, Shape, TextShape,
};

pub const STAGE_W: f64 = 1000.0;
pub const STAGE_H: f64 = 640.0;

const VIEW_X: f64 = 60.0;
const VIEW_Y: f64 = 40.0;
const VIEW_W: f64 = 660.0;
const VIEW_H: f64 = 460.0;
const METRIC_X: f64 = 740.0;
const METRIC_Y: f64 = 40.0;
const METRIC_W: f64 = 220.0;
const METRIC_H: f64 = 460.0;

const COLOR_IDLE: &str = "#9ca3af";
const COLOR_UP: &str = "#16a34a";
const COLOR_DOWN: &str = "#2563eb";
const COLOR_SERVE: &str = "#f59e0b";

/// `'idle' | 'up' | 'down'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Idle,
    Up,
    Down,
}

impl Direction {
    pub fn label(self) -> &'static str {
        match self {
            Direction::Idle => "idle",
            Direction::Up => "up",
            Direction::Down => "down",
        }
    }
}

/// `'IDLE' | 'MOVING' | 'SERVING'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElevatorState {
    Idle,
    Moving,
    Serving,
}

impl ElevatorState {
    pub fn label(self) -> &'static str {
        match self {
            ElevatorState::Idle => "IDLE",
            ElevatorState::Moving => "MOVING",
            ElevatorState::Serving => "SERVING",
        }
    }
}

// PORT NOTE: local mirror of the elevator model (subset used by the scene).
#[derive(Clone, Debug)]
pub struct QueuePerson {
    pub to_floor: f64,
}

#[derive(Clone, Debug)]
pub struct Passenger {
    pub to_floor: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Floor {
    pub up_queue: Vec<QueuePerson>,
    pub down_queue: Vec<QueuePerson>,
}

#[derive(Clone, Debug)]
pub struct Elevator {
    pub current_floor: f64,
    pub passengers: Vec<Passenger>,
    pub capacity: f64,
    pub direction: Direction,
    pub state: ElevatorState,
    pub target_floor: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct CollectedPassenger {
    pub exit_time: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Sink {
    pub collected: Vec<CollectedPassenger>,
}

#[derive(Clone, Debug, Default)]
pub struct BuildingConfig {
    pub n_floors: usize,
    pub dispatch_mode: Option<String>,
    pub capacity: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Building {
    pub config: BuildingConfig,
    pub elevators: Vec<Elevator>,
    pub floors: Vec<Floor>,
    pub sink: Sink,
}

fn dir_color(dir: Direction, state: ElevatorState) -> &'static str {
    if state == ElevatorState::Serving {
        return COLOR_SERVE;
    }
    if dir == Direction::Up {
        return COLOR_UP;
    }
    if dir == Direction::Down {
        return COLOR_DOWN;
    }
    COLOR_IDLE
}

pub fn build_elevator_frame(t: f64, tick: f64, b: &Building) -> FrameParts {
    let mut shapes: Vec<Shape> = Vec::new();
    let cfg = &b.config;
    let n_f = cfg.n_floors;
    let n_e = b.elevators.len();

    // Frame border.
    shapes.push(Shape::Rect(RectShape {
        x: VIEW_X,
        y: VIEW_Y,
        w: VIEW_W,
        h: VIEW_H,
        fill: "#fff".to_string(),
        stroke: Some("#bbb".to_string()),
        stroke_width: Some(1.0),
        rx: Some(4.0),
        ..Default::default()
    }));

    // Floor lanes. Floor 1 at the bottom, floor nF at the top.
    let lane_h = VIEW_H / n_f as f64;
    for f in 1..=n_f {
        let y = VIEW_Y + (n_f - f) as f64 * lane_h;
        shapes.push(Shape::Line(LineShape {
            x1: VIEW_X,
            y1: y,
            x2: VIEW_X + VIEW_W,
            y2: y,
            stroke: "#e5e7eb".to_string(),
            stroke_width: Some(1.0),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: VIEW_X + 8.0,
            y: y + lane_h / 2.0 + 4.0,
            text: format!("F{f}"),
            font_size: Some(12.0),
            fill: Some("#666".to_string()),
            anchor: Some(Anchor::Start),
            ..Default::default()
        }));
    }
    // Top frame line.
    shapes.push(Shape::Line(LineShape {
        x1: VIEW_X,
        y1: VIEW_Y,
        x2: VIEW_X + VIEW_W,
        y2: VIEW_Y,
        stroke: "#e5e7eb".to_string(),
        stroke_width: Some(1.0),
        ..Default::default()
    }));

    // Floor up/down queues.
    let queue_x = VIEW_X + 50.0;
    let queue_dot = 5.0;
    for f in 1..=n_f {
        let y = VIEW_Y + (n_f - f) as f64 * lane_h + lane_h / 2.0;
        let floor = &b.floors[f - 1];
        let mut dx = 0.0;
        for person in &floor.up_queue {
            shapes.push(Shape::Circle(CircleShape {
                x: queue_x + dx,
                y: y - 6.0,
                r: queue_dot,
                fill: COLOR_UP.to_string(),
                stroke: Some("#0a3d22".to_string()),
                stroke_width: Some(0.5),
                title: Some(format!("up-bound to F{}", js_num(person.to_floor))),
                ..Default::default()
            }));
            dx += queue_dot * 2.0 + 1.0;
            if dx > 160.0 {
                break;
            }
        }
        let mut dx = 0.0;
        for person in &floor.down_queue {
            shapes.push(Shape::Circle(CircleShape {
                x: queue_x + dx,
                y: y + 6.0,
                r: queue_dot,
                fill: COLOR_DOWN.to_string(),
                stroke: Some("#162e60".to_string()),
                stroke_width: Some(0.5),
                title: Some(format!("down-bound to F{}", js_num(person.to_floor))),
                ..Default::default()
            }));
            dx += queue_dot * 2.0 + 1.0;
            if dx > 160.0 {
                break;
            }
        }
    }

    // Elevators.
    let car_x0 = VIEW_X + 250.0;
    let car_slot_w = (VIEW_W - 250.0 - 16.0) / n_e as f64;
    let car_w = 60.0_f64.min(car_slot_w * 0.8);
    let car_h = lane_h * 0.85;
    for k in 0..n_e {
        let e = &b.elevators[k];
        let y_center = VIEW_Y + VIEW_H - (e.current_floor - 0.5) * lane_h;
        let x = car_x0 + k as f64 * car_slot_w + (car_slot_w - car_w) / 2.0;
        let y = y_center - car_h / 2.0;
        let fill = dir_color(e.direction, e.state);
        let target_str = match e.target_floor {
            Some(tf) => format!(" target=F{}", js_num(tf)),
            None => String::new(),
        };
        shapes.push(Shape::Rect(RectShape {
            x,
            y,
            w: car_w,
            h: car_h,
            fill: fill.to_string(),
            rx: Some(3.0),
            stroke: Some("#222".to_string()),
            stroke_width: Some(0.8),
            title: Some(format!(
                "E{}: state={} dir={} floor={} pax={}/{}{}",
                k,
                e.state.label(),
                e.direction.label(),
                to_fixed(e.current_floor, 2),
                e.passengers.len(),
                js_num(e.capacity),
                target_str
            )),
            ..Default::default()
        }));
        // Passenger count (big number in the middle).
        shapes.push(Shape::Text(TextShape {
            x: x + car_w / 2.0,
            y: y + car_h / 2.0 + 5.0,
            text: e.passengers.len().to_string(),
            font_size: Some(14.0),
            fill: Some("#fff".to_string()),
            anchor: Some(Anchor::Middle),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
        // Capacity label below.
        shapes.push(Shape::Text(TextShape {
            x: x + car_w / 2.0,
            y: y + car_h + 12.0,
            text: format!("E{k}"),
            font_size: Some(10.0),
            fill: Some("#444".to_string()),
            anchor: Some(Anchor::Middle),
            ..Default::default()
        }));
        // Target floor indicator.
        if let Some(tf) = e.target_floor {
            if e.state == ElevatorState::Moving {
                let tgt_y = VIEW_Y + VIEW_H - (tf - 0.5) * lane_h;
                shapes.push(Shape::Line(LineShape {
                    x1: x + car_w / 2.0,
                    y1: y_center,
                    x2: x + car_w / 2.0,
                    y2: tgt_y,
                    stroke: "#999".to_string(),
                    stroke_width: Some(1.0),
                    dasharray: Some("2,2".to_string()),
                    ..Default::default()
                }));
                shapes.push(Shape::Circle(CircleShape {
                    x: x + car_w / 2.0,
                    y: tgt_y,
                    r: 3.0,
                    fill: "#999".to_string(),
                    ..Default::default()
                }));
            }
        }
        // Passenger destination ticks in the car.
        let num_dests = e.passengers.len();
        for p in 0..num_dests {
            let dx = (p % 4) as f64 * 6.0 + 4.0;
            let dy = (p / 4) as f64 * 4.0 + 4.0;
            shapes.push(Shape::Circle(CircleShape {
                x: x + dx,
                y: y + dy,
                r: 1.5,
                fill: "#fff".to_string(),
                ..Default::default()
            }));
        }
    }

    // Metrics panel.
    shapes.push(Shape::Rect(RectShape {
        x: METRIC_X,
        y: METRIC_Y,
        w: METRIC_W,
        h: METRIC_H,
        fill: "#fafafa".to_string(),
        stroke: Some("#ddd".to_string()),
        stroke_width: Some(1.0),
        rx: Some(4.0),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: METRIC_X + 12.0,
        y: METRIC_Y + 22.0,
        text: format!("Tick {}   t={}", js_num(tick), to_fixed(t, 2)),
        font_size: Some(13.0),
        fill: Some("#222".to_string()),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));

    let total_waiting: usize =
        b.floors.iter().map(|f| f.up_queue.len() + f.down_queue.len()).sum();
    let total_in_car: usize = b.elevators.iter().map(|e| e.passengers.len()).sum();
    let total_served = b.sink.collected.iter().filter(|p| p.exit_time > 0.0).count();
    let lines: [(String, String); 7] = [
        ("waiting".to_string(), total_waiting.to_string()),
        ("in elevator".to_string(), total_in_car.to_string()),
        ("served".to_string(), total_served.to_string()),
        (
            "mode".to_string(),
            b.config.dispatch_mode.clone().unwrap_or_else(|| "uncoordinated".to_string()),
        ),
        ("nFloors".to_string(), b.config.n_floors.to_string()),
        ("nElevators".to_string(), b.elevators.len().to_string()),
        ("capacity".to_string(), js_num(b.config.capacity)),
    ];
    for (i, (label, value)) in lines.iter().enumerate() {
        let y = METRIC_Y + 50.0 + i as f64 * 22.0;
        shapes.push(Shape::Text(TextShape {
            x: METRIC_X + 12.0,
            y,
            text: label.clone(),
            font_size: Some(11.0),
            fill: Some("#666".to_string()),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: METRIC_X + METRIC_W - 12.0,
            y,
            text: value.clone(),
            font_size: Some(12.0),
            fill: Some("#222".to_string()),
            anchor: Some(Anchor::End),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
    }
    // Per-elevator status block.
    let base_y = METRIC_Y + 50.0 + lines.len() as f64 * 22.0 + 10.0;
    shapes.push(Shape::Text(TextShape {
        x: METRIC_X + 12.0,
        y: base_y,
        text: "Per elevator".to_string(),
        font_size: Some(12.0),
        fill: Some("#444".to_string()),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    for k in 0..n_e {
        let e = &b.elevators[k];
        let y = base_y + 18.0 + k as f64 * 30.0;
        let sw = 12.0;
        let fill = dir_color(e.direction, e.state);
        shapes.push(Shape::Rect(RectShape {
            x: METRIC_X + 12.0,
            y: y - 9.0,
            w: sw,
            h: sw,
            fill: fill.to_string(),
            rx: Some(2.0),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: METRIC_X + 32.0,
            y,
            text: format!("E{k}"),
            font_size: Some(11.0),
            fill: Some("#222".to_string()),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: METRIC_X + 60.0,
            y,
            text: format!("F{}  {}/{}", to_fixed(e.current_floor, 1), e.passengers.len(), js_num(e.capacity)),
            font_size: Some(11.0),
            fill: Some("#444".to_string()),
            ..Default::default()
        }));
        let target_arrow = match e.target_floor {
            Some(tf) => format!(" \u{2192}F{}", js_num(tf)),
            None => String::new(),
        };
        shapes.push(Shape::Text(TextShape {
            x: METRIC_X + 12.0,
            y: y + 12.0,
            text: format!("{} {}{}", e.state.label().to_lowercase(), e.direction.label(), target_arrow),
            font_size: Some(10.0),
            fill: Some("#666".to_string()),
            ..Default::default()
        }));
    }

    // Legend at bottom.
    let leg_y = VIEW_Y + VIEW_H + 28.0;
    let legend_items: [(&str, &str); 4] = [
        ("idle", COLOR_IDLE),
        ("moving up", COLOR_UP),
        ("moving down", COLOR_DOWN),
        ("serving", COLOR_SERVE),
    ];
    let mut lx = VIEW_X;
    for (label, color) in legend_items {
        shapes.push(Shape::Rect(RectShape {
            x: lx,
            y: leg_y - 10.0,
            w: 14.0,
            h: 14.0,
            fill: color.to_string(),
            rx: Some(2.0),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: lx + 20.0,
            y: leg_y,
            text: label.to_string(),
            font_size: Some(11.0),
            fill: Some("#555".to_string()),
            ..Default::default()
        }));
        lx += 110.0;
    }

    let caption = format!(
        "tick={}  t={}s  waiting={}  in-car={}  served={}",
        js_num(tick),
        to_fixed(t, 2),
        total_waiting,
        total_in_car,
        total_served
    );
    FrameParts::with_caption(shapes, caption)
}

/// A trace of system occupancy over time for [`build_elevator_chart`].
#[derive(Clone, Debug, Default)]
pub struct ElevatorSeries {
    pub t: Vec<f64>,
    pub waiting: Vec<f64>,
    pub in_car: Vec<f64>,
    pub served: Vec<f64>,
}

/// `buildElevatorChart(series, panelY = 510, panelH = 110)`.
pub fn build_elevator_chart(series: &ElevatorSeries, panel_y: f64, panel_h: f64) -> ChartSpec {
    ChartSpec {
        x: VIEW_X,
        y: panel_y,
        w: VIEW_W,
        h: panel_h,
        title: Some("System occupancy over time".to_string()),
        y_min: Some(0.0),
        series: vec![
            ChartSeries { label: "waiting".to_string(), color: "#dc2626".to_string(), t: series.t.clone(), y: series.waiting.clone() },
            ChartSeries { label: "in elevator".to_string(), color: "#2563eb".to_string(), t: series.t.clone(), y: series.in_car.clone() },
            ChartSeries { label: "served".to_string(), color: "#16a34a".to_string(), t: series.t.clone(), y: series.served.clone() },
        ],
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dir_color_priorities_serving() {
        assert_eq!(dir_color(Direction::Up, ElevatorState::Serving), COLOR_SERVE);
        assert_eq!(dir_color(Direction::Up, ElevatorState::Moving), COLOR_UP);
        assert_eq!(dir_color(Direction::Down, ElevatorState::Idle), COLOR_DOWN);
        assert_eq!(dir_color(Direction::Idle, ElevatorState::Idle), COLOR_IDLE);
    }

    #[test]
    fn frame_caption_totals() {
        let b = Building {
            config: BuildingConfig { n_floors: 3, dispatch_mode: None, capacity: 4.0 },
            elevators: vec![Elevator {
                current_floor: 1.5,
                passengers: vec![Passenger { to_floor: 3.0 }],
                capacity: 4.0,
                direction: Direction::Up,
                state: ElevatorState::Moving,
                target_floor: Some(3.0),
            }],
            floors: vec![
                Floor { up_queue: vec![QueuePerson { to_floor: 2.0 }], down_queue: vec![] },
                Floor::default(),
                Floor::default(),
            ],
            sink: Sink { collected: vec![CollectedPassenger { exit_time: 5.0 }] },
        };
        let fp = build_elevator_frame(12.0, 3.25, &b);
        assert_eq!(fp.caption.as_deref(), Some("tick=3.25  t=12.00s  waiting=1  in-car=1  served=1"));
    }
}
