//! Input model and paste parser for the delivery time-window planner.

use serde::{Deserialize, Serialize};

pub const DEFAULT_SPEED_MPH: f64 = 28.0;
pub const DEFAULT_SERVICE_MINUTES: f64 = 6.0;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeliveryStopInput {
    pub id: String,
    pub label: String,
    pub address: String,
    pub lat: f64,
    pub lon: f64,
    pub window_start: u32,
    pub window_end: u32,
    pub service_minutes: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum DeliveryObjectiveMode {
    Distance,
    TravelTime,
    WindowCenter,
}

impl Default for DeliveryObjectiveMode {
    fn default() -> Self {
        Self::Distance
    }
}

impl DeliveryObjectiveMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Distance => "distance",
            Self::TravelTime => "travelTime",
            Self::WindowCenter => "windowCenter",
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeliveryRouteRules {
    #[serde(default)]
    pub locked_order: bool,
    #[serde(default)]
    pub ordered_stop_ids: Vec<String>,
    #[serde(default)]
    pub pinned_positions: Vec<DeliveryPinnedStop>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeliveryPinnedStop {
    pub stop_id: String,
    pub position: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeliveryPlannerRequest {
    pub user_id: String,
    pub depot_label: String,
    pub depot_address: String,
    pub depot_lat: f64,
    pub depot_lon: f64,
    pub depart_time: u32,
    pub average_speed_mph: f64,
    pub default_service_minutes: f64,
    pub solver_time_limit_ms: f64,
    pub solver_max_nodes: usize,
    pub solver_max_ticks: usize,
    pub solver_lp_max_iters: usize,
    #[serde(default)]
    pub objective_mode: DeliveryObjectiveMode,
    #[serde(default)]
    pub route_rules: DeliveryRouteRules,
    pub stops: Vec<DeliveryStopInput>,
    pub raw_manifest: String,
}

#[derive(Clone, Debug)]
pub struct ParsedManifest {
    pub depot_label: String,
    pub depot_address: String,
    pub depot_lat: f64,
    pub depot_lon: f64,
    pub depart_time: u32,
    pub stops: Vec<DeliveryStopInput>,
}

#[derive(Clone, Debug)]
pub struct ManifestParseError {
    pub line: usize,
    pub message: String,
}

impl ManifestParseError {
    fn new(line: usize, message: impl Into<String>) -> Self {
        ManifestParseError {
            line,
            message: message.into(),
        }
    }
}

pub fn default_delivery_manifest() -> &'static str {
    "Depot | 1100 Congress Ave, Austin, TX | 30.2747,-97.7404 | 08:00\n\
     Customer A | 200 E 6th St, Austin, TX | 08:30-09:30 | 30.2676,-97.7410 | service=6\n\
     Customer B | 501 W 3rd St, Austin, TX | 09:00-10:15 | 30.2672,-97.7483 | service=8\n\
     Customer C | 1600 S Congress Ave, Austin, TX | 10:00-11:20 | 30.2475,-97.7500 | service=5\n\
     Customer D | 4200 N Lamar Blvd, Austin, TX | 11:00-12:30 | 30.3136,-97.7406 | service=7\n\
     Customer E | 1900 Aldrich St, Austin, TX | 12:00-13:30 | 30.2979,-97.7042 | service=6"
}

pub fn default_delivery_request() -> DeliveryPlannerRequest {
    let raw_manifest = default_delivery_manifest().to_string();
    let parsed = parse_manifest(&raw_manifest, DEFAULT_SERVICE_MINUTES)
        .expect("default delivery manifest should parse");
    DeliveryPlannerRequest {
        user_id: "dispatcher@local".to_string(),
        depot_label: parsed.depot_label,
        depot_address: parsed.depot_address,
        depot_lat: parsed.depot_lat,
        depot_lon: parsed.depot_lon,
        depart_time: parsed.depart_time,
        average_speed_mph: DEFAULT_SPEED_MPH,
        default_service_minutes: DEFAULT_SERVICE_MINUTES,
        solver_time_limit_ms: 10_000.0,
        solver_max_nodes: 30_000,
        solver_max_ticks: 120_000,
        solver_lp_max_iters: 8_000,
        objective_mode: DeliveryObjectiveMode::Distance,
        route_rules: DeliveryRouteRules::default(),
        stops: parsed.stops,
        raw_manifest,
    }
}

pub fn normalize_delivery_request(req: &mut DeliveryPlannerRequest) {
    req.average_speed_mph = finite_or(req.average_speed_mph, DEFAULT_SPEED_MPH).max(1.0);
    req.default_service_minutes =
        finite_or(req.default_service_minutes, DEFAULT_SERVICE_MINUTES).max(0.0);
    req.solver_time_limit_ms = finite_or(req.solver_time_limit_ms, 10_000.0).max(100.0);
    req.solver_max_nodes = req.solver_max_nodes.max(1);
    req.solver_max_ticks = req.solver_max_ticks.max(1);
    req.solver_lp_max_iters = req.solver_lp_max_iters.max(1);
    req.route_rules.ordered_stop_ids = dedupe_ordered_stop_ids(&req.route_rules.ordered_stop_ids);
    normalize_pinned_positions(&mut req.route_rules.pinned_positions);
    if req.user_id.trim().is_empty() {
        req.user_id = "dispatcher@local".to_string();
    }
    if req.depot_label.trim().is_empty() {
        req.depot_label = "Depot".to_string();
    }
    for (i, stop) in req.stops.iter_mut().enumerate() {
        if stop.id.trim().is_empty() {
            stop.id = format!("S{}", i + 1);
        }
        if stop.label.trim().is_empty() {
            stop.label = format!("Stop {}", i + 1);
        }
        stop.service_minutes =
            finite_or(stop.service_minutes, req.default_service_minutes).max(0.0);
        if stop.window_end < stop.window_start {
            std::mem::swap(&mut stop.window_start, &mut stop.window_end);
        }
    }
}

fn dedupe_ordered_stop_ids(ids: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for id in ids {
        let trimmed = id.trim();
        if !trimmed.is_empty() && !out.iter().any(|seen: &String| seen == trimmed) {
            out.push(trimmed.to_string());
        }
    }
    out
}

fn normalize_pinned_positions(pins: &mut Vec<DeliveryPinnedStop>) {
    pins.retain_mut(|pin| {
        pin.stop_id = pin.stop_id.trim().to_string();
        !pin.stop_id.is_empty() && pin.position > 0
    });
}

fn finite_or(x: f64, fallback: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        fallback
    }
}

pub fn parse_manifest(
    text: &str,
    default_service_minutes: f64,
) -> Result<ParsedManifest, ManifestParseError> {
    let mut depot: Option<(String, String, f64, f64, u32)> = None;
    let mut stops: Vec<DeliveryStopInput> = Vec::new();
    for (zero_idx, raw_line) in text.lines().enumerate() {
        let line_no = zero_idx + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts = split_manifest_line(line);
        if parts.len() < 3 {
            return Err(ManifestParseError::new(
                line_no,
                "expected label, address, coordinates, and a time or time window",
            ));
        }
        let label = parts[0].trim().to_string();
        let is_depot =
            label.eq_ignore_ascii_case("depot") || label.to_ascii_lowercase().starts_with("depot ");
        let coords = find_coords(&parts);
        if is_depot {
            let (lat, lon) = coords.unwrap_or((30.2747, -97.7404));
            let depart = parts
                .iter()
                .filter_map(|p| parse_single_time(p))
                .next()
                .unwrap_or_else(|| parse_time_minutes("08:00").unwrap());
            depot = Some((label, infer_address(&parts), lat, lon, depart));
            continue;
        }
        let (base_lat, base_lon) = depot
            .as_ref()
            .map(|(_, _, lat, lon, _)| (*lat, *lon))
            .unwrap_or((30.2747, -97.7404));
        let (lat, lon) =
            coords.unwrap_or_else(|| synthetic_coordinate(line, stops.len(), base_lat, base_lon));
        let (window_start, window_end) = parts
            .iter()
            .filter_map(|p| parse_time_window(p))
            .next()
            .ok_or_else(|| {
            ManifestParseError::new(line_no, "missing drop-off window like 09:00-10:30")
        })?;
        let service_minutes = parts
            .iter()
            .filter_map(|p| parse_service_minutes(p))
            .next()
            .unwrap_or(default_service_minutes);
        stops.push(DeliveryStopInput {
            id: format!("S{}", stops.len() + 1),
            label,
            address: infer_address(&parts),
            lat,
            lon,
            window_start,
            window_end,
            service_minutes,
        });
    }
    let (depot_label, depot_address, depot_lat, depot_lon, depart_time) =
        depot.ok_or_else(|| ManifestParseError::new(0, "manifest needs a Depot line"))?;
    if stops.is_empty() {
        return Err(ManifestParseError::new(
            0,
            "manifest needs at least one customer stop",
        ));
    }
    Ok(ParsedManifest {
        depot_label,
        depot_address,
        depot_lat,
        depot_lon,
        depart_time,
        stops,
    })
}

fn split_manifest_line(line: &str) -> Vec<String> {
    let delimiter = if line.contains('|') {
        '|'
    } else if line.matches(';').count() >= 2 {
        ';'
    } else {
        ','
    };
    line.split(delimiter)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

fn infer_address(parts: &[String]) -> String {
    parts
        .iter()
        .skip(1)
        .filter(|p| find_coord_pair(p).is_none())
        .filter(|p| parse_time_window(p).is_none())
        .filter(|p| parse_single_time(p).is_none())
        .filter(|p| parse_service_minutes(p).is_none())
        .cloned()
        .collect::<Vec<_>>()
        .join(", ")
}

fn find_coords(parts: &[String]) -> Option<(f64, f64)> {
    parts.iter().find_map(|p| find_coord_pair(p))
}

fn find_coord_pair(text: &str) -> Option<(f64, f64)> {
    let cleaned = text
        .replace('(', " ")
        .replace(')', " ")
        .replace('[', " ")
        .replace(']', " ");
    let pieces: Vec<&str> = cleaned
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .collect();
    for pair in pieces.windows(2) {
        let Ok(a) = pair[0].parse::<f64>() else {
            continue;
        };
        let Ok(b) = pair[1].parse::<f64>() else {
            continue;
        };
        if (-90.0..=90.0).contains(&a) && (-180.0..=180.0).contains(&b) {
            return Some((a, b));
        }
    }
    None
}

fn synthetic_coordinate(seed: &str, idx: usize, base_lat: f64, base_lon: f64) -> (f64, f64) {
    let mut hash: u32 = 2_166_136_261;
    for b in seed.bytes().chain((idx as u64).to_le_bytes()) {
        hash ^= b as u32;
        hash = hash.wrapping_mul(16_777_619);
    }
    let angle = (hash % 360) as f64 * std::f64::consts::PI / 180.0;
    let radius = 0.018 + (((hash >> 8) % 1000) as f64 / 1000.0) * 0.045;
    (
        base_lat + angle.sin() * radius,
        base_lon + angle.cos() * radius,
    )
}

pub fn parse_time_window(text: &str) -> Option<(u32, u32)> {
    let normalized = text
        .replace('–', "-")
        .replace('—', "-")
        .replace(" to ", "-")
        .replace(" TO ", "-");
    let (a, b) = normalized.split_once('-')?;
    let start = parse_time_minutes(a.trim())?;
    let end = parse_time_minutes(b.trim())?;
    Some((start, end))
}

pub fn parse_time_minutes(text: &str) -> Option<u32> {
    let mut s = text.trim().to_ascii_lowercase();
    if s.is_empty() {
        return None;
    }
    let pm = s.ends_with("pm");
    let am = s.ends_with("am");
    if pm || am {
        s.truncate(s.len().saturating_sub(2));
    }
    s = s.trim().to_string();
    let (h_raw, m_raw) = match s.split_once(':') {
        Some((h, m)) => (h, m),
        None => (s.as_str(), "0"),
    };
    let mut hour = h_raw.trim().parse::<u32>().ok()?;
    let minute = m_raw.trim().parse::<u32>().ok()?;
    if minute >= 60 {
        return None;
    }
    if pm && hour < 12 {
        hour += 12;
    }
    if am && hour == 12 {
        hour = 0;
    }
    if hour >= 24 {
        return None;
    }
    Some(hour * 60 + minute)
}

fn parse_single_time(text: &str) -> Option<u32> {
    if text.contains('-') || text.contains(" to ") {
        return None;
    }
    parse_time_minutes(text)
}

fn parse_service_minutes(text: &str) -> Option<f64> {
    let lower = text.trim().to_ascii_lowercase();
    let value = lower
        .strip_prefix("service=")
        .or_else(|| lower.strip_prefix("service "))
        .or_else(|| lower.strip_prefix("svc="))
        .or_else(|| lower.strip_prefix("svc "))?;
    value
        .trim()
        .trim_end_matches("min")
        .trim()
        .parse::<f64>()
        .ok()
}

pub fn format_minutes(minutes: u32) -> String {
    let h = minutes / 60;
    let m = minutes % 60;
    format!("{h:02}:{m:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_default_manifest() {
        let parsed = parse_manifest(default_delivery_manifest(), DEFAULT_SERVICE_MINUTES).unwrap();
        assert_eq!(parsed.stops.len(), 5);
        assert_eq!(parsed.depart_time, 8 * 60);
        assert_eq!(parsed.stops[0].window_start, 8 * 60 + 30);
        assert_eq!(parsed.stops[1].service_minutes, 8.0);
    }

    #[test]
    fn parses_am_pm_times() {
        assert_eq!(parse_time_minutes("8:15am"), Some(495));
        assert_eq!(parse_time_minutes("12:05 PM"), Some(725));
        assert_eq!(parse_time_window("1pm-2:30pm"), Some((780, 870)));
    }

    #[test]
    fn plain_addresses_get_deterministic_coordinates() {
        let manifest = "Depot | 1100 Congress Ave, Austin, TX | 08:00\n\
                        A | 200 E 6th St, Austin, TX | 09:00-10:00\n\
                        B | 501 W 3rd St, Austin, TX | 10:00-11:00";
        let parsed = parse_manifest(manifest, DEFAULT_SERVICE_MINUTES).unwrap();
        assert_eq!(parsed.stops.len(), 2);
        assert!(parsed.stops[0].lat.is_finite());
        assert_ne!(parsed.stops[0].lat, parsed.stops[1].lat);
    }
}
