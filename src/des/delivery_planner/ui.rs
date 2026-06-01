//! Self-contained delivery planner HTML export and JSON response helpers.

use serde_json::Value;

use super::model::default_delivery_request;
use super::solve::DeliveryPlannerResponse;
use crate::des::animation::types::Animation;

pub fn delivery_planner_page_html() -> String {
    let default_json = serde_json::to_string_pretty(&default_delivery_request())
        .unwrap_or_else(|_| "{}".to_string());
    let default_escaped = default_json.replace("</script", "<\\/script");
    include_str!("delivery_planner_ui.html")
        .replace("__DEFAULT_DELIVERY_CONFIG__", &default_escaped)
}

pub fn delivery_response_to_json(resp: &DeliveryPlannerResponse) -> Value {
    let animation = animation_to_serde(&resp.route_animation);
    serde_json::json!({
        "ok": resp.ok,
        "error": resp.error,
        "solverStatus": resp.solver_status,
        "solverKind": resp.solver_kind,
        "usedFallback": resp.used_fallback,
        "fallbackReason": resp.fallback_reason,
        "inHouseOnly": resp.in_house_only,
        "usesExternalSolvers": resp.uses_external_solvers,
        "elapsedMs": resp.elapsed_ms,
        "nodesExplored": resp.nodes_explored,
        "lpSolves": resp.lp_solves,
        "numVariables": resp.num_variables,
        "numConstraints": resp.num_constraints,
        "objectiveMode": resp.objective_mode.as_str(),
        "objectiveValue": resp.objective_value,
        "objectiveDistance": resp.objective_distance,
        "windowEdgePenalty": resp.window_edge_penalty,
        "windowCenterPenalty": resp.window_center_penalty,
        "totalDistance": resp.total_distance,
        "totalTravelMinutes": resp.total_travel_minutes,
        "totalWaitMinutes": resp.total_wait_minutes,
        "route": resp.route,
        "visits": resp.visits,
        "legs": resp.legs,
        "itineraryText": resp.itinerary_text,
        "solverNotes": resp.solver_notes,
        "solverTrace": resp.solver_trace,
        "routeAnimation": animation,
    })
}

fn animation_to_serde(anim: &Animation) -> Value {
    let s = anim.to_json().to_string();
    serde_json::from_str(&s).unwrap_or(Value::Null)
}

pub fn write_delivery_planner_artifacts() {
    let ui_path = std::path::Path::new("out/delivery-planner.html");
    let _ = std::fs::create_dir_all("out");
    let _ = std::fs::write(ui_path, delivery_planner_page_html());
    println!("# Delivery planner UI: {}", ui_path.display());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_contains_delivery_controls() {
        let html = delivery_planner_page_html();
        assert!(html.contains("Delivery Scheduler"));
        assert!(html.contains("copyItinerary"));
        assert!(html.contains("routeFrame"));
        assert!(html.contains("routeTable"));
        assert!(html.contains("openGoogleMaps"));
        assert!(html.contains("openWazeNext"));
        assert!(html.contains("osmFrame"));
        assert!(html.contains("objectiveMode"));
        assert!(html.contains("loadManifestBtn"));
        assert!(html.contains("stopList"));
        assert!(html.contains("stop-position-lock"));
        assert!(html.contains("lockedPositions"));
        assert!(html.contains("windowEdgePenalty"));
    }
}
