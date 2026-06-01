//! Delivery route planner: pasted address manifest, time-windowed TSP/IP-MIP
//! solve, copyable itinerary, and route animation player.

pub mod model;
pub mod solve;
pub mod ui;

pub use model::{
    default_delivery_manifest, default_delivery_request, normalize_delivery_request,
    parse_manifest, DeliveryLockedPosition, DeliveryObjectiveMode, DeliveryPlannerRequest,
    DeliveryRouteRules, DeliveryStopInput,
};
pub use solve::{solve_delivery_planner, solve_delivery_planner_summary, DeliveryPlannerResponse};
pub use ui::{
    delivery_planner_page_html, delivery_response_to_json, write_delivery_planner_artifacts,
};
