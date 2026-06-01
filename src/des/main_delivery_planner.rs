//! Writes `out/delivery-planner.html` (interactive delivery scheduler UI).

pub fn run() {
    crate::des::delivery_planner::write_delivery_planner_artifacts();
}
