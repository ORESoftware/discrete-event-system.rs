//! Port of `src/des/main-wind-mppt.ts`.
//!
//! Runnable MPPT demo for a PMSG wind-energy conversion system. Wires the
//! self-clocking turbine plant to an MPPT controller (optimal-torque or PI
//! speed-loop) and a trajectory sink, runs the lightweight DES loop, and prints
//! convergence to the optimal tip-speed ratio / power coefficient.
//!
//! Conversion notes:
//!   - `class WindMpptDemo` → struct + impl; top-level run → [`run`].
//!   - `process.env.CONTROLLER` → `std::env`.
//!   - delegates to `general::control_systems::wind_mppt`.

use std::cell::RefCell;
use std::rc::Rc;

use crate::des::general::control_systems::wind_mppt::{
    OptimalTorqueMpptController, SpeedPiMpptController, SpeedPiMpptOpts, WindMpptChannels,
    WindMpptSinkStation, WindProfile, WindProfileSegment, WindTurbineAeroOpts,
    WindTurbineAerodynamics, WindTurbinePlantOpts, WindTurbinePlantStation,
};
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::station::{DESStation, StationRef};

/// `Number.prototype.toExponential(digits)` (signed exponent, no leading zeros).
fn to_exponential(x: f64, digits: usize) -> String {
    let s = format!("{:.*e}", digits, x);
    let (mant, exp) = s.split_once('e').unwrap_or((s.as_str(), "0"));
    let exp_num: i32 = exp.parse().unwrap_or(0);
    let sign = if exp_num < 0 { '-' } else { '+' };
    format!("{}e{}{}", mant, sign, exp_num.abs())
}

/// Builds, runs, and reports a single wind-MPPT simulation.
struct WindMpptDemo {
    aero: WindTurbineAerodynamics,
    dt: f64,
    steps: usize,
}

impl WindMpptDemo {
    fn new() -> Self {
        WindMpptDemo {
            aero: WindTurbineAerodynamics::new(WindTurbineAeroOpts {
                air_density: None,
                blade_radius: 2.5,
                pitch_deg: Some(0.0),
            }),
            dt: 0.05,
            steps: 1200,
        }
    }

    fn run(&self, controller_kind: &str) {
        let wind_profile = WindProfile::new(&[
            WindProfileSegment {
                from_time: 0.0,
                speed: 8.0,
            },
            WindProfileSegment {
                from_time: 20.0,
                speed: 11.0,
            },
            WindProfileSegment {
                from_time: 40.0,
                speed: 9.0,
            },
        ]);

        let plant = Rc::new(RefCell::new(WindTurbinePlantStation::new(
            "turbine",
            WindTurbinePlantOpts {
                aero: self.aero.clone(),
                wind_profile,
                inertia: 6.0,
                friction: 0.02,
                dt: self.dt,
                steps: self.steps,
                initial_omega: 2.0,
            },
        )));

        let controller: StationRef = if controller_kind == "pi" {
            Rc::new(RefCell::new(SpeedPiMpptController::new(
                "mppt-pi",
                &self.aero,
                SpeedPiMpptOpts {
                    kp: 8.0,
                    ki: 4.0,
                    dt: self.dt,
                    max_torque: None,
                },
            )))
        } else {
            Rc::new(RefCell::new(OptimalTorqueMpptController::new(
                "mppt-opt-torque",
                &self.aero,
            )))
        };

        let sink = Rc::new(RefCell::new(WindMpptSinkStation::new("sink")));

        let plant_ref: StationRef = plant.clone();
        let sink_ref: StationRef = sink.clone();

        plant.borrow_mut().core_mut().pipe(
            controller.clone(),
            WindMpptChannels::STATE,
            WindMpptChannels::STATE,
        );
        plant.borrow_mut().core_mut().pipe(
            sink_ref.clone(),
            WindMpptChannels::STATE,
            WindMpptChannels::STATE,
        );
        controller.borrow_mut().core_mut().pipe(
            plant_ref.clone(),
            WindMpptChannels::TORQUE,
            WindMpptChannels::TORQUE,
        );

        let summary = run_iterative_des(
            vec![plant_ref, controller, sink_ref],
            IterativeRunOptions {
                shuffle: false,
                max_ticks: Some(self.steps + 5),
                ..Default::default()
            },
        );

        self.report(controller_kind, &sink.borrow(), summary.ticks);
    }

    fn report(&self, kind: &str, sink: &WindMpptSinkStation, ticks: usize) {
        let lambda_star = self.aero.optimal_tip_speed_ratio();
        let cp_max = self.aero.max_power_coefficient();
        println!();
        println!("============================================================");
        println!(" Wind MPPT — PMSG WECS   (controller: {})", kind);
        println!("============================================================");
        println!("  blade radius R          : {} m", self.aero.blade_radius);
        println!(
            "  swept area A            : {:.3} m²",
            self.aero.swept_area()
        );
        println!("  optimal λ*              : {:.4}", lambda_star);
        println!("  C_p,max                 : {:.4}", cp_max);
        println!(
            "  K_opt (½ρπR⁵C_p/λ*³)    : {}",
            to_exponential(self.aero.optimal_torque_gain(), 4)
        );
        println!("  ticks run               : {}  (dt={}s)", ticks, self.dt);
        println!("  ----------------------------------------------------------");
        println!("   step      V[m/s]   ω[rad/s]     λ       C_p     P[kW]");
        let n = sink.samples.len();
        let idxs = [
            0,
            (n as f64 * 0.25).floor() as usize,
            (n as f64 * 0.5).floor() as usize,
            (n as f64 * 0.75).floor() as usize,
            n - 1,
        ];
        for i in idxs {
            let s = &sink.samples[i];
            println!(
                "   {}   {}   {}   {}   {}   {}",
                format!("{:>5}", s.tick),
                format!("{:>7}", format!("{:.2}", s.wind_speed)),
                format!("{:>8}", format!("{:.3}", s.omega)),
                format!("{:>6}", format!("{:.3}", s.lambda)),
                format!("{:>6}", format!("{:.4}", s.cp)),
                format!("{:>6}", format!("{:.3}", s.mech_power / 1000.0)),
            );
        }
        println!("  ----------------------------------------------------------");
        let lambda_err = (sink.final_lambda() - lambda_star).abs();
        println!(
            "  final λ                 : {:.4}   (|λ−λ*| = {:.4})",
            sink.final_lambda(),
            lambda_err
        );
        println!(
            "  final C_p / C_p,max     : {:.2}%",
            sink.final_cp() / cp_max * 100.0
        );
        println!(
            "  final captured power    : {:.3} kW",
            sink.final_power() / 1000.0
        );
        println!("============================================================");
        println!();
    }
}

/// Entry point (`main()` in the TS source).
pub fn run() {
    let kind = if std::env::var("CONTROLLER")
        .unwrap_or_default()
        .to_lowercase()
        == "pi"
    {
        "pi"
    } else {
        "optimal-torque"
    };
    WindMpptDemo::new().run(kind);
}
