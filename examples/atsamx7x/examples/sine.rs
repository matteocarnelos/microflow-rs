#![no_std]
#![no_main]

use panic_probe as _;

#[rtic::app(device = hal::pac, peripherals = true, dispatchers = [I2SC0])]
mod app {
    use libm::sinf;
    use microflow::model;
    use nalgebra::matrix;
    use rtt_target::{rprintln, rtt_init_print};

    #[model("../../models/sine.tflite")]
    struct Sine;

    #[shared]
    struct Shared {}

    #[local]
    struct Local {}

    #[init]
    fn init(cx: init::Context) -> (Shared, Local, init::Monotonics) {
        hal::watchdog::Watchdog::new(cx.device.WDT).disable();
        rtt_init_print!();

        let x = 1.5;
        let y_predicted = Sine::predict(matrix![x])[0];
        let y_exact = sinf(x);
        rprintln!("Predicted sin({}): {}", x, y_predicted);
        rprintln!("Exact sin({}): {}", x, y_exact);
        rprintln!("Error: {}", y_exact - y_predicted);

        (Shared {}, Local {}, init::Monotonics())
    }
}
