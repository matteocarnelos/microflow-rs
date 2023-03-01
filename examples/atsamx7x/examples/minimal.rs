#![no_std]
#![no_main]

use panic_probe as _;

#[rtic::app(device = atsamx7x_hal::pac, peripherals = true, dispatchers = [I2SC0])]
mod app {
    use atsamx7x_hal as hal;
    use libm::sinf;
    use microflow::model;
    use nalgebra::vector;
    use rtt_target::{rprintln, rtt_init_print};

    #[model("../models/sine.tflite")]
    struct Model;

    #[shared]
    struct Shared {}

    #[local]
    struct Local {}

    #[init]
    fn init(cx: init::Context) -> (Shared, Local, init::Monotonics) {
        hal::watchdog::Watchdog::new(cx.device.WDT).disable();
        rtt_init_print!();

        let x = 1.3;
        let predicted = Model::evaluate(vector![x])[0];
        let exact = sinf(x);
        rprintln!("Predicted sin({}): {}", x, predicted);
        rprintln!("Exact sin({}): {}", x, exact);
        rprintln!("Error: {}", exact - predicted);

        (Shared {}, Local {}, init::Monotonics())
    }
}
