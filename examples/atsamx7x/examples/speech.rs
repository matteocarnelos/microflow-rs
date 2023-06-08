#![no_std]
#![no_main]

use panic_probe as _;

#[path = "../../../features/speech.rs"]
mod features;

#[rtic::app(device = hal::pac, peripherals = true, dispatchers = [I2SC0])]
mod app {
    use microflow::buffer::Buffer2D;
    use microflow::model;
    use rtt_target::{rprintln, rtt_init_print};

    #[model("../../models/speech.tflite")]
    struct Speech;

    #[shared]
    struct Shared {}

    #[local]
    struct Local {}

    fn print_prediction(prediction: Buffer2D<f32, 1, 4>) {
        rprintln!(
            "Prediction: {:.1}% silence, {:.1}% unknown, {:.1}% yes, {:.1}% no",
            prediction[0] * 100.,
            prediction[1] * 100.,
            prediction[2] * 100.,
            prediction[3] * 100.,
        );
        rprintln!(
            "Outcome: {}",
            match prediction.iamax_full().1 {
                0 => "SILENCE",
                1 => "UNKNOWN",
                2 => "YES",
                3 => "NO",
                _ => unreachable!(),
            }
        );
    }

    #[init]
    fn init(cx: init::Context) -> (Shared, Local, init::Monotonics) {
        hal::watchdog::Watchdog::new(cx.device.WDT).disable();
        rtt_init_print!();

        let yes_predicted = Speech::predict_quantized(super::features::YES);
        let no_predicted = Speech::predict_quantized(super::features::NO);
        rprintln!();
        rprintln!("Input sample: 'yes.wav'");
        print_prediction(yes_predicted);
        rprintln!();
        rprintln!("Input sample: 'no.wav'");
        print_prediction(no_predicted);

        (Shared {}, Local {}, init::Monotonics())
    }
}
