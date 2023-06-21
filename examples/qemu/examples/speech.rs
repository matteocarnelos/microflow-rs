#![no_std]
#![no_main]

use cortex_m::asm::nop;
use cortex_m_rt::entry;
use cortex_m_semihosting::debug::{exit, EXIT_SUCCESS};
use cortex_m_semihosting::hprintln;
use microflow::buffer::Buffer2D;
use microflow::model;
use panic_halt as _;

#[path = "../../../samples/features/speech.rs"]
mod features;

#[model("../../models/speech.tflite")]
struct Speech;

fn print_prediction(prediction: Buffer2D<f32, 1, 4>) {
    hprintln!(
        "Prediction: {:.1}% silence, {:.1}% unknown, {:.1}% yes, {:.1}% no",
        prediction[0] * 100.,
        prediction[1] * 100.,
        prediction[2] * 100.,
        prediction[3] * 100.,
    );
    hprintln!(
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

#[entry]
fn main() -> ! {
    let yes_predicted = Speech::predict_quantized(features::YES);
    let no_predicted = Speech::predict_quantized(features::NO);
    hprintln!();
    hprintln!("Input sample: 'yes.wav'");
    print_prediction(yes_predicted);
    hprintln!();
    hprintln!("Input sample: 'no.wav'");
    print_prediction(no_predicted);

    exit(EXIT_SUCCESS);
    loop {
        nop()
    }
}
