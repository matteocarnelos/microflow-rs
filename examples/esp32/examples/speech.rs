#![no_std]
#![no_main]

use esp_hal::{clock::CpuClock, main, rtc_cntl::Rtc};
use esp_println::println;
use microflow::buffer::Buffer2D;
use microflow::model;

use esp_backtrace as _;

#[path = "../../../samples/features/speech.rs"]
mod features;

#[model("../../models/speech.tflite")]
struct Speech;

fn print_prediction(prediction: Buffer2D<f32, 1, 4>) {
    println!(
        "Prediction: {:.1}% silence, {:.1}% unknown, {:.1}% yes, {:.1}% no",
        prediction[0] * 100.,
        prediction[1] * 100.,
        prediction[2] * 100.,
        prediction[3] * 100.,
    );
    println!(
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

#[main]
fn main() -> ! {
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);
    let rtc = Rtc::new(peripherals.LPWR);

    let start = rtc.time_since_boot();
    let yes_predicted = Speech::predict_quantized(features::YES);
    let end = rtc.time_since_boot();
    println!(" ");
    println!("Input sample: 'yes.wav'");
    print_prediction(yes_predicted);
    println!("Execution time: {}", end - start);

    let start = rtc.time_since_boot();
    let no_predicted = Speech::predict_quantized(features::NO);
    let end = rtc.time_since_boot();
    println!(" ");
    println!("Input sample: 'no.wav'");
    print_prediction(no_predicted);
    println!("Execution time: {}", end - start);

    loop {}
}
