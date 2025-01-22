#![no_std]
#![no_main]

use esp_hal::{clock::CpuClock, main, rtc_cntl::Rtc};
use esp_println::println;
use microflow::buffer::Buffer2D;
use microflow::model;

use esp_backtrace as _;

#[path = "../../../samples/features/person_detect.rs"]
mod features;

#[model("../../models/person_detect.tflite")]
struct PersonDetect;

fn print_prediction(prediction: Buffer2D<f32, 1, 2>) {
    println!(
        "Prediction: {:.1}% no person, {:.1}% person",
        prediction[0] * 100.,
        prediction[1] * 100.,
    );
    println!(
        "Outcome: {}",
        match prediction.iamax_full().1 {
            0 => "NO PERSON",
            1 => "PERSON",
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
    let person_predicted = PersonDetect::predict_quantized(features::PERSON);
    let end = rtc.time_since_boot();
    println!(" ");
    println!("Input sample: 'person.bmp'");
    print_prediction(person_predicted);
    println!("Execution time: {}", end - start);

    let start = rtc.time_since_boot();
    let no_person_predicted = PersonDetect::predict_quantized(features::NO_PERSON);
    let end = rtc.time_since_boot();
    println!(" ");
    println!("Input sample: 'no_person.bmp'");
    print_prediction(no_person_predicted);
    println!("Execution time: {}", end - start);

    loop {}
}
