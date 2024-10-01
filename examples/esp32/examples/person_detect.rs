#![no_std]
#![no_main]

use esp_println::println;
use hal::clock::CpuClock;
use hal::{
    clock::ClockControl, peripherals::Peripherals, prelude::*, rtc_cntl::Rtc,
    system::SystemControl, timer::timg::TimerGroup,
};
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

#[entry]
fn main() -> ! {
    let peripherals = Peripherals::take();
    let system = SystemControl::new(peripherals.SYSTEM);
    let clocks = ClockControl::configure(system.clock_control, CpuClock::Clock240MHz).freeze();

    let mut rtc = Rtc::new(peripherals.LPWR);
    let timer_group0 = TimerGroup::new_async(peripherals.TIMG0, &clocks);
    let mut wdt0 = timer_group0.wdt;
    let timer_group1 = TimerGroup::new_async(peripherals.TIMG1, &clocks);
    let mut wdt1 = timer_group1.wdt;

    rtc.rwdt.disable();
    wdt0.disable();
    wdt1.disable();

    let start = rtc.get_time_us();
    let person_predicted = PersonDetect::predict_quantized(features::PERSON);
    let end = rtc.get_time_us();
    println!(" ");
    println!("Input sample: 'person.bmp'");
    print_prediction(person_predicted);
    println!("Execution time: {} us", end - start);

    let start = rtc.get_time_us();
    let no_person_predicted = PersonDetect::predict_quantized(features::NO_PERSON);
    let end = rtc.get_time_us();
    println!(" ");
    println!("Input sample: 'no_person.bmp'");
    print_prediction(no_person_predicted);
    println!("Execution time: {} us", end - start);

    loop {}
}
