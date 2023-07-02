#![no_std]
#![no_main]

use esp_backtrace as _;

use esp_println::println;
use hal::clock::CpuClock;
use hal::{clock::ClockControl, peripherals::Peripherals, prelude::*, timer::TimerGroup, Rtc};
use microflow::buffer::Buffer2D;
use microflow::model;

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

#[entry]
fn main() -> ! {
    let peripherals = Peripherals::take();
    let mut system = peripherals.DPORT.split();
    let clocks = ClockControl::configure(system.clock_control, CpuClock::Clock240MHz).freeze();

    let mut rtc = Rtc::new(peripherals.RTC_CNTL);
    let timer_group0 = TimerGroup::new(
        peripherals.TIMG0,
        &clocks,
        &mut system.peripheral_clock_control,
    );
    let mut wdt0 = timer_group0.wdt;
    let timer_group1 = TimerGroup::new(
        peripherals.TIMG1,
        &clocks,
        &mut system.peripheral_clock_control,
    );
    let mut wdt1 = timer_group1.wdt;

    rtc.rwdt.disable();
    wdt0.disable();
    wdt1.disable();

    let start = rtc.get_time_us();
    let yes_predicted = Speech::predict_quantized(features::YES);
    let end = rtc.get_time_us();
    println!(" ");
    println!("Input sample: 'yes.wav'");
    print_prediction(yes_predicted);
    println!("Execution time: {} us", end - start);

    let start = rtc.get_time_us();
    let no_predicted = Speech::predict_quantized(features::NO);
    let end = rtc.get_time_us();
    println!(" ");
    println!("Input sample: 'no.wav'");
    print_prediction(no_predicted);
    println!("Execution time: {} us", end - start);

    loop {}
}
