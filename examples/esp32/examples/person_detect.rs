#![no_std]
#![no_main]

use esp_backtrace as _;

use esp_println::println;
use hal::clock::CpuClock;
use hal::{clock::ClockControl, peripherals::Peripherals, prelude::*, timer::TimerGroup, Rtc};
use microflow::buffer::Buffer2D;
use microflow::model;

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

    println!(" ");
    println!("--- Benchmark ---");

    let mut benchmark_done = false;

    loop {
        if benchmark_done {
            continue;
        }
        for i in 1..101 {
            let start = rtc.get_time_us();
            let _ = PersonDetect::predict_quantized(features::PERSON);
            let end = rtc.get_time_us();
            println!("{},{:.0}", i, end - start);
        }
        println!("-----------------");
        benchmark_done = true;
    }
}
