#![no_std]
#![no_main]

use esp_backtrace as _;

use esp_println::println;
use hal::{clock::ClockControl, peripherals::Peripherals, prelude::*, timer::TimerGroup, Rtc};
use libm::sinf;
use microflow::model;
use nalgebra::matrix;

#[model("../../models/sine.tflite")]
struct Sine;

#[entry]
fn main() -> ! {
    let peripherals = Peripherals::take();
    let mut system = peripherals.DPORT.split();
    let clocks = ClockControl::boot_defaults(system.clock_control).freeze();

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

    let x = 0.5;
    let start = rtc.get_time_us();
    let y_predicted = Sine::predict(matrix![x])[0];
    let end = rtc.get_time_us();
    let y_exact = sinf(x);
    println!(" ");
    println!("Predicted sin({}): {}", x, y_predicted);
    println!("Exact sin({}): {}", x, y_exact);
    println!("Error: {}", y_exact - y_predicted);
    println!("Execution time: {} us", end - start);

    loop {}
}
