#![no_std]
#![no_main]

use esp_println::println;
use hal::{
    clock::ClockControl, peripherals::Peripherals, prelude::*, rtc_cntl::Rtc,
    system::SystemControl, timer::timg::TimerGroup,
};
use libm::sinf;
use microflow::model;
use nalgebra::matrix;

use esp_backtrace as _;

#[model("../../models/sine.tflite")]
struct Sine;

#[entry]
fn main() -> ! {
    let peripherals = Peripherals::take();
    let system = SystemControl::new(peripherals.SYSTEM);
    let clocks = ClockControl::boot_defaults(system.clock_control).freeze();

    let mut rtc = Rtc::new(peripherals.LPWR, None);
    let timer_group0 = TimerGroup::new_async(peripherals.TIMG0, &clocks);
    let mut wdt0 = timer_group0.wdt;
    let timer_group1 = TimerGroup::new_async(peripherals.TIMG1, &clocks);
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
