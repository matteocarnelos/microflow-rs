#![no_std]
#![no_main]

use esp_backtrace as _;

use esp_println::println;
use hal::{clock::ClockControl, peripherals::Peripherals, prelude::*, timer::TimerGroup, Rtc};
use libm::sinf;
use microflow::model;
use nalgebra::matrix;

#[model("../../models/sine.tflite")]
struct Model;

#[entry]
fn main() -> ! {
    let peripherals = Peripherals::take();
    let system = peripherals.DPORT.split();
    let clocks = ClockControl::boot_defaults(system.clock_control).freeze();

    let mut rtc = Rtc::new(peripherals.RTC_CNTL);
    let timer_group0 = TimerGroup::new(peripherals.TIMG0, &clocks);
    let mut wdt0 = timer_group0.wdt;
    let timer_group1 = TimerGroup::new(peripherals.TIMG1, &clocks);
    let mut wdt1 = timer_group1.wdt;

    rtc.rwdt.disable();
    wdt0.disable();
    wdt1.disable();

    let x = 1.5;
    let y_predicted = Model::predict(matrix![x])[0];
    let y_exact = sinf(x);
    println!(" ");
    println!("Predicted sin({}): {}", x, y_predicted);
    println!("Exact sin({}): {}", x, y_exact);
    println!("Error: {}", y_exact - y_predicted);

    loop {}
}
