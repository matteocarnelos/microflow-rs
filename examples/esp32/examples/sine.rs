#![no_std]
#![no_main]

use esp_hal::{clock::CpuClock, main, rtc_cntl::Rtc};
use esp_println::println;
use libm::sinf;
use microflow::model;
use nalgebra::matrix;

use esp_backtrace as _;

#[model("../../models/sine.tflite")]
struct Sine;

#[main]
fn main() -> ! {
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);
    let rtc = Rtc::new(peripherals.LPWR);

    let x = 0.5;
    let start = rtc.time_since_boot();
    let y_predicted = Sine::predict(matrix![x])[0];
    let end = rtc.time_since_boot();
    let y_exact = sinf(x);
    println!(" ");
    println!("Predicted sin({}): {}", x, y_predicted);
    println!("Exact sin({}): {}", x, y_exact);
    println!("Error: {}", y_exact - y_predicted);
    println!("Execution time: {}", end - start);

    loop {}
}
