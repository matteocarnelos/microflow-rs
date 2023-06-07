#![no_std]
#![no_main]

use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hprintln};
use libm::sinf;
use nalgebra::matrix;

use microflow::model;
use panic_halt as _;

#[model("../../models/sine.tflite")]
struct Sine;

#[entry]
fn main() -> ! {
    let x = 0.5;
    let y_predicted = Sine::predict(matrix![x])[0];
    let y_exact = sinf(x);
    hprintln!();
    hprintln!("Predicted sin({}): {}", x, y_predicted);
    hprintln!("Exact sin({}): {}", x, y_exact);
    hprintln!("Error: {}", y_exact - y_predicted);
    debug::exit(debug::EXIT_SUCCESS);
    loop {
        cortex_m::asm::nop();
    }
}
