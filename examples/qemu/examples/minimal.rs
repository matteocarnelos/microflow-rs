#![no_std]
#![no_main]

use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hprintln};
use libm::sinf;
use nalgebra::vector;

use microflow::model;
use panic_halt as _;

#[model("../models/sine.tflite")]
struct Model;

#[entry]
fn main() -> ! {
    let x = 1.3;
    let predicted = Model::evaluate(vector![x])[0];
    let exact = sinf(x);
    hprintln!();
    hprintln!("Predicted sin({}): {}", x, predicted);
    hprintln!("Exact sin({}): {}", x, exact);
    hprintln!("Error: {}", exact - predicted);
    debug::exit(debug::EXIT_SUCCESS);
    loop {
        cortex_m::asm::nop();
    }
}
