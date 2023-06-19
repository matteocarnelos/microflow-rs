#![no_std]
#![no_main]

use core::fmt::Write;
use panic_halt as _;

use cortex_m::asm::nop;
use cortex_m_rt::entry;
use hal::gpio;
use hal::gpio::Level;
use hal::uarte::{Baudrate, Parity};
use hal::{uarte, Clocks, Uarte};
use libm::sinf;
use microflow::model;
use nalgebra::matrix;

#[model("../../models/sine.tflite")]
struct Sine;

#[entry]
fn main() -> ! {
    let p = hal::pac::Peripherals::take().unwrap();
    let _clocks = Clocks::new(p.CLOCK).enable_ext_hfosc();
    let port1 = gpio::p1::Parts::new(p.P1);

    let mut serial = Uarte::new(
        p.UARTE0,
        uarte::Pins {
            rxd: port1.p1_10.into_floating_input().degrade(),
            txd: port1.p1_03.into_push_pull_output(Level::High).degrade(),
            cts: None,
            rts: None,
        },
        Parity::EXCLUDED,
        Baudrate::BAUD115200,
    );

    let x = 0.5;
    let y_predicted = Sine::predict(matrix![x])[0];
    let y_exact = sinf(x);

    writeln!(serial).unwrap();
    writeln!(serial, "Predicted sin({}): {}", x, y_predicted).unwrap();
    writeln!(serial, "Exact sin({}): {}", x, y_exact).unwrap();
    writeln!(serial, "Error: {}", y_exact - y_predicted).unwrap();

    loop {
        nop()
    }
}
