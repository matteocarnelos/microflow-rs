#![no_std]
#![no_main]

use avr_device::asm::nop;
use hal::prelude::*;
use libm::sinf;
use microflow::model;
use nalgebra::matrix;
use ufmt::uwriteln;
use ufmt_float::uFmt_f32;

use panic_halt as _;

#[model("../../models/sine.tflite")]
struct Sine;

#[hal::entry]
fn main() -> ! {
    let dp = hal::Peripherals::take().unwrap();
    let pins = hal::pins!(dp);
    let mut serial = hal::default_serial!(dp, pins, 57600);

    let x = 0.5;
    let y_predicted = Sine::predict(matrix![x])[0];
    let y_exact = sinf(x);
    let x_display = uFmt_f32::One(x);
    uwriteln!(&mut serial, "").void_unwrap();
    uwriteln!(
        &mut serial,
        "Predicted sin({}): {}",
        x_display,
        uFmt_f32::Five(y_predicted)
    )
    .void_unwrap();
    uwriteln!(
        &mut serial,
        "Exact sin({}): {}",
        x_display,
        uFmt_f32::Five(y_exact)
    )
    .void_unwrap();
    uwriteln!(
        &mut serial,
        "Error: {}",
        uFmt_f32::Five(y_exact - y_predicted)
    )
    .void_unwrap();

    loop {
        nop();
    }
}
