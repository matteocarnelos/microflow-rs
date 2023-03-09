#![no_std]
#![no_main]

use arduino_hal::prelude::*;
use microflow::model;
use nalgebra::vector;
use ufmt::uwriteln;

use panic_halt as _;

#[model("../models/sine.tflite", capacity = 1)]
struct Model;

#[arduino_hal::entry]
fn main() -> ! {
    let dp = arduino_hal::Peripherals::take().unwrap();
    let pins = arduino_hal::pins!(dp);
    let mut serial = arduino_hal::default_serial!(dp, pins, 57600);

    let input = vector![1.3];

    let y = Model::evaluate(input)[0].to_be_bytes();

    uwriteln!(&mut serial, "{:x}{:x}{:x}{:x}", y[0], y[1], y[2], y[3]).void_unwrap();

    let mut led = pins.d13.into_output();
    loop {
        led.toggle();
        arduino_hal::delay_ms(500);
    }
}
