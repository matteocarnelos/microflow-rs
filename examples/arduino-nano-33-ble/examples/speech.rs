#![no_std]
#![no_main]

use panic_halt as _;

use core::fmt::Write;
use cortex_m::asm::nop;
use cortex_m_rt::entry;
use hal::gpio;
use hal::gpio::Level;
use hal::uarte::{Baudrate, Parity};
use hal::{uarte, Clocks, Uarte};
use microflow::buffer::Buffer2D;
use microflow::model;

#[path = "../../../features/speech.rs"]
mod features;

#[model("../../models/speech.tflite")]
struct Speech;

fn print_prediction(serial: &mut impl Write, prediction: Buffer2D<f32, 1, 4>) {
    writeln!(
        serial,
        "Prediction: {:.1}% silence, {:.1}% unknown, {:.1}% yes, {:.1}% no",
        prediction[0] * 100.,
        prediction[1] * 100.,
        prediction[2] * 100.,
        prediction[3] * 100.,
    )
    .unwrap();
    writeln!(
        serial,
        "Outcome: {}",
        match prediction.iamax_full().1 {
            0 => "SILENCE",
            1 => "UNKNOWN",
            2 => "YES",
            3 => "NO",
            _ => unreachable!(),
        }
    )
    .unwrap();
}

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

    let yes_predicted = Speech::predict_quantized(features::YES);
    writeln!(serial).unwrap();
    writeln!(serial, "Input sample: 'yes.wav'").unwrap();
    print_prediction(&mut serial, yes_predicted);

    let no_predicted = Speech::predict_quantized(features::NO);
    writeln!(serial).unwrap();
    writeln!(serial, "Input sample: 'no.wav'").unwrap();
    print_prediction(&mut serial, no_predicted);

    loop {
        nop();
    }
}
