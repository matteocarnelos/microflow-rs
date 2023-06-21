#![no_std]
#![no_main]

use core::fmt::Write;
use cortex_m_rt::entry;
use hal::gpio::Level;
use hal::uarte::{Baudrate, Parity};
use hal::{gpio, uarte, Clocks, Rtc, Uarte};
use microflow::buffer::Buffer2D;
use microflow::model;
use panic_halt as _;

const RTC_FREQ_MHZ: f32 = 0.032_768;

#[path = "../../../samples/features/person_detect.rs"]
mod features;

#[model("../../models/person_detect.tflite")]
struct PersonDetect;

fn print_prediction(serial: &mut impl Write, prediction: Buffer2D<f32, 1, 2>) {
    writeln!(
        serial,
        "Prediction: {:.1}% no person, {:.1}% person",
        prediction[0] * 100.,
        prediction[1] * 100.,
    )
    .unwrap();
    writeln!(
        serial,
        "Outcome: {}",
        match prediction.iamax_full().1 {
            0 => "NO PERSON",
            1 => "PERSON",
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
    let rtc = Rtc::new(p.RTC0, 0).unwrap();
    rtc.enable_counter();

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

    let start = rtc.get_counter();
    let person_predicted = PersonDetect::predict_quantized(features::PERSON);
    let end = rtc.get_counter();
    writeln!(serial).unwrap();
    writeln!(serial, "Input sample: 'person.bmp'").unwrap();
    print_prediction(&mut serial, person_predicted);
    writeln!(
        serial,
        "Execution time: {:.0} us",
        (end - start) as f32 / RTC_FREQ_MHZ
    )
    .unwrap();

    let start = rtc.get_counter();
    let no_person_predicted = PersonDetect::predict_quantized(features::NO_PERSON);
    let end = rtc.get_counter();
    writeln!(serial).unwrap();
    writeln!(serial, "Input sample: 'no_person.bmp'").unwrap();
    print_prediction(&mut serial, no_person_predicted);
    writeln!(
        serial,
        "Execution time: {:.0} us",
        (end - start) as f32 / RTC_FREQ_MHZ
    )
    .unwrap();

    writeln!(serial).unwrap();
    writeln!(serial, "--- Benchmark ---").unwrap();

    let mut benchmark_done = false;

    loop {
        if benchmark_done {
            continue;
        }
        for i in 1..101 {
            let start = rtc.get_counter();
            let _ = PersonDetect::predict_quantized(features::PERSON);
            let end = rtc.get_counter();
            writeln!(serial, "{},{:.0}", i, (end - start) as f32 / RTC_FREQ_MHZ).unwrap();
        }
        writeln!(serial, "-----------------").unwrap();
        benchmark_done = true;
    }
}
