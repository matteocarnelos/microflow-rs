#![no_std]
#![no_main]

use panic_probe as _;

#[path = "../../../samples/features/person_detect.rs"]
mod features;

#[rtic::app(device = hal::pac, peripherals = true, dispatchers = [I2SC0])]
mod app {
    use hal::clocks::*;
    use hal::efc::*;
    use hal::fugit::RateExtU32;
    use microflow::buffer::Buffer2D;
    use microflow::model;
    use rtt_target::{rprintln, rtt_init_print};

    #[model("../../models/person_detect.tflite")]
    struct PersonDetect;

    #[shared]
    struct Shared {}

    #[local]
    struct Local {}

    fn print_prediction(prediction: Buffer2D<f32, 1, 2>) {
        rprintln!(
            "Prediction: {:.1}% no person, {:.1}% person",
            prediction[0] * 100.,
            prediction[1] * 100.,
        );
        rprintln!(
            "Outcome: {}",
            match prediction.iamax_full().1 {
                0 => "NO PERSON",
                1 => "PERSON",
                _ => unreachable!(),
            }
        );
    }

    #[init]
    fn init(cx: init::Context) -> (Shared, Local, init::Monotonics) {
        rtt_init_print!();

        let clocks = Tokens::new(
            (cx.device.PMC, cx.device.SUPC, cx.device.UTMI),
            &cx.device.WDT.into(),
        );

        clocks.slck.configure_external_normal();
        let mainck = clocks.mainck.configure_external_normal(12.MHz()).unwrap();

        let pllack = clocks
            .pllack
            .configure(&mainck, PllaConfig { div: 1, mult: 12 })
            .unwrap();

        HostClockController::new(clocks.hclk, clocks.mck)
            .configure(
                &pllack,
                &mut Efc::new(cx.device.EFC, VddioLevel::V3),
                HostClockConfig {
                    pres: HccPrescaler::Div1,
                    div: MckDivider::Div1,
                },
            )
            .unwrap();

        let person_predicted = PersonDetect::predict_quantized(super::features::PERSON);
        let no_person_predicted = PersonDetect::predict_quantized(super::features::NO_PERSON);
        rprintln!();
        rprintln!("Input sample: 'person.bmp'");
        print_prediction(person_predicted);
        rprintln!();
        rprintln!("Input sample: 'no_person.bmp'");
        print_prediction(no_person_predicted);

        (Shared {}, Local {}, init::Monotonics())
    }
}
