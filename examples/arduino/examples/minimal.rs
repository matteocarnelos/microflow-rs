#![no_std]
#![no_main]

use arduino_hal::prelude::*;
use nalgebra::vector;
use ufmt::uwriteln;

use panic_halt as _;

#[arduino_hal::entry]
fn main() -> ! {
    let dp = arduino_hal::Peripherals::take().unwrap();
    let pins = arduino_hal::pins!(dp);
    let mut serial = arduino_hal::default_serial!(dp, pins, 57600);

    let input = vector![1.3];

    let output = microflow::tensor::QuantizedTensor::quantize(input, 0.024573976f32, -128i8);
    let output = microflow::ops::fully_connected(
        output,
        microflow::tensor::QuantizedTensor::new(
            nalgebra::matrix![50i8 , 107i8 , -78i8 , 85i8 , -45i8 , -102i8 , 124i8 , 104i8 , -83i8 , -35i8 , -48i8 , -127i8 , -67i8 , -73i8 , 34i8 , -109i8 ; ],
            0.004107884f32,
            0i8,
        ),
        0.0110060545f32,
        -128i8,
        microflow::activations::Activation::RELU,
        (
            -128i8,
            nalgebra::matrix![137.36838f32 ; -96.86502f32 ; 0f32 ; -68.53285f32 ; 0f32 ; 0f32 ; -83.93256f32 ; 0f32 ; 0f32 ; 0f32 ; 0f32 ; 0f32 ; 0f32 ; 0f32 ; 42.016727f32 ; 0f32 ; ],
            0.009171955f32,
            nalgebra::matrix![-6400i32 , -13696i32 , 9984i32 , -10880i32 , 5760i32 , 13056i32 , -15872i32 , -13312i32 , 10624i32 , 4480i32 , 6144i32 , 16256i32 , 8576i32 , 9344i32 , -4352i32 , 13952i32 ; ],
            0i32,
        ),
    );
    // let output = microflow::ops::fully_connected(output, microflow::tensor::QuantizedTensor::new(nalgebra::matrix![-23i8 , -38i8 , 29i8 , -8i8 , -17i8 , -34i8 , -34i8 , -35i8 , -46i8 , -44i8 , 71i8 , -24i8 , -15i8 , -29i8 , -13i8 , 57i8 ; 15i8 , 38i8 , -47i8 , -3i8 , -97i8 , -10i8 , -20i8 , 10i8 , -35i8 , -3i8 , -55i8 , 36i8 , -31i8 , -14i8 , 10i8 , -54i8 ; 40i8 , 27i8 , -41i8 , 19i8 , 37i8 , -29i8 , 21i8 , 33i8 , 31i8 , 3i8 , -34i8 , 20i8 , 22i8 , 21i8 , 25i8 , 21i8 ; 44i8 , -23i8 , -43i8 , 41i8 , -50i8 , -11i8 , 9i8 , 29i8 , -12i8 , -30i8 , -106i8 , -39i8 , 43i8 , -38i8 , 37i8 , -52i8 ; -15i8 , -3i8 , 32i8 , -30i8 , -45i8 , -46i8 , -12i8 , 39i8 , -41i8 , -28i8 , 45i8 , -15i8 , 17i8 , 22i8 , 16i8 , 43i8 ; -35i8 , 21i8 , 17i8 , 4i8 , 19i8 , 31i8 , 36i8 , 34i8 , 27i8 , -26i8 , -30i8 , -29i8 , 45i8 , -38i8 , 36i8 , -28i8 ; -18i8 , 25i8 , -5i8 , -11i8 , -60i8 , -25i8 , -23i8 , 17i8 , -47i8 , 29i8 , -104i8 , -20i8 , 0i8 , -28i8 , -36i8 , -43i8 ; -17i8 , -42i8 , -104i8 , -29i8 , 26i8 , 13i8 , 1i8 , 7i8 , -43i8 , -23i8 , -15i8 , -36i8 , -37i8 , 36i8 , -7i8 , -4i8 ; -39i8 , 32i8 , -16i8 , -28i8 , 2i8 , -7i8 , -6i8 , -26i8 , -35i8 , 3i8 , 12i8 , -10i8 , -37i8 , 25i8 , -34i8 , 30i8 ; 25i8 , -40i8 , -10i8 , 14i8 , -18i8 , 18i8 , -1i8 , -11i8 , -11i8 , 23i8 , -42i8 , 2i8 , -9i8 , 1i8 , 5i8 , 37i8 ; -8i8 , 36i8 , -26i8 , 16i8 , -13i8 , 36i8 , 44i8 , -5i8 , -42i8 , -6i8 , 6i8 , 3i8 , -34i8 , 26i8 , 44i8 , 1i8 ; -39i8 , 7i8 , 9i8 , -5i8 , 7i8 , -29i8 , 38i8 , -38i8 , -16i8 , 40i8 , 37i8 , 36i8 , 23i8 , 35i8 , 43i8 , -3i8 ; -14i8 , 1i8 , 25i8 , 44i8 , 37i8 , 5i8 , -7i8 , -2i8 , 17i8 , -28i8 , 26i8 , -25i8 , 41i8 , -10i8 , 20i8 , 20i8 ; -26i8 , 3i8 , 24i8 , 7i8 , 12i8 , -25i8 , -8i8 , 8i8 , 39i8 , -20i8 , -32i8 , 16i8 , -6i8 , -21i8 , -22i8 , 36i8 ; -15i8 , -27i8 , -127i8 , -14i8 , 26i8 , 30i8 , -21i8 , -34i8 , 45i8 , -32i8 , 40i8 , -8i8 , -46i8 , 23i8 , 32i8 , 49i8 ; 26i8 , 4i8 , 3i8 , 6i8 , -16i8 , 29i8 , 10i8 , 9i8 , -31i8 , 19i8 , 31i8 , 24i8 , -3i8 , 46i8 , -2i8 , 44i8 ; ], 0.009258529f32, 0i8, ), 0.008667699f32, -128i8, microflow::activations::Activation::RELU, (-128i8, nalgebra::matrix![0f32 ; 0f32 ; 83.31674f32 ; 0f32 ; -4.13821f32 ; 0f32 ; 0f32 ; -5.161006f32 ; -0.81118315f32 ; 0f32 ; 93.53294f32 ; 0f32 ; 0f32 ; -2.539356f32 ; -1.2226529f32 ; 52.32719f32 ; ], 0.011756278f32, nalgebra::matrix![12672i32 , -2688i32 , 35840i32 , -2944i32 , 19200i32 , 6912i32 , -3456i32 , -4480i32 , 25600i32 , 15744i32 , 19200i32 , 8832i32 , 3456i32 , -7296i32 , -19712i32 , -19712i32 ; ], 0i32, ), );
    let output = microflow::ops::fully_connected(
        output,
        microflow::tensor::QuantizedTensor::new(
            nalgebra::matrix![34i8 ; -17i8 ; -127i8 ; 15i8 ; -53i8 ; -14i8 ; 13i8 ; 29i8 ; -42i8 ; -9i8 ; 113i8 ; 27i8 ; -18i8 ; 4i8 ; -24i8 ; -83i8 ; ],
            0.012458475f32,
            0i8,
        ),
        0.008269669f32,
        7i8,
        microflow::activations::Activation::NONE,
        (
            7i8,
            nalgebra::matrix![-38.78261f32 ; ],
            0.0130581185f32,
            nalgebra::matrix![19456i32 ; ],
            0i32,
        ),
    );
    let y = output.dequantize();

    uwriteln!(&mut serial, "{:?}", y[0].to_be_bytes()).void_unwrap();

    let mut led = pins.d13.into_output();
    loop {
        led.toggle();
        arduino_hal::delay_ms(500);
    }
}
