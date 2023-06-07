#![feature(test)]

extern crate test;

use microflow::buffer::Buffer2D;
use microflow_macros::model;
use test::Bencher;

#[model("models/speech.tflite")]
struct Speech;

#[bench]
fn speech_model(b: &mut Bencher) {
    let input = Buffer2D::from_element(0.5);
    b.iter(|| {
        Speech::predict(input);
    });
}
