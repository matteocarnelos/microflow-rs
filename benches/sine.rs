#![feature(test)]

extern crate test;

use microflow_macros::model;
use nalgebra::matrix;
use test::Bencher;

#[model("models/sine.tflite")]
struct Sine;

#[bench]
fn sine_model(b: &mut Bencher) {
    let input = matrix![0.5];
    b.iter(|| {
        Sine::predict(input);
    });
}
