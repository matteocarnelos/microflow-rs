#![feature(test)]

extern crate test;

use microflow::buffer::Buffer2D;
use microflow_macros::model;
use test::Bencher;

#[model("models/person_detect.tflite")]
struct PersonDetect;

#[bench]
fn person_detect_model(b: &mut Bencher) {
    let input = [Buffer2D::from_element([0.5])];
    b.iter(|| {
        PersonDetect::predict(input);
    });
}
