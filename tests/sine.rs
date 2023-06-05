use microflow_macros::model;
use nalgebra::matrix;

#[model("models/sine.tflite")]
struct Sine;

#[test]
fn sine_model() {
    let input = matrix![1.5];
    let output = matrix![0.95928156];
    assert_eq!(Sine::predict(input), output);
}
