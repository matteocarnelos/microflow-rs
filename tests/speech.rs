use microflow::buffer::Buffer2D;
use microflow_macros::model;
use nalgebra::matrix;

#[model("models/speech.tflite")]
struct Speech;

#[test]
fn speech_model() {
    let input = Buffer2D::from_element(1.5);
    let output = matrix![0.015625, 0.2734375, 0.328125, 0.39453125];
    assert_eq!(Speech::predict(input), output);
}
