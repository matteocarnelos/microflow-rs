use microflow::buffer::Buffer2D;
use microflow_macros::model;
use nalgebra::matrix;

#[model("models/speech.tflite")]
struct Speech;

#[test]
fn speech_model() {
    let input = Buffer2D::from_element(0.5);
    let output = matrix![0.15625, 0.2734375, 0.2734375, 0.296875];
    assert_eq!(Speech::predict(input), output);
}
