use microflow::buffer::Buffer2D;
use microflow_macros::model;
use nalgebra::matrix;

#[model("models/speech.tflite")]
struct Speech;

#[test]
fn speech_model() {
    let input = Buffer2D::from_element(0.5);
    let output = matrix![0.1640625, 0.2578125, 0.28125, 0.30859375];
    assert_eq!(Speech::predict(input), output);
}
