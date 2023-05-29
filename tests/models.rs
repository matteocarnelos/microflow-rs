use microflow_macros::model;
use nalgebra::{matrix, SMatrix};

#[model("models/sine.tflite")]
struct Sine;

#[test]
fn sine_model() {
    let input = matrix![1.5];
    let output = matrix![0.98409057];
    assert_eq!(Sine::predict(input), output);
}

#[model("models/speech.tflite")]
struct Speech;

#[test]
fn speech_model() {
    let input = [SMatrix::from_element([1.5])];
    let output = matrix![0.0234375, 0.23828125, 0.3359375, 0.40234375];
    assert_eq!(Speech::predict(input), output);
}

#[model("models/person_detect.tflite")]
struct PersonDetect;

#[test]
fn person_detect_model() {
    let input = [SMatrix::from_element([0.5])];
    let output = matrix![0., 0.];
    assert_eq!(PersonDetect::predict(input), output);
}
