use microflow_macros::model;
use nalgebra::{matrix, SMatrix};

#[model("models/speech.tflite")]
struct Speech;

#[test]
fn speech_model() {
    let input = [SMatrix::from_element([1.5])];
    let output = matrix![0.0390625, 0.23828125, 0.33984375, 0.375];
    assert_eq!(Speech::predict(input), output);
}
