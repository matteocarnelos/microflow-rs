use microflow_macros::model;
use nalgebra::{matrix, SMatrix};

#[model("models/person_detect.tflite")]
struct PersonDetect;

#[test]
fn person_detect_model() {
    let input = [SMatrix::from_element([0.5])];
    let output = matrix![0., 0.];
    assert_eq!(PersonDetect::predict(input), output);
}
