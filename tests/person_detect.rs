use microflow::buffer::Buffer2D;
use microflow_macros::model;
use nalgebra::matrix;

#[model("models/person_detect.tflite")]
struct PersonDetect;

#[test]
fn person_detect_model() {
    let input = [Buffer2D::from_element([0.5])];
    let output = matrix![0.82421875, 0.17578125];
    assert_eq!(PersonDetect::predict(input), output);
}
