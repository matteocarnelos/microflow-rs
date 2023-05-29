use nalgebra::SMatrix;

use microflow_macros::model;

#[model("models/speech.tflite")]
struct Speech;

fn main() {
    let input: [SMatrix<[f32; 1], 49, 40>; 1] = [SMatrix::from_element([0.])];
    let output_predicted = Speech::predict(input);
    println!("{}", output_predicted);
}
