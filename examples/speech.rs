use nalgebra::SMatrix;

use microflow_macros::model;

#[model("examples/models/speech.tflite")]
struct Model;

fn main() {
    let input: [SMatrix<[f32; 1], 49, 40>; 1] = [SMatrix::from_element([0.])];
    let y_predicted = Model::evaluate(input);
    println!("{}", y_predicted);
}
