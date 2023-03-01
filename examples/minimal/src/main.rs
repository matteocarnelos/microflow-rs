use libm::sinf;
use microflow::model;
use nalgebra::vector;

#[model("../models/sine.tflite")]
struct Model;

fn main() {
    let x = 1.3;
    let predicted = Model::evaluate(vector![x])[0];
    let exact = sinf(x);
    println!("Predicted sin({}): {}", x, predicted);
    println!("Exact sin({}): {}", x, exact);
    println!("Error: {}", exact - predicted);
}
