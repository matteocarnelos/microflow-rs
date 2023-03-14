use libm::sinf;
use microflow::model;
use nalgebra::vector;

#[model("examples/models/sine.tflite")]
struct Model;

fn main() {
    let x = 1.5;
    let y_predicted = Model::evaluate(vector![x])[0];
    let y_exact = sinf(x);
    println!("Predicted sin({}): {}", x, y_predicted);
    println!("Exact sin({}): {}", x, y_exact);
    println!("Error: {}", y_exact - y_predicted);
}
