use libm::sinf;
use microflow::model;
use nalgebra::matrix;

#[model("models/sine.tflite")]
struct Sine;

fn main() {
    let x = 1.5;
    let y_predicted = Sine::predict(matrix![x])[0];
    let y_exact = sinf(x);
    println!("Predicted sin({}): {}", x, y_predicted);
    println!("Exact sin({}): {}", x, y_exact);
    println!("Error: {}", y_exact - y_predicted);
}
