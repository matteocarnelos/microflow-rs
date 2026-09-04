//! A real Keras `Conv1D` model through `#[model]` (week 3, D3).
//!
//! `conv1d.tflite` is built by `ml/scripts/build_conv1d_model.py`
//! (Conv1D(8) -> AvgPool -> Conv1D(16) -> AvgPool -> Flatten -> Dense ->
//! Softmax, full-int8, input (1, 128, 1)). The macro folds the 18 file
//! operators into 6 layers (spec §2.1) and maps the rank-3 input onto the
//! user-facing `Buffer2D<f32, 128, 1>` (§2.2).
//!
//! The input window imitates the simulator's current signal: a 50 Hz sine
//! with harmonics at ~1 A amplitude, 80 ms at 1.6 kHz (WindowSpec(A)).

use microflow::model;
use nalgebra::SMatrix;

#[model("models/conv1d.tflite")]
struct Conv1DSpike;

fn main() {
    let input: SMatrix<f32, 128, 1> = SMatrix::from_fn(|t, _| {
        let ts = t as f32 / 1600.0;
        let mains = (2.0 * core::f32::consts::PI * 50.0 * ts).sin();
        let third = (2.0 * core::f32::consts::PI * 150.0 * ts).sin();
        let fifth = (2.0 * core::f32::consts::PI * 250.0 * ts).sin();
        1.0 * (mains + 0.15 * third + 0.07 * fifth)
    });
    let output = Conv1DSpike::predict(input);
    println!();
    print!("Probabilities: [");
    for (i, p) in output.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{p:.4}");
    }
    println!("]");
    let argmax = output
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(usize::MAX);
    println!("Predicted class: {argmax} (0=idle, 1=run, 2=jam, 3=overload)");
}
