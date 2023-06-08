use microflow::buffer::Buffer2D;
use microflow_macros::model;

#[path = "../features/speech.rs"]
mod features;

#[model("models/speech.tflite")]
struct Speech;

fn print_prediction(prediction: Buffer2D<f32, 1, 4>) {
    println!(
        "Prediction: {:.1}% silence, {:.1}% unknown, {:.1}% yes, {:.1}% no",
        prediction[0] * 100.,
        prediction[1] * 100.,
        prediction[2] * 100.,
        prediction[3] * 100.,
    );
    println!(
        "Outcome: {}",
        match prediction.iamax_full().1 {
            0 => "SILENCE",
            1 => "UNKNOWN",
            2 => "YES",
            3 => "NO",
            _ => unreachable!(),
        }
    );
}

fn main() {
    let yes_predicted = Speech::predict_quantized(features::YES);
    let no_predicted = Speech::predict_quantized(features::NO);
    println!();
    println!("Input sample: 'yes.wav'");
    print_prediction(yes_predicted);
    println!();
    println!("Input sample: 'no.wav'");
    print_prediction(no_predicted);
}
