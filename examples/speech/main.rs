use microflow::buffer::Buffer2D;
use microflow_macros::model;

mod samples;

#[model("models/speech.tflite")]
struct Speech;

fn print_prediction(prediction: Buffer2D<f32, 1, 4>) {
    println!(
        "Prediction: {:.1}% silence, {:.1}% unknown, {:.1}% yes, {:.1}% no",
        prediction[0] * 100.,
        prediction[1] * 100.,
        prediction[2] * 100.,
        prediction[3] * 100.
    );
}

fn main() {
    let yes_predicted = Speech::predict_quantized(samples::YES);
    let no_predicted = Speech::predict_quantized(samples::NO);
    let silence_predicted = Speech::predict_quantized(samples::SILENCE);
    println!("Input sample: 'Yes'");
    print_prediction(yes_predicted);
    println!();
    println!("Input sample: 'No'");
    print_prediction(no_predicted);
    println!();
    println!("Input sample: [silence]");
    print_prediction(silence_predicted);
}
