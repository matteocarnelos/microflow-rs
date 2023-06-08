use microflow::buffer::Buffer2D;
use microflow_macros::model;

mod features;

#[model("models/person_detect.tflite")]
struct PersonDetect;

fn print_prediction(prediction: Buffer2D<f32, 1, 2>) {
    println!(
        "Prediction: {:.1}% no person, {:.1}% person",
        prediction[0] * 100.,
        prediction[1] * 100.,
    );
    println!(
        "Outcome: {}",
        match prediction.iamax_full().1 {
            0 => "NO PERSON",
            1 => "PERSON",
            _ => unreachable!(),
        }
    );
}

fn main() {
    let person_predicted = PersonDetect::predict_quantized(features::PERSON);
    let no_person_predicted = PersonDetect::predict_quantized(features::NO_PERSON);
    println!();
    println!("Input sample: 'person.bmp'");
    print_prediction(person_predicted);
    println!();
    println!("Input sample: 'no_person.bmp'");
    print_prediction(no_person_predicted);
}
