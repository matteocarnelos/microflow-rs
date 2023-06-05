use microflow::model;
use nalgebra::matrix;

#[model("models/sine.tflite")]
struct Sine;

fn main() {
    let mut rdr = csv::Reader::from_path("tflite.csv").unwrap();
    let mut wtr = csv::Writer::from_path("microflow.csv").unwrap();

    wtr.write_record(["x", "y_pred"]).unwrap();

    for record in rdr.records() {
        let record = record.unwrap();
        let x = record.get(0).unwrap();
        wtr.write_record([
            x,
            Sine::predict(matrix![x.parse::<f32>().unwrap()])[0]
                .to_string()
                .as_str(),
        ])
        .unwrap();
    }
}
