use microflow_macros::model;
use nalgebra::matrix;

#[model("examples/models/sine.tflite")]
struct Model;

fn main() {
    let mut rdr = csv::Reader::from_path("examples/microflow-vs-tflite/tflite.csv").unwrap();
    let mut wtr = csv::Writer::from_path("examples/microflow-vs-tflite/microflow.csv").unwrap();

    wtr.write_record(["x", "y_pred"]).unwrap();

    for record in rdr.records() {
        let record = record.unwrap();
        let x = record.get(0).unwrap();
        wtr.write_record([
            x,
            Model::evaluate(matrix![x.parse::<f32>().unwrap()])[0]
                .to_string()
                .as_str(),
        ])
        .unwrap();
    }
}
