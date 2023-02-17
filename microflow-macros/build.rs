extern crate flatc_rust;

use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=examples/models/sine.tflite");
    println!("cargo:rerun-if-changed=flatbuffers/tflite.fbs");
    flatc_rust::run(flatc_rust::Args {
        inputs: &[Path::new("flatbuffers/tflite.fbs")],
        out_dir: Path::new("target/flatbuffers/"),
        ..Default::default()
    }).unwrap();
}
