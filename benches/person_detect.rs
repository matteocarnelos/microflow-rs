use criterion::{criterion_group, criterion_main, Criterion};
use microflow::buffer::Buffer2D;
use microflow_macros::model;

#[model("models/person_detect.tflite")]
struct PersonDetect;

fn person_detect_model(c: &mut Criterion) {
    let input = [Buffer2D::from_element([0.5])];

    c.bench_function("person_detect_model", |b| {
        b.iter(|| PersonDetect::predict(input))
    });
}

criterion_group!(benches, person_detect_model);
criterion_main!(benches);
