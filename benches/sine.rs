use criterion::{criterion_group, criterion_main, Criterion};
use microflow_macros::model;
use nalgebra::matrix;

#[model("models/sine.tflite")]
struct Sine;

fn sine_model(c: &mut Criterion) {
    let input = matrix![0.5];

    c.bench_function("sine_model", |b| b.iter(|| Sine::predict(input)));
}

criterion_group!(benches, sine_model);
criterion_main!(benches);
