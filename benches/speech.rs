use criterion::{criterion_group, criterion_main, Criterion};
use microflow::buffer::Buffer2D;
use microflow_macros::model;

#[model("models/speech.tflite")]
struct Speech;

fn speech_model(c: &mut Criterion) {
    let input = Buffer2D::from_element(0.5);

    c.bench_function("speech_model", |b| b.iter(|| Speech::predict(input)));
}

criterion_group!(benches, speech_model);
criterion_main!(benches);
