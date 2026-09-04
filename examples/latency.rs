//! Min/avg/max model latency (week 6, plan section 10).
//!
//! Criterion (benches/conv1d.rs) reports medians with confidence intervals;
//! this example adds the robust extrema the report table quotes — the same
//! `predict()` call timed over N runs on a quiet host.
//!
//! The `MICROFLOW_CONV2D_ONLY=1` cross-build (own CARGO_TARGET_DIR, see
//! ../NOTES.md week 6) turns both conv layers of each model onto the
//! generic conv_2d path — the "reshape trick" rows of the same table.

use microflow::model;
use nalgebra::SMatrix;
use std::time::Instant;

#[model("models/model_a.tflite")]
struct ModelA;

#[model("models/model_q.tflite")]
struct ModelQ;

const WARMUP: usize = 200;
const RUNS: usize = 20_000;

fn report(name: &str, runs: usize, mut predict: impl FnMut()) {
    for _ in 0..WARMUP {
        predict();
    }
    let mut min = f64::MAX;
    let mut max = 0.0f64;
    let mut sum = 0.0f64;
    for _ in 0..runs {
        let started = Instant::now();
        predict();
        let micros = started.elapsed().as_secs_f64() * 1e6;
        min = min.min(micros);
        max = max.max(micros);
        sum += micros;
    }
    println!(
        "{name}: min {min:.1} us | avg {:.1} us | max {max:.1} us (n={runs})",
        sum / runs as f64
    );
}

fn main() {
    // Deterministic windows (the kernels are data-independent — the timing
    // does not depend on the values, only on the shapes).
    let window_a = SMatrix::<f32, 128, 1>::from_fn(|i, _| ((i % 16) as f32 - 8.0) * 0.2);
    let window_q = SMatrix::<f32, 1024, 1>::from_fn(|i, _| ((i % 32) as f32 - 16.0) * 0.05);

    report("model_a (window 128)", RUNS, || {
        ModelA::predict(window_a);
    });
    report("model_q (window 1024)", RUNS / 4, || {
        ModelQ::predict(window_q);
    });
}
