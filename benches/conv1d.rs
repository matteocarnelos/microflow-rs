//! Week-6 benchmarks (plan §10): the dedicated 1-D kernel vs the reshape
//! trick, and the node model latency per window.
//!
//! Fairness rules:
//! * the kernels see the same geometry and the same weight values — model
//!   A's two conv layers (`models/model_a.tflite`: 128x1 -> 126x8, then
//!   63x8 -> 61x16), int8, Valid padding, stride 1;
//! * per-channel quant scales are identity (1.0) on both sides, so the
//!   requant arithmetic is symmetric and the comparison isolates the loop
//!   structure (the dot product along the time axis vs the generic 4-D view
//!   machinery), not the constants;
//! * conv_2d's precomputed constants (compile-time in real models) are
//!   built once outside the timed loop.
//!
//! The `MICROFLOW_CONV2D_ONLY=1` A/B of a whole model (every conv through
//! conv_2d) is a *cross-build* comparison: run this bench twice with
//! separate `CARGO_TARGET_DIR`s (see ../NOTES.md, week 6) and compare the
//! model rows.

use criterion::{criterion_group, criterion_main, Criterion};
use microflow::activation::FusedActivation;
use microflow::buffer::Buffer2D;
use microflow::model;
use microflow::ops::{conv_1d, conv_2d, Conv1DOptions, Conv2DOptions};
use microflow::tensor::{Tensor2D, Tensor4D, TensorViewPadding};
use nalgebra::SMatrix;

#[model("models/model_a.tflite")]
struct ModelA;

#[model("models/model_q.tflite")]
struct ModelQ;

/// Deterministic int8 filler in [-125, 125].
fn filler(i: usize) -> i8 {
    ((i * 31 + 7) % 251 - 125) as i8
}

/// `(1, 1, T, C)` input.
fn input<const T: usize, const C: usize>(zp: i8) -> Tensor4D<i8, 1, 1, T, C, 1> {
    Tensor4D::new(
        [Buffer2D::from_fn(|_, t| {
            core::array::from_fn(|c| filler(t * C + c))
        })],
        [1.0],
        [zp],
    )
}

/// `(F, 1, K, C)` filters with per-channel identity scales.
fn filters<const F: usize, const K: usize, const C: usize>(zp: i8) -> Tensor4D<i8, F, 1, K, C, F> {
    Tensor4D::new(
        core::array::from_fn(|f| {
            Buffer2D::from_fn(|_, m| core::array::from_fn(|c| filler(f * K * C + m * C + c)))
        }),
        [1.0; F],
        [zp; F],
    )
}

/// Zero int32 biases (acc units; the identity scales make them symmetric).
fn biases<const F: usize>() -> Tensor2D<i32, F, 1, F> {
    Tensor2D::new(Buffer2D::from_fn(|f, _| f as i32), [0.0; F], [0; F])
}

/// The conv_1d side of one layer.
///
/// The input is (re)built inside the timed loop: the kernels take tensors
/// by value, and Tensor4D is not Clone. The construction is ~T*C stores vs
/// thousands of MACs of convolution, identical on both sides of the A/B —
/// the comparison isolates the loop structure, not the setup.
fn bench_conv_1d_layer(c: &mut Criterion, name: &str) {
    let filters = filters::<8, 3, 1>(0);
    let biases = biases::<8>();
    c.bench_function(name, |b| {
        b.iter(|| {
            let output: Tensor4D<i8, 1, 1, 126, 8, 1> = conv_1d(
                input::<128, 1>(0),
                &filters,
                &biases,
                [1.0],
                [0],
                Conv1DOptions {
                    fused_activation: FusedActivation::None,
                    padding: TensorViewPadding::Valid,
                    stride: 1,
                },
            );
            output
        })
    });
}

/// The conv_2d side of one layer (the reshape trick: the same 1-D work
/// expressed as a 4-D convolution with height-1 filters).
fn bench_conv_2d_layer(c: &mut Criterion, name: &str) {
    let filters = filters::<8, 3, 1>(0);
    // Compile-time constants in real models: bias term and multiplier per
    // filter — identity scales keep both at neutral values.
    let constants = (
        Buffer2D::from_fn(|f, _| f as f32),
        Buffer2D::from_fn(|_, _| 1.0f32),
    );
    c.bench_function(name, |b| {
        b.iter(|| {
            let output: Tensor4D<i8, 1, 1, 126, 8, 1> = conv_2d(
                input::<128, 1>(0),
                &filters,
                [1.0],
                [0],
                Conv2DOptions {
                    fused_activation: FusedActivation::None,
                    view_padding: TensorViewPadding::Valid,
                    strides: (1, 1),
                },
                constants,
            );
            output
        })
    });
}

/// Layer 2 of model A: (1, 1, 63, 8) -> (1, 1, 61, 16), kernel 3.
fn bench_conv_1d_layer2(c: &mut Criterion, name: &str) {
    let filters = filters::<16, 3, 8>(0);
    let biases = biases::<16>();
    c.bench_function(name, |b| {
        b.iter(|| {
            let output: Tensor4D<i8, 1, 1, 61, 16, 1> = conv_1d(
                input::<63, 8>(0),
                &filters,
                &biases,
                [1.0],
                [0],
                Conv1DOptions {
                    fused_activation: FusedActivation::None,
                    padding: TensorViewPadding::Valid,
                    stride: 1,
                },
            );
            output
        })
    });
}

fn bench_conv_2d_layer2(c: &mut Criterion, name: &str) {
    let filters = filters::<16, 3, 8>(0);
    let constants = (
        Buffer2D::from_fn(|f, _| f as f32),
        Buffer2D::from_fn(|_, _| 1.0f32),
    );
    c.bench_function(name, |b| {
        b.iter(|| {
            let output: Tensor4D<i8, 1, 1, 61, 16, 1> = conv_2d(
                input::<63, 8>(0),
                &filters,
                [1.0],
                [0],
                Conv2DOptions {
                    fused_activation: FusedActivation::None,
                    view_padding: TensorViewPadding::Valid,
                    strides: (1, 1),
                },
                constants,
            );
            output
        })
    });
}

fn bench_model_a(c: &mut Criterion) {
    // A deterministic window (the kernels are data-independent: a sawtooth
    // exercises the same MACs as a recorded window).
    let window = SMatrix::<f32, 128, 1>::from_fn(|i, _| ((i % 16) as f32 - 8.0) * 0.2);
    c.bench_function("model_a_predict_128", |b| {
        b.iter(|| ModelA::predict(window))
    });
}

fn bench_model_q(c: &mut Criterion) {
    let window = SMatrix::<f32, 1024, 1>::from_fn(|i, _| ((i % 32) as f32 - 16.0) * 0.05);
    c.bench_function("model_q_predict_1024", |b| {
        b.iter(|| ModelQ::predict(window))
    });
}

fn conv1d_benchmarks(c: &mut Criterion) {
    bench_conv_1d_layer(c, "conv1d_kernel_layer1_128x1_k3_f8");
    bench_conv_2d_layer(c, "conv2d_trick_layer1_128x1_k3_f8");
    bench_conv_1d_layer2(c, "conv1d_kernel_layer2_63x8_k3_f16");
    bench_conv_2d_layer2(c, "conv2d_trick_layer2_63x8_k3_f16");
    bench_model_a(c);
    bench_model_q(c);
}

criterion_group!(benches, conv1d_benchmarks);
criterion_main!(benches);
