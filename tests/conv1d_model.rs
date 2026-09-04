//! The week-3 model test: the real `conv1d.tflite` builds through `#[model]`
//! and predicts deterministically (spec §6, DoD week 3).
//!
//! Numerical parity against the TFLite interpreter is a separate fixture
//! test: `tests/conv1d_parity.rs` (±1 quantum, §5.3).

use microflow_macros::model;
use nalgebra::SMatrix;

#[model("models/conv1d.tflite")]
struct Conv1DSpike;

/// The synthetic current window used by all cases (deterministic, no RNG):
/// a 50 Hz sine + harmonics with the given amplitude envelope.
fn window(amplitude: f32) -> SMatrix<f32, 128, 1> {
    SMatrix::from_fn(|t, _| {
        let ts = t as f32 / 1600.0;
        let mains = (2.0 * core::f32::consts::PI * 50.0 * ts).sin();
        let third = (2.0 * core::f32::consts::PI * 150.0 * ts).sin();
        let fifth = (2.0 * core::f32::consts::PI * 250.0 * ts).sin();
        amplitude * (mains + 0.15 * third + 0.07 * fifth)
    })
}

#[test]
fn conv1d_model_predicts_softmax() {
    for amplitude in [0.05f32, 0.5, 1.0, 1.6] {
        let output = Conv1DSpike::predict(window(amplitude));
        let sum: f32 = output.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.02,
            "softmax output must sum to 1, got {sum} (amplitude {amplitude})"
        );
        for p in output.iter() {
            assert!((0.0..=1.0).contains(p), "probability out of range: {p}");
        }
    }
}

#[test]
fn conv1d_model_is_deterministic() {
    let first = Conv1DSpike::predict(window(1.0));
    let second = Conv1DSpike::predict(window(1.0));
    assert_eq!(first, second);
}

#[test]
fn conv1d_model_predict_quantized_smoke() {
    // A mid-range constant window quantizes to a constant offset around the
    // zero point; the quantized entry point must behave like the float one.
    let input = SMatrix::from_element(0i8);
    let output = Conv1DSpike::predict_quantized(input);
    let sum: f32 = output.iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.02,
        "softmax output must sum to 1, got {sum}"
    );
}
