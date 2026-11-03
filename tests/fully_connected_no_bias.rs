use microflow_macros::model;
use nalgebra::matrix;

// A single `Dense(4, use_bias=False)` layer, full-integer quantized. The TFLite converter
// omits the bias of a FULLY_CONNECTED operator when it is all zeros, and marks the optional
// input as missing (tensor index -1). The expected output comes from the TFLite interpreter.
#[model("models/fully_connected_no_bias.tflite")]
struct FullyConnectedNoBias;

#[test]
fn fully_connected_no_bias_model() {
    let input = matrix![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
    let output = matrix![-0.07270127, -0.15994279, 0.014540253, 0.14540254];
    assert_eq!(FullyConnectedNoBias::predict(input), output);
}
