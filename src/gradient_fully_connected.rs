use crate::{
    activation::FusedActivation,
    buffer::Buffer2D,
    quantize::{quantize, Trainable},
    tensor::Tensor2D,
    update_layer::{accumulate_gradient_2D, clip_norm_2D_mut, is_cut_off},
};
use nalgebra::{SMatrix, SVector};
use simba::scalar::{SubsetOf, SupersetOf};
/// Clips the output gradient and computes backpropagation gradients for a quantized fully connected layer.
///
/// This function performs the backward pass preparation and delegates the main gradient
/// computation to `grad_fully_connected`. Before computing gradients, it applies optional
/// L2-norm clipping to the output gradient to prevent exploding gradients and improve
/// numerical stability during training.
///
/// The function updates:
/// - **Weight gradients** (`weights_gradient`) via accumulation.
/// - **Bias/constant gradients** (`constants_gradient`).
/// - **Input gradients** (returned), which are propagated to the previous layer.
///
/// # Type Parameters
/// - `T`: Quantized numeric type implementing the `Trainable` trait.
/// - `INPUT_ROWS`: Number of input samples (batch size).
/// - `INPUT_COLS`: Number of input features.
/// - `WEIGHTS_COLS`: Number of output neurons.
///
/// # Parameters
/// - `input`: Input tensor used in the forward fully connected layer.
/// - `output`: Output tensor produced during the forward pass.
/// - `weights`: Fully connected layer weights.
/// - `weights_gradient`: Mutable buffer where weight gradients are accumulated.
/// - `constants`: Fully connected layer constants (bias, scale, and quantization metadata).
/// - `constants_gradient`: Mutable buffer where constant gradients are accumulated.
/// - `activation`: Fused activation function applied during the forward pass.
/// - `output_grad`: Gradient of the loss with respect to the layer output.
/// - `bias_scale`: Scaling factor applied to bias gradients.
/// - `learning_rate`: Learning rate (provided for training pipeline consistency).
/// - `backwards_clip_val`: Maximum allowed L2 norm for the output gradient.
///   If ≤ 0, no clipping is applied.
///
/// # Returns
/// A `Buffer2D<i32, INPUT_ROWS, INPUT_COLS>` containing accumulated gradients
/// with respect to the input tensor.
///
/// # Notes
/// - Gradient clipping is applied in place before propagation.
/// - Gradients are accumulated as `i32` to preserve precision in quantized training.
/// - The returned input gradient is used for backpropagation into earlier layers.
///
/// # Typical use
/// Backward pass step for quantized fully connected (dense) layers.
pub fn update_grad_fully_connected<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const WEIGHTS_COLS: usize,
>(
    input: &Tensor2D<T, INPUT_ROWS, INPUT_COLS, 1>,
    output: &Tensor2D<T, INPUT_ROWS, WEIGHTS_COLS, 1>,
    weights: &Tensor2D<T, INPUT_COLS, WEIGHTS_COLS, 1>,
    weights_gradient: &mut Buffer2D<i32, INPUT_COLS, WEIGHTS_COLS>,
    // weights_gradient_unquantized: &mut Buffer2D<i32, INPUT_COLS, WEIGHTS_COLS>,
    constants: &(
        Buffer2D<f32, WEIGHTS_COLS, 1>,
        f32,
        Buffer2D<i32, 1, WEIGHTS_COLS>,
        i32,
    ),
    constants_gradient: &mut (
        Buffer2D<f32, WEIGHTS_COLS, 1>,
        f32,
        Buffer2D<i32, 1, WEIGHTS_COLS>,
        i32,
    ),
    activation: FusedActivation,
    mut output_grad: Buffer2D<i32, INPUT_ROWS, WEIGHTS_COLS>,
    // output_grad_unquantized: Buffer2D<f32, INPUT_ROWS, WEIGHTS_COLS>,
    bias_scale: f32,
    learning_rate: f32,
    backwards_clip_val: f32,
) -> Buffer2D<i32, INPUT_ROWS, INPUT_COLS> {
    clip_norm_2D_mut(&mut output_grad, backwards_clip_val);
    grad_fully_connected(
        input,
        output,
        weights,
        weights_gradient,
        constants_gradient,
        &activation,
        &output_grad,
        bias_scale,
    )
}
pub fn update_bias_fully_connected<const WEIGHTS_COLS: usize>(
    constants: &mut (
        Buffer2D<f32, WEIGHTS_COLS, 1>,
        f32,
        Buffer2D<i32, 1, WEIGHTS_COLS>,
        i32,
    ),
    bias_gradient: Buffer2D<f32, WEIGHTS_COLS, 1>,
) {
    constants.0 = SMatrix::from_fn(|i, j| constants.0[(i, j)] + bias_gradient[(i, j)]);
}
/// Computes gradients for a quantized fully connected (dense) layer.
///
/// This function performs the core backpropagation step for a fully connected layer,
/// accumulating gradients for the weights and bias/constants while also computing
/// the gradient with respect to the input tensor.
///
/// Specifically, it computes:
/// - **Weight gradients** (`weights_gradient`): accumulated using the input activations
///   (adjusted by the input zero point) multiplied by the output gradient.
/// - **Bias/constant gradients** (`constants_gradient`): accumulated using the output gradient
///   scaled by `bias_scale`.
/// - **Input gradients** (returned): propagated using the weights (adjusted by their zero point)
///   multiplied by the output gradient.
///
/// Quantization zero points are subtracted before multiplication to ensure correct
/// gradient computation in the quantized domain.
///
/// If a fused activation function (such as ReLU or ReLU6) was applied during the forward
/// pass, gradients are not propagated for outputs that were clipped or inactive.
///
/// # Type Parameters
/// - `T`: Quantized numeric type implementing the `Trainable` trait.
/// - `INPUT_ROWS`: Number of input samples (batch size).
/// - `INPUT_COLS`: Number of input features.
/// - `WEIGHTS_COLS`: Number of output neurons.
///
/// # Parameters
/// - `input`: Input tensor from the forward pass.
/// - `output`: Output tensor from the forward pass.
/// - `weights`: Fully connected layer weights.
/// - `weights_gradient`: Mutable buffer where weight gradients are accumulated.
/// - `constants_gradient`: Mutable tuple storing accumulated gradients for bias/constants.
/// - `activation`: Fused activation function applied during the forward pass.
/// - `output_grad`: Gradient of the loss with respect to the output tensor.
/// - `bias_scale`: Scaling factor applied to bias gradients.
///
/// # Returns
/// A `Buffer2D<i32, INPUT_ROWS, INPUT_COLS>` containing accumulated gradients
/// with respect to the input tensor.
///
/// # Notes
/// - Gradients are accumulated as `i32` to preserve precision during quantized training.
/// - Zero points for both inputs and weights are subtracted before multiplication.
/// - Gradients are skipped for outputs clipped by the activation function.
///
/// # Typical use
/// Core backward computation for quantized dense layers in low-precision neural networks.
pub fn grad_fully_connected<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const WEIGHTS_COLS: usize,
>(
    input: &Tensor2D<T, INPUT_ROWS, INPUT_COLS, 1>,
    output: &Tensor2D<T, INPUT_ROWS, WEIGHTS_COLS, 1>,
    weights: &Tensor2D<T, INPUT_COLS, WEIGHTS_COLS, 1>,
    weights_gradient: &mut Buffer2D<i32, INPUT_COLS, WEIGHTS_COLS>,
    constants_gradient: &mut (
        Buffer2D<f32, WEIGHTS_COLS, 1>,
        f32,
        Buffer2D<i32, 1, WEIGHTS_COLS>,
        i32,
    ),
    activation: &FusedActivation,
    output_grad: &Buffer2D<i32, INPUT_ROWS, WEIGHTS_COLS>,
    bias_scale: f32,
) -> Buffer2D<i32, INPUT_ROWS, INPUT_COLS> {
    let mut accum: Buffer2D<i32, INPUT_ROWS, INPUT_COLS> = SMatrix::zeros();
    let quantized_6 = quantize(6f32, output.scale[0], output.zero_point[0]);
    output_grad
        .row_iter()
        .zip(output.buffer.row_iter())
        .enumerate()
        .for_each(|(output_row, (row_grad, row_output))| {
            row_grad.iter().zip(row_output.iter()).enumerate().for_each(
                |(output_col, (output_grad_val, output_val))| {
                    //check if it is a cut-off gradient
                    if is_cut_off(*output_val, quantized_6, output.zero_point[0], activation) {
                        return;
                    }
                    //update constants gradient
                    constants_gradient.0[output_col] +=
                        f32::from_subset(output_grad_val) * bias_scale;
                    for weight_row in 0..INPUT_COLS {
                        //update weights gradient
                        let tmp_w = i32::from_subset(&input.buffer[(output_row, weight_row)])
                            - i32::from_subset(&input.zero_point[0]);
                        weights_gradient[(weight_row, output_col)] += tmp_w * output_grad_val;
                        //update input gradient
                        let tmp_i = i32::from_subset(&weights.buffer[(weight_row, output_col)])
                            - i32::from_subset(&weights.zero_point[0]);
                        accum[(output_row, weight_row)] += tmp_i * output_grad_val;
                    }
                },
            )
        });
    accum
}

pub fn grad_fully_connected_input<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const WEIGHTS_COLS: usize,
>(
    input: &Tensor2D<T, INPUT_ROWS, INPUT_COLS, 1>,
    output: &Tensor2D<T, INPUT_ROWS, WEIGHTS_COLS, 1>,
    weights: &Tensor2D<T, INPUT_COLS, WEIGHTS_COLS, 1>,
    activation: &FusedActivation,
    output_grad: &Buffer2D<i32, INPUT_ROWS, WEIGHTS_COLS>,
) -> Buffer2D<i32, INPUT_ROWS, INPUT_COLS> {
    let mut accum: Buffer2D<i32, INPUT_ROWS, INPUT_COLS> = SMatrix::zeros();
    let quantized_6 = quantize(6f32, output.scale[0], output.zero_point[0]);
    // let mut normalization_factor: Buffer2D<i32, INPUT_ROWS, INPUT_COLS> = SMatrix::zeros();
    for output_row in 0..INPUT_ROWS {
        for output_col in 0..WEIGHTS_COLS {
            let val = output.buffer[(output_row, output_col)];
            if !(match activation {
                FusedActivation::Relu => val > T::zero(),
                FusedActivation::Relu6 => val > T::zero() && val < quantized_6,
                _ => true,
            }) {
                continue;
            }
            for weight_row in 0..INPUT_COLS {
                let tmp = i32::from_subset(&weights.buffer[(weight_row, output_col)])
                    - i32::from_subset(&weights.zero_point[0]);
                accum[(output_row, weight_row)] +=
                    tmp * i32::from_subset(&output_grad[(output_row, output_col)]);
                // normalization_factor[(output_row, weight_row)] +=
                //     i32::from_subset(&output_grad[(output_row, output_col)]).abs();
            }
        }
    }
    // SMatrix::from_fn(|i, j| {
    //     if normalization_factor[(i, j)] != 0 {
    //         accum[(i, j)] / normalization_factor[(i, j)]
    //     } else {
    //         0
    //     }
    // })
    accum
}
pub fn grad_fully_connected_input_unquantized<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const WEIGHTS_COLS: usize,
>(
    input: &Tensor2D<T, INPUT_ROWS, INPUT_COLS, 1>,
    output: &Tensor2D<T, INPUT_ROWS, WEIGHTS_COLS, 1>,
    weights: &Tensor2D<T, INPUT_COLS, WEIGHTS_COLS, 1>,
    activation: &FusedActivation,
    output_grad: &Buffer2D<f32, INPUT_ROWS, WEIGHTS_COLS>,
) -> Buffer2D<f32, INPUT_ROWS, INPUT_COLS> {
    let mut accum: Buffer2D<f32, INPUT_ROWS, INPUT_COLS> = SMatrix::zeros();
    let quantized_6 = quantize(6f32, output.scale[0], output.zero_point[0]);
    for output_row in 0..INPUT_ROWS {
        for output_col in 0..WEIGHTS_COLS {
            let val = output.buffer[(output_row, output_col)];
            if !(match activation {
                FusedActivation::Relu => val > T::zero(),
                FusedActivation::Relu6 => val > T::zero() && val < quantized_6,
                _ => true,
            }) {
                continue;
            }
            for weight_row in 0..INPUT_COLS {
                let tmp = weights.scale[0]
                    * (f32::from_subset(&weights.buffer[(weight_row, output_col)])
                        - f32::from_subset(&weights.zero_point[0]));
                accum[(output_row, weight_row)] +=
                    tmp * f32::from_subset(&output_grad[(output_row, output_col)]);
            }
        }
    }
    accum
}
pub fn grad_fully_connected_bias<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const WEIGHTS_COLS: usize,
>(
    input: &Tensor2D<T, INPUT_ROWS, INPUT_COLS, 1>,
    output: &Tensor2D<T, INPUT_ROWS, WEIGHTS_COLS, 1>,
    weights: &Tensor2D<T, INPUT_COLS, WEIGHTS_COLS, 1>,
    activation: &FusedActivation,
    output_grad: &Buffer2D<i32, INPUT_ROWS, WEIGHTS_COLS>,
    bias_scale: f32,
) -> SVector<f32, WEIGHTS_COLS> {
    let quantized_6 = quantize(6f32, output.scale[0], output.zero_point[0]);
    let mut accum: SVector<i32, WEIGHTS_COLS> = SVector::zeros();
    for output_row in 0..INPUT_ROWS {
        for output_col in 0..WEIGHTS_COLS {
            let val = output.buffer[(output_row, output_col)].saturating_sub(output.zero_point[0]);
            if !(match activation {
                FusedActivation::Relu => val > T::zero(),
                FusedActivation::Relu6 => val > T::zero() && val < quantized_6,
                _ => true,
            }) {
                continue;
            }
            accum[output_col] += output_grad[(output_row, output_col)];
        }
    }
    //let scale = bias_scale / (weights.scale[0] * input.scale[0]).powi(2);
    //let scale = 1f32 / (weights.scale[0] * input.scale[0]).powi(2);
    accum.map(|el| {
        let tmp: f32 = i32::to_superset(&el);
        tmp * bias_scale
    })
}
pub fn grad_fully_connected_bias_unquantized<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const WEIGHTS_COLS: usize,
>(
    input: &Tensor2D<T, INPUT_ROWS, INPUT_COLS, 1>,
    output: &Tensor2D<T, INPUT_ROWS, WEIGHTS_COLS, 1>,
    weights: &Tensor2D<T, INPUT_COLS, WEIGHTS_COLS, 1>,
    activation: &FusedActivation,
    output_grad: &Buffer2D<f32, INPUT_ROWS, WEIGHTS_COLS>,
    bias_scale: f32,
) -> SVector<f32, WEIGHTS_COLS> {
    let quantized_6 = quantize(6f32, output.scale[0], output.zero_point[0]);
    let mut accum: SVector<f32, WEIGHTS_COLS> = SVector::zeros();
    for output_row in 0..INPUT_ROWS {
        for output_col in 0..WEIGHTS_COLS {
            let val = output.buffer[(output_row, output_col)].saturating_sub(output.zero_point[0]);
            if !(match activation {
                FusedActivation::Relu => val > T::zero(),
                FusedActivation::Relu6 => val > T::zero() && val < quantized_6,
                _ => true,
            }) {
                continue;
            }
            accum[output_col] += output_grad[(output_row, output_col)];
        }
    }
    //let scale = bias_scale / (weights.scale[0] * input.scale[0]).powi(2);
    //let scale = 1f32 / (weights.scale[0] * input.scale[0]).powi(2);
    accum
}
