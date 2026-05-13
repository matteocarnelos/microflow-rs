use crate::{
    activation::FusedActivation,
    buffer::{Buffer2D, Buffer4D},
    quantize::{quantize, Trainable},
    tensor::{Tensor4D, TensorView, TensorViewPadding},
    update_layer::{accumulate_gradient_4D, clip_norm_4D_mut, get_input_index, is_cut_off},
};
use core::{array, ops::Mul};
use nalgebra::{SMatrix, SVector};
use simba::scalar::SupersetOf;
/// Computes gradients for a quantized 2-D convolution layer during backpropagation.
///
/// This function performs three gradient computations simultaneously:
/// 1. **Weight gradients** (`weights_gradient`): accumulates gradients for each filter
///    element using the corresponding input patch and output gradient.
/// 2. **Bias/constant gradients** (`constants_gradient`): accumulates gradients for
///    per-filter bias or scaling constants.
/// 3. **Input gradients** (return value): propagates gradients back to the input tensor
///    so earlier layers can be updated.
///
/// The function respects quantization parameters (scale and zero point) and subtracts
/// zero-points before multiplication to correctly operate in the quantized domain.
/// If a fused activation function (e.g., ReLU or ReLU6) was applied in the forward pass,
/// gradients are not propagated for outputs that were clipped or inactive.
///
/// # Type Parameters
/// - `T`: Quantized numeric type implementing the `Trainable` trait.
/// - `INPUT_ROWS`, `INPUT_COLS`: Spatial dimensions of the input tensor.
/// - `OUTPUT_ROWS`, `OUTPUT_COLS`: Spatial dimensions of the output tensor.
/// - `INPUT_CHANS`: Number of input channels.
/// - `WEIGHTS_ROWS`, `WEIGHTS_COLS`: Spatial dimensions of each convolution kernel.
/// - `FILTERS_NUM`: Number of convolution filters (output channels).
/// - `FILTERS_QUANTS`: Number of quantization groups for filter constants.
///
/// # Parameters
/// - `input`: Input tensor used during the forward convolution.
/// - `weights`: Quantized convolution filters.
/// - `weights_gradient`: Mutable buffer where computed weight gradients are accumulated.
/// - `constants_gradient`: Mutable tuple containing accumulated gradients for bias and scaling constants.
/// - `outputs`: Output tensor from the forward convolution.
/// - `output_grad`: Gradient of the loss with respect to the convolution output.
/// - `activation`: Fused activation function applied after convolution.
/// - `strides`: Convolution stride in `(row_stride, col_stride)` format.
/// - `padding`: Padding mode (`Same` or `Valid`).
/// - `bias_scale`: Scaling factors applied to bias gradients for each quantization group.
///
/// # Returns
/// A `Buffer4D<i32, ...>` containing accumulated gradients with respect to the input tensor.
///
/// # Notes
/// - Gradients are accumulated as `i32` to preserve precision before requantization.
/// - Zero-points of both input and weights are subtracted before multiplication.
/// - Output positions masked by padding are ignored.
/// - Activation clipping prevents gradients from flowing through inactive neurons.
///
/// # Typical use
/// Backpropagation step for quantized convolutional layers in embedded,
/// fixed-point, or low-precision neural network training pipelines.
pub fn update_grad_conv_2d<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const WEIGHTS_ROWS: usize,
    const WEIGHTS_COLS: usize,
    const FILTER_QUANTS: usize,
    const FILTER_NUM: usize,
>(
    input: &Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    weights: &Tensor4D<T, FILTER_NUM, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS, FILTER_QUANTS>,
    weights_gradient: &mut Buffer4D<i32, FILTER_NUM, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS>,
    constants: &(
        Buffer2D<f32, FILTER_NUM, 1>,
        Buffer2D<f32, FILTER_QUANTS, 1>,
    ),
    constants_gradient: &mut (
        Buffer2D<f32, FILTER_NUM, 1>,
        Buffer2D<f32, FILTER_QUANTS, 1>,
    ),
    outputs: Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, FILTER_NUM, 1>,
    mut output_grad: Buffer4D<i32, 1, OUTPUT_ROWS, OUTPUT_COLS, FILTER_NUM>,
    activation: FusedActivation,
    strides: (usize, usize),
    padding: TensorViewPadding,
    bias_scale: [f32; FILTER_QUANTS],
    learning_rate: f32,
    backwards_clip_val: f32,
) -> Buffer4D<i32, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> {
    clip_norm_4D_mut(&mut output_grad, backwards_clip_val);
    grad_conv_2d(
        input,
        weights,
        weights_gradient,
        constants_gradient,
        &outputs,
        &output_grad,
        &activation,
        strides,
        padding,
        bias_scale,
    )
}
pub fn update_bias_conv2d<const FILTER_QUANTS: usize, const FILTER_NUM: usize>(
    constants: &mut (
        Buffer2D<f32, FILTER_NUM, 1>,
        Buffer2D<f32, FILTER_QUANTS, 1>,
    ),
    bias_gradient: Buffer2D<f32, FILTER_NUM, 1>,
) {
    constants.0 = SMatrix::from_fn(|i, j| constants.0[(i, j)] + bias_gradient[(i, j)]);
}
pub fn grad_conv_2d_inputs<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const WEIGHTS_ROWS: usize,
    const WEIGHTS_COLS: usize,
    const FILTERS_NUM: usize,
    const FILTERS_QUANTS: usize,
>(
    input: &Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    weights: &Tensor4D<T, FILTERS_NUM, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS, FILTERS_QUANTS>,
    outputs: &Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, FILTERS_NUM, 1>,
    output_grad: &Buffer4D<i32, 1, OUTPUT_ROWS, OUTPUT_COLS, FILTERS_NUM>,
    activation: &FusedActivation,
    strides: (usize, usize),
    padding: TensorViewPadding,
) -> Buffer4D<i32, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> {
    let mut accum: Buffer4D<i32, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> =
        array::from_fn(|_| SMatrix::from_fn(|_, _| array::from_fn(|_| 0i32)));
    let quantized_6 = quantize(6f32, outputs.scale[0], outputs.zero_point[0]);
    for output_batch in 0..FILTERS_NUM {
        for output_row in 0..OUTPUT_ROWS {
            for output_col in 0..OUTPUT_COLS {
                let coord = get_input_index(
                    WEIGHTS_ROWS,
                    WEIGHTS_COLS,
                    (output_row, output_col),
                    padding,
                    strides,
                );
                let val = outputs.buffer[0][(output_row, output_col)][output_batch]
                    .saturating_sub(outputs.zero_point[0]);
                if !(match activation {
                    FusedActivation::Relu => val > T::zero(),
                    FusedActivation::Relu6 => val > T::zero() && val < quantized_6,
                    _ => true,
                }) {
                    continue;
                }
                for filter_row in 0..WEIGHTS_ROWS {
                    if (coord.0 + filter_row as i32) < 0
                        || (coord.0 + filter_row as i32) >= INPUT_ROWS as i32
                    {
                        continue;
                    }
                    for filter_col in 0..WEIGHTS_COLS {
                        if (coord.1 + filter_col as i32) < 0
                            || (coord.1 + filter_col as i32) >= INPUT_COLS as i32
                        {
                            continue;
                        }
                        for filter_chans in 0..INPUT_CHANS {
                            let cur_coord = (
                                (coord.0 + filter_row as i32) as usize,
                                (coord.1 + filter_col as i32) as usize,
                            );
                            let filters_zero_point: i32 = weights
                                .zero_point
                                .get(output_batch)
                                .copied()
                                .unwrap_or(weights.zero_point[0])
                                .to_superset();

                            let tmp: i32 = weights.buffer[output_batch][(filter_row, filter_col)]
                                [filter_chans]
                                .to_superset();
                            accum[0][cur_coord][filter_chans] += (tmp - filters_zero_point)
                                * output_grad[0][(output_row, output_col)][output_batch];
                        }
                    }
                }
            }
        }
    }
    accum
}

pub fn grad_conv_2d_weights<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const WEIGHTS_ROWS: usize,
    const WEIGHTS_COLS: usize,
    const FILTERS_NUM: usize,
    const FILTERS_QUANTS: usize,
>(
    input: &Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    weights: &Tensor4D<T, FILTERS_NUM, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS, FILTERS_QUANTS>,
    outputs: &Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, FILTERS_NUM, 1>,
    output_grad: &Buffer4D<i32, 1, OUTPUT_ROWS, OUTPUT_COLS, FILTERS_NUM>,
    activation: &FusedActivation,
    strides: (usize, usize),
    padding: TensorViewPadding,
) -> Buffer4D<i32, FILTERS_NUM, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS> {
    let mut accum: Buffer4D<i32, FILTERS_NUM, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS> =
        array::from_fn(|_| SMatrix::from_fn(|_, _| [0i32; INPUT_CHANS]));
    let quantized_6 = quantize(6f32, outputs.scale[0], outputs.zero_point[0]);
    for output_row in 0..OUTPUT_ROWS {
        for output_col in 0..OUTPUT_COLS {
            let view: TensorView<T, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS> =
                input.view((output_row, output_col), 0, padding, strides);
            for output_batch in 0..FILTERS_NUM {
                let val = outputs.buffer[0][(output_row, output_col)][output_batch]
                    .saturating_sub(outputs.zero_point[0]);
                if !(match activation {
                    FusedActivation::Relu => val > T::zero(),
                    FusedActivation::Relu6 => val > T::zero() && val < quantized_6,
                    _ => true,
                }) {
                    continue;
                }
                let coord = get_input_index(
                    WEIGHTS_ROWS,
                    WEIGHTS_COLS,
                    (output_row, output_col),
                    padding,
                    strides,
                );
                let input_zero_point: i32 = input.zero_point[0].to_superset();
                for filter_row in 0..WEIGHTS_ROWS {
                    for filter_cols in 0..WEIGHTS_COLS {
                        for filter_chans in 0..INPUT_CHANS {
                            if view.mask[(filter_row, filter_cols)] {
                                let tmp: i32 = view.buffer[(filter_row, filter_cols)][filter_chans]
                                    .to_superset();
                                accum[output_batch][(filter_row, filter_cols)][filter_chans] +=
                                    (tmp - input_zero_point).mul(
                                        output_grad[0][(output_row, output_col)][output_batch],
                                    );
                            }
                        }
                    }
                }
            }
        }
    }
    accum
}
/// Computes gradients for a quantized 2-D convolution layer during backpropagation.
///
/// This function performs three gradient computations simultaneously:
/// 1. **Weight gradients** (`weights_gradient`): accumulates gradients for each filter
///    element using the corresponding input patch and output gradient.
/// 2. **Bias/constant gradients** (`constants_gradient`): accumulates gradients for
///    per-filter bias or scaling constants.
/// 3. **Input gradients** (return value): propagates gradients back to the input tensor
///    so earlier layers can be updated.
///
/// The function respects quantization parameters (scale and zero point) and subtracts
/// zero-points before multiplication to correctly operate in the quantized domain.
/// If a fused activation function (e.g., ReLU or ReLU6) was applied in the forward pass,
/// gradients are not propagated for outputs that were clipped or inactive.
///
/// # Type Parameters
/// - `T`: Quantized numeric type implementing the `Trainable` trait.
/// - `INPUT_ROWS`, `INPUT_COLS`: Spatial dimensions of the input tensor.
/// - `OUTPUT_ROWS`, `OUTPUT_COLS`: Spatial dimensions of the output tensor.
/// - `INPUT_CHANS`: Number of input channels.
/// - `WEIGHTS_ROWS`, `WEIGHTS_COLS`: Spatial dimensions of each convolution kernel.
/// - `FILTERS_NUM`: Number of convolution filters (output channels).
/// - `FILTERS_QUANTS`: Number of quantization groups for filter constants.
///
/// # Parameters
/// - `input`: Input tensor used during the forward convolution.
/// - `weights`: Quantized convolution filters.
/// - `weights_gradient`: Mutable buffer where computed weight gradients are accumulated.
/// - `constants_gradient`: Mutable tuple containing accumulated gradients for bias and scaling constants.
/// - `outputs`: Output tensor from the forward convolution.
/// - `output_grad`: Gradient of the loss with respect to the convolution output.
/// - `activation`: Fused activation function applied after convolution.
/// - `strides`: Convolution stride in `(row_stride, col_stride)` format.
/// - `padding`: Padding mode (`Same` or `Valid`).
/// - `bias_scale`: Scaling factors applied to bias gradients for each quantization group.
///
/// # Returns
/// A `Buffer4D<i32, ...>` containing accumulated gradients with respect to the input tensor.
///
/// # Notes
/// - Gradients are accumulated as `i32` to preserve precision before requantization.
/// - Zero-points of both input and weights are subtracted before multiplication.
/// - Output positions masked by padding are ignored.
/// - Activation clipping prevents gradients from flowing through inactive neurons.
///
/// # Typical use
/// Backpropagation step for quantized convolutional layers in embedded or low-precision
pub fn grad_conv_2d<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const WEIGHTS_ROWS: usize,
    const WEIGHTS_COLS: usize,
    const FILTERS_NUM: usize,
    const FILTERS_QUANTS: usize,
>(
    input: &Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    weights: &Tensor4D<T, FILTERS_NUM, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS, FILTERS_QUANTS>,
    weights_gradient: &mut Buffer4D<i32, FILTERS_NUM, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS>,
    constants_gradient: &mut (
        Buffer2D<f32, FILTERS_NUM, 1>,
        Buffer2D<f32, FILTERS_QUANTS, 1>,
    ),
    outputs: &Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, FILTERS_NUM, 1>,
    output_grad: &Buffer4D<i32, 1, OUTPUT_ROWS, OUTPUT_COLS, FILTERS_NUM>,
    activation: &FusedActivation,
    strides: (usize, usize),
    padding: TensorViewPadding,
    bias_scale: [f32; FILTERS_QUANTS],
) -> Buffer4D<i32, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> {
    let quantized_6 = quantize(6f32, outputs.scale[0], outputs.zero_point[0]);
    let mut accum: Buffer4D<i32, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> =
        array::from_fn(|_| SMatrix::from_fn(|_, _| array::from_fn(|_| 0i32)));
    for output_row in 0..OUTPUT_ROWS {
        for output_col in 0..OUTPUT_COLS {
            let view: TensorView<T, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS> =
                input.view((output_row, output_col), 0, padding, strides);
            for output_batch in 0..FILTERS_NUM {
                if is_cut_off(
                    outputs.buffer[0][(output_row, output_col)][output_batch],
                    quantized_6,
                    outputs.zero_point[0],
                    activation,
                ) {
                    continue;
                }
                constants_gradient.0[output_batch] +=
                    f32::from_subset(&output_grad[0][(output_row, output_col)][output_batch])
                        * bias_scale.get(output_batch).unwrap_or(&bias_scale[0]);
                let input_zero_point: i32 = input.zero_point[0].to_superset();
                let coord = get_input_index(
                    WEIGHTS_ROWS,
                    WEIGHTS_COLS,
                    (output_row, output_col),
                    padding,
                    strides,
                );
                for filter_row in 0..WEIGHTS_ROWS {
                    for filter_col in 0..WEIGHTS_COLS {
                        for filter_chans in 0..INPUT_CHANS {
                            if view.mask[(filter_row, filter_col)] {
                                let tmp_w: i32 = view.buffer[(filter_row, filter_col)]
                                    [filter_chans]
                                    .to_superset();
                                weights_gradient[output_batch][(filter_row, filter_col)]
                                    [filter_chans] += (tmp_w - input_zero_point)
                                    .mul(output_grad[0][(output_row, output_col)][output_batch]);
                                let cur_coord = (
                                    (coord.0 + filter_row as i32) as usize,
                                    (coord.1 + filter_col as i32) as usize,
                                );
                                let filters_zero_point: i32 = weights
                                    .zero_point
                                    .get(output_batch)
                                    .copied()
                                    .unwrap_or(weights.zero_point[0])
                                    .to_superset();

                                let tmp_i: i32 = weights.buffer[output_batch]
                                    [(filter_row, filter_col)][filter_chans]
                                    .to_superset();
                                accum[0][cur_coord][filter_chans] += (tmp_i - filters_zero_point)
                                    * output_grad[0][(output_row, output_col)][output_batch];
                            }
                        }
                    }
                }
            }
        }
    }
    accum
}
pub fn grad_conv_2d_bias<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const WEIGHTS_ROWS: usize,
    const WEIGHTS_COLS: usize,
    const FILTERS_NUM: usize,
    const FILTERS_QUANTS: usize,
>(
    input: &Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    weights: &Tensor4D<T, FILTERS_NUM, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS, FILTERS_QUANTS>,
    outputs: &Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, FILTERS_NUM, 1>,
    output_grad: &Buffer4D<i32, 1, OUTPUT_ROWS, OUTPUT_COLS, FILTERS_NUM>,
    activation: &FusedActivation,
    bias_scale: [f32; FILTERS_QUANTS],
) -> SVector<f32, FILTERS_NUM> {
    let quantized_6 = quantize(6f32, outputs.scale[0], outputs.zero_point[0]);
    let mut accum: SVector<i32, FILTERS_NUM> = SVector::zeros();
    for output_row in 0..OUTPUT_ROWS {
        for output_col in 0..OUTPUT_COLS {
            for output_batch in 0..FILTERS_NUM {
                let val = outputs.buffer[0][(output_row, output_col)][output_batch]
                    .saturating_sub(outputs.zero_point[0]);
                if !(match activation {
                    FusedActivation::Relu => val > T::zero(),
                    FusedActivation::Relu6 => val > T::zero() && val < quantized_6,
                    _ => true,
                }) {
                    continue;
                }
                accum[output_batch] =
                    accum[output_batch] + output_grad[0][(output_row, output_col)][output_batch];
            }
        }
    }
    SMatrix::from_fn(|i, _| {
        // let filters_scale = weights.scale.get(i).copied().unwrap_or(weights.scale[0]);
        // let scale = 1f32 / (filters_scale * input.scale[0]).powi(2);
        let tmp: f32 = accum[i] as f32;
        // tmp * scale
        // tmp / normalization_param
        tmp * bias_scale.get(i).unwrap_or(&bias_scale[0])
    })
}
