use crate::{
    activation::FusedActivation,
    buffer::{Buffer2D, Buffer4D},
    quantize::{quantize, Trainable},
    tensor::{Tensor4D, TensorView, TensorViewPadding},
    update_layer::{clip_norm_4D_mut, get_input_index, is_cut_off},
};
use core::array;
use nalgebra::{SMatrix, SVector};
use simba::scalar::SupersetOf;
use num_traits::float::FloatCore;
/// Clips the output gradient and computes backpropagation gradients for a quantized depthwise convolution layer.
///
/// This function prepares the backward pass for a depthwise convolution by first applying
/// optional L2-norm clipping to the output gradients to prevent exploding gradients.
/// It then calls `grad_depthwise_conv_2d` to compute and accumulate gradients for:
/// - **Depthwise filter weights** (`weights_gradient`)
/// - **Bias/constant terms** (`constants_gradient`)
/// - **Input tensor** (returned), enabling gradient propagation to earlier layers.
///
/// Depthwise convolution differs from standard convolution in that each input channel
/// is convolved with its own dedicated filter (no cross-channel mixing).
///
/// # Type Parameters
/// - `T`: Quantized numeric type implementing the `Trainable` trait.
/// - `INPUT_ROWS`, `INPUT_COLS`: Spatial dimensions of the input tensor.
/// - `OUTPUT_ROWS`, `OUTPUT_COLS`: Spatial dimensions of the output tensor.
/// - `INPUT_CHANS`: Number of input channels.
/// - `WEIGHTS_ROWS`, `WEIGHTS_COLS`: Spatial dimensions of each depthwise filter.
/// - `WEIGHTS_CHANS`: Number of depthwise filters (typically equal to input channels).
/// - `FILTER_QUANTS`: Number of quantization groups for filter constants.
///
/// # Parameters
/// - `input`: Input tensor from the forward depthwise convolution.
/// - `weights`: Depthwise convolution filters.
/// - `weights_gradient`: Mutable buffer where filter gradients are accumulated.
/// - `constants`: Depthwise convolution constants (bias and quantization scaling).
/// - `constants_gradient`: Mutable buffer where constant gradients are accumulated.
/// - `outputs`: Output tensor produced during the forward pass.
/// - `output_grad`: Gradient of the loss with respect to the output tensor.
/// - `activation`: Fused activation function applied during the forward pass.
/// - `strides`: Convolution stride in `(row_stride, col_stride)` format.
/// - `padding`: Padding mode (`Same` or `Valid`).
/// - `bias_scale`: Scaling factors applied to bias gradients per quantization group.
/// - `learning_rate`: Learning rate (included for training pipeline consistency).
/// - `backwards_clip_val`: Maximum allowed L2 norm for the output gradient.
///   If ≤ 0, clipping is not applied.
///
/// # Returns
/// A `Buffer4D<i32, ...>` containing accumulated gradients with respect to the input tensor.
///
/// # Notes
/// - Gradient clipping is applied in place before propagation.
/// - Gradients are accumulated as `i32` to preserve precision in quantized training.
/// - Each channel is processed independently, consistent with depthwise convolution behavior.
/// - Activation clipping prevents gradients from flowing through inactive neurons.
///
/// # Typical use
/// Backward pass preparation for quantized depthwise convolution layers in embedded
/// or low-precision convolutional neural networks.
pub fn update_grad_depthwise_conv_2d<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const WEIGHTS_ROWS: usize,
    const WEIGHTS_COLS: usize,
    const WEIGHTS_CHANS: usize,
    const FILTER_QUANTS: usize,
>(
    input: &Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    weights: &Tensor4D<T, 1, WEIGHTS_ROWS, WEIGHTS_COLS, WEIGHTS_CHANS, FILTER_QUANTS>,
    weights_gradient: &mut Buffer4D<i32, 1, WEIGHTS_ROWS, WEIGHTS_COLS, WEIGHTS_CHANS>,
    constants: &(
        Buffer2D<f32, WEIGHTS_CHANS, 1>,
        Buffer2D<f32, FILTER_QUANTS, 1>,
    ),
    constants_gradient: &mut (
        Buffer2D<f32, WEIGHTS_CHANS, 1>,
        Buffer2D<f32, FILTER_QUANTS, 1>,
    ),
    outputs: Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, WEIGHTS_CHANS, 1>,
    mut output_grad: Buffer4D<i32, 1, OUTPUT_ROWS, OUTPUT_COLS, WEIGHTS_CHANS>,
    activation: FusedActivation,
    strides: (usize, usize),
    padding: TensorViewPadding,
    bias_scale: [f32; FILTER_QUANTS],
    learning_rate: f32,
    backwards_clip_val: f32,
) -> Buffer4D<i32, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> {
    clip_norm_4D_mut(&mut output_grad, backwards_clip_val);
    grad_depthwise_conv_2d(
        input,
        weights,
        weights_gradient,
        constants_gradient,
        &outputs,
        &output_grad,
        bias_scale,
        &activation,
        strides,
        padding,
    )
}
/// Computes gradients for a quantized depthwise 2-D convolution layer.
///
/// This function performs the core backpropagation step for depthwise convolution,
/// accumulating gradients for the depthwise filter weights and bias/constants,
/// while also computing gradients with respect to the input tensor.
///
/// In depthwise convolution, each output channel corresponds to exactly one input
/// channel and its own spatial filter. Gradients are therefore computed independently
/// per channel, without cross-channel accumulation.
///
/// The function performs three main operations:
/// - **Weight gradients** (`weights_gradient`): accumulated using the corresponding
///   input patch values (adjusted by the input zero point) multiplied by the output gradient.
/// - **Bias/constant gradients** (`constants_gradient`): accumulated per output channel,
///   scaled by the appropriate bias quantization factor.
/// - **Input gradients** (returned): propagated using the depthwise filter values
///   (adjusted by their zero point) multiplied by the output gradient.
///
/// Quantization zero points for both inputs and weights are subtracted before
/// multiplication to ensure mathematically correct gradient propagation.
///
/// If a fused activation function (such as ReLU or ReLU6) was applied during the
/// forward pass, gradients are not propagated for outputs that were clipped or inactive.
///
/// # Type Parameters
/// - `T`: Quantized numeric type implementing the `Trainable` trait.
/// - `INPUT_ROWS`, `INPUT_COLS`: Spatial dimensions of the input tensor.
/// - `OUTPUT_ROWS`, `OUTPUT_COLS`: Spatial dimensions of the output tensor.
/// - `INPUT_CHANS`: Number of input channels.
/// - `WEIGHTS_ROWS`, `WEIGHTS_COLS`: Spatial dimensions of each depthwise filter.
/// - `WEIGHTS_CHANS`: Number of depthwise output channels (typically equal to input channels).
/// - `FILTER_QUANTS`: Number of quantization groups for filter constants.
///
/// # Parameters
/// - `input`: Input tensor used during the forward depthwise convolution.
/// - `weights`: Depthwise convolution filters.
/// - `weights_gradient`: Mutable buffer where filter gradients are accumulated.
/// - `constants_gradient`: Mutable buffers where bias and scaling constant gradients are accumulated.
/// - `outputs`: Output tensor produced during the forward pass.
/// - `output_grad`: Gradient of the loss with respect to the output tensor.
/// - `bias_scale`: Scaling factors applied to bias gradients for each quantization group.
/// - `activation`: Fused activation function applied during the forward pass.
/// - `strides`: Convolution stride in `(row_stride, col_stride)` format.
/// - `padding`: Padding mode (`Same` or `Valid`).
///
/// # Returns
/// A `Buffer4D<i32, ...>` containing accumulated gradients with respect to the input tensor.
///
/// # Notes
/// - Gradients are accumulated as `i32` to preserve precision before requantization.
/// - Input and weight zero points are subtracted before multiplication.
/// - Padding masks ensure gradients are only applied to valid input regions.
/// - Each channel is processed independently, consistent with depthwise convolution.
/// - Activation clipping prevents gradients from propagating through inactive neurons.
///
/// # Typical use
/// Core backward computation for quantized depthwise convolution layers in
/// embedded or fixed-point convolutional neural networks.
pub fn grad_depthwise_conv_2d<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const WEIGHTS_ROWS: usize,
    const WEIGHTS_COLS: usize,
    const WEIGHTS_CHANS: usize,
    const FILTER_QUANTS: usize,
>(
    input: &Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    weights: &Tensor4D<T, 1, WEIGHTS_ROWS, WEIGHTS_COLS, WEIGHTS_CHANS, FILTER_QUANTS>,
    weights_gradient: &mut Buffer4D<i32, 1, WEIGHTS_ROWS, WEIGHTS_COLS, WEIGHTS_CHANS>,
    constants_gradient: &mut (
        Buffer2D<f32, WEIGHTS_CHANS, 1>,
        Buffer2D<f32, FILTER_QUANTS, 1>,
    ),
    outputs: &Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, WEIGHTS_CHANS, 1>,
    output_grad: &Buffer4D<i32, 1, OUTPUT_ROWS, OUTPUT_COLS, WEIGHTS_CHANS>,
    bias_scale: [f32; FILTER_QUANTS],
    activation: &FusedActivation,
    strides: (usize, usize),
    padding: TensorViewPadding,
) -> Buffer4D<i32, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> {
    let mut accum: Buffer4D<i32, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> =
        array::from_fn(|_| SMatrix::from_fn(|_, _| [0i32; INPUT_CHANS]));
    let quantized_6 = quantize(6f32, outputs.scale[0], outputs.zero_point[0]);
    let input_zero_point: i32 = input.zero_point[0].to_superset();
    for output_row in 0..OUTPUT_ROWS {
        for output_col in 0..OUTPUT_COLS {
            let view: TensorView<T, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS> =
                input.view((output_row, output_col), 0, padding, strides);
            let coord = get_input_index(
                WEIGHTS_ROWS,
                WEIGHTS_COLS,
                (output_row, output_col),
                padding,
                strides,
            );
            for output_channel in 0..WEIGHTS_CHANS {
                if is_cut_off(
                    outputs.buffer[0][(output_row, output_col)][output_channel],
                    quantized_6,
                    outputs.zero_point[0],
                    activation,
                ) {
                    continue;
                }
                constants_gradient.0[output_channel] +=
                    f32::from_subset(&output_grad[0][(output_row, output_col)][output_channel])
                        * bias_scale.get(output_channel).unwrap_or(&bias_scale[0]);
                for filter_row in 0..WEIGHTS_ROWS {
                    for filter_col in 0..WEIGHTS_COLS {
                        if view.mask[(filter_row, filter_col)] {
                            let tmp_w: i32 =
                                view.buffer[(filter_row, filter_col)][if output_channel
                                    < INPUT_CHANS
                                {
                                    output_channel
                                } else {
                                    0
                                }]
                                .to_superset();
                            weights_gradient[0][(filter_row, filter_col)][output_channel] += (tmp_w
                                - input_zero_point)
                                * &output_grad[0][(output_row, output_col)][output_channel];
                            let weights_zero_point: i32 = weights
                                .zero_point
                                .get(output_channel)
                                .copied()
                                .unwrap_or(weights.zero_point[0])
                                .to_superset();
                            let tmp_i: i32 = weights.buffer[0][(filter_row, filter_col)]
                                [output_channel]
                                .to_superset();
                            let cur_coord = (
                                (coord.0 + filter_row as i32) as usize,
                                (coord.1 + filter_col as i32) as usize,
                            );
                            accum[0][cur_coord][if output_channel < INPUT_CHANS {
                                output_channel
                            } else {
                                0
                            }] += (tmp_i - weights_zero_point)
                                * output_grad[0][(output_row, output_col)][output_channel]
                        }
                    }
                }
            }
        }
    }
    accum
}
pub fn update_bias_dephtwise_conv_2d<const FILTER_QUANTS: usize, const WEIGHT_CHANS: usize>(
    constants: &mut (
        Buffer2D<f32, WEIGHT_CHANS, 1>,
        Buffer2D<f32, FILTER_QUANTS, 1>,
    ),
    bias_gradient: Buffer2D<f32, WEIGHT_CHANS, 1>,
) {
    constants.0 = SMatrix::from_fn(|i, j| constants.0[(i, j)] + bias_gradient[(i, j)]);
}
pub fn grad_depthwise_conv_2d_inputs<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const WEIGHTS_ROWS: usize,
    const WEIGHTS_COLS: usize,
    const WEIGHTS_CHANS: usize,
    const FILTER_QUANTS: usize,
>(
    input: &Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    weights: &Tensor4D<T, 1, WEIGHTS_ROWS, WEIGHTS_COLS, WEIGHTS_CHANS, FILTER_QUANTS>,
    outputs: &Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, WEIGHTS_CHANS, 1>,
    output_grad: &Buffer4D<i32, 1, OUTPUT_ROWS, OUTPUT_COLS, WEIGHTS_CHANS>,
    activation: &FusedActivation,
    strides: (usize, usize),
    padding: TensorViewPadding,
) -> Buffer4D<i32, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> {
    let mut accum: Buffer4D<i32, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> =
        array::from_fn(|_| SMatrix::from_fn(|_, _| [0i32; INPUT_CHANS]));
    let quantized_6 = quantize(6f32, outputs.scale[0], outputs.zero_point[0]);
    let normalization_param = output_grad.iter().fold(0f32, |acc, val| {
        acc + val.iter().fold(0f32, |acc1, val1| {
            acc1 + val1
                .iter()
                .fold(0f32, |acc2, val2| acc2 + val2.abs() as f32)
        })
    }) as f32;
    for output_row in 0..OUTPUT_ROWS {
        for output_col in 0..OUTPUT_COLS {
            let coord = get_input_index(
                WEIGHTS_ROWS,
                WEIGHTS_COLS,
                (output_row, output_col),
                padding,
                strides,
            );
            for output_channel in 0..WEIGHTS_CHANS {
                let val = outputs.buffer[0][(output_row, output_col)][output_channel]
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
                        let cur_coord = (
                            (coord.0 + filter_row as i32) as usize,
                            (coord.1 + filter_col as i32) as usize,
                        );
                        let zero_point: i32 = weights
                            .zero_point
                            .get(output_channel)
                            .copied()
                            .unwrap_or(weights.zero_point[0])
                            .to_superset();
                        let tmp: i32 = weights.buffer[0][(filter_row, filter_col)][output_channel]
                            .to_superset();
                        accum[0][cur_coord][if output_channel < INPUT_CHANS {
                            output_channel
                        } else {
                            0
                        }] += (tmp - zero_point)
                            * output_grad[0][(output_row, output_col)][output_channel]
                    }
                }
            }
        }
    }
    accum.map(|batch| {
        batch.map(|channels| channels.map(|ch| (ch as f32 / normalization_param).round() as i32))
    })
}
pub fn grad_depthwise_conv_2d_weights<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const WEIGHTS_ROWS: usize,
    const WEIGHTS_COLS: usize,
    const WEIGHTS_CHANS: usize,
    const WEIGHTS_QUANTS: usize,
>(
    input: &Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    weights: &Tensor4D<T, 1, WEIGHTS_ROWS, WEIGHTS_COLS, WEIGHTS_CHANS, WEIGHTS_QUANTS>,
    outputs: &Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, WEIGHTS_CHANS, 1>,
    output_grad: &Buffer4D<i32, 1, OUTPUT_ROWS, OUTPUT_COLS, WEIGHTS_CHANS>,
    activation: &FusedActivation,
    strides: (usize, usize),
    padding: TensorViewPadding,
) -> Buffer4D<i32, 1, WEIGHTS_ROWS, WEIGHTS_COLS, WEIGHTS_CHANS> {
    let mut accum: Buffer4D<i32, 1, WEIGHTS_ROWS, WEIGHTS_COLS, WEIGHTS_CHANS> =
        array::from_fn(|_| SMatrix::from_fn(|_, _| [0i32; WEIGHTS_CHANS]));
    let quantized_6 = quantize(6f32, outputs.scale[0], outputs.zero_point[0]);
    for output_row in 0..OUTPUT_ROWS {
        for output_col in 0..OUTPUT_COLS {
            let view: TensorView<T, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS> =
                input.view((output_row, output_col), 0, padding, strides);
            for output_channel in 0..WEIGHTS_CHANS {
                let val = outputs.buffer[0][(output_row, output_col)][output_channel]
                    .saturating_sub(outputs.zero_point[0]);
                if !(match activation {
                    FusedActivation::Relu => val > T::zero(),
                    FusedActivation::Relu6 => val > T::zero() && val < quantized_6,
                    _ => true,
                }) {
                    continue;
                }
                for filter_row in 0..WEIGHTS_ROWS {
                    for filter_cols in 0..WEIGHTS_COLS {
                        let zero_point: i32 = input.zero_point[0].to_superset();
                        if view.mask[(filter_row, filter_cols)] {
                            let tmp: i32 = view.buffer[(filter_row, filter_cols)][if output_channel
                                < INPUT_CHANS
                            {
                                output_channel
                            } else {
                                0
                            }]
                            .to_superset();
                            accum[0][(filter_row, filter_cols)][output_channel] += (tmp
                                - zero_point)
                                * &output_grad[0][(output_row, output_col)][output_channel];
                        }
                    }
                }
            }
        }
    }
    accum
}
pub fn grad_depthwise_conv_2d_bias<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const WEIGHTS_ROWS: usize,
    const WEIGHTS_COLS: usize,
    const WEIGHTS_CHANS: usize,
    const FILTERS_QUANTS: usize,
>(
    input: &Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    weights: &Tensor4D<T, 1, WEIGHTS_ROWS, WEIGHTS_COLS, WEIGHTS_CHANS, FILTERS_QUANTS>,
    outputs: &Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, WEIGHTS_CHANS, 1>,
    output_grad: &Buffer4D<i32, 1, OUTPUT_ROWS, OUTPUT_COLS, WEIGHTS_CHANS>,
    activation: &FusedActivation,
    bias_scale: [f32; FILTERS_QUANTS],
) -> SVector<f32, WEIGHTS_CHANS> {
    let quantized_6 = quantize(6f32, outputs.scale[0], outputs.zero_point[0]);
    let mut accum: SVector<i32, WEIGHTS_CHANS> = SVector::zeros();
    for output_row in 0..OUTPUT_ROWS {
        for output_col in 0..OUTPUT_COLS {
            for output_batch in 0..WEIGHTS_CHANS {
                let val = outputs.buffer[0][(output_row, output_col)][output_batch]
                    .saturating_sub(outputs.zero_point[0]);
                if !(match activation {
                    FusedActivation::Relu => val > T::zero(),
                    FusedActivation::Relu6 => val > T::zero() && val < quantized_6,
                    _ => true,
                }) {
                    continue;
                }
                accum[output_batch] += output_grad[0][(output_row, output_col)][output_batch];
            }
        }
    }
    SMatrix::from_fn(|i, _| {
        // let filters_scale = weights.scale.get(i).copied().unwrap_or(weights.scale[0]);
        // let bias_scale_cur = bias_scale.get(i).copied().unwrap_or(bias_scale[0]);
        // let scale = bias_scale_cur / (filters_scale * input.scale[0]).powi(2);
        let tmp: f32 = accum[i] as f32;
        tmp * bias_scale.get(i).unwrap_or(&bias_scale[0])
    })
}
