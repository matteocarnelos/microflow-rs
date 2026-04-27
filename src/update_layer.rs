use core::i32;

use crate::{
    activation::FusedActivation,
    buffer::{Buffer2D, Buffer4D},
    ops::softmax_borrow,
    quantize::{Quantized, Trainable},
    tensor::{Tensor2D, Tensor4D, TensorViewPadding},
};
use libm::logf;
use nalgebra::SMatrix;
use simba::scalar::{SubsetOf, SupersetOf};
use num_traits::float::FloatCore;

/// Computes the L2 norm of a 2-D integer buffer, normalized by batch size.
///
/// Each element is first divided by `batch_size`, squared, and summed.
/// The square root of the total sum is returned.
///
/// # Parameters
/// - `mat`: Reference to the tensor buffer.
/// - `batch_size`: Number of samples used to normalize the values.
///
/// # Returns
/// The normalized L2 norm as `f32`.
pub fn safe_norm_2D<const ROWS: usize, const COLS: usize>(
    mat: &Buffer2D<i32, ROWS, COLS>,
    batch_size: usize,
) -> f32 {
    libm::sqrtf(
        mat.iter()
            .map(|el| {
                let tmp: f32 = *el as f32 / batch_size as f32;
                tmp * tmp
            })
            .fold(0f32, |acc, el| acc + el as f32),
    )
}
/// Computes the L2 norm of a 4-D integer buffer, normalized by batch size.
///
/// Iterates over batches, rows, columns, and channels. Each value is divided
/// by `batch_size`, squared, summed, and square-rooted.
///
/// # Parameters
/// - `mat`: Reference to the 4-D tensor buffer.
/// - `batch_size`: Number of samples used for normalization.
///
/// # Returns
/// The normalized L2 norm as `f32`.
pub fn safe_norm_4D<
    const BATCHES: usize,
    const ROWS: usize,
    const COLS: usize,
    const CHANS: usize,
>(
    mat: &Buffer4D<i32, BATCHES, ROWS, COLS, CHANS>,
    batch_size: usize,
) -> f32 {
    libm::sqrtf(
        mat.iter()
            .flat_map(|batch| {
                batch.iter().flat_map(|chans| {
                    chans.iter().map(|el| {
                        let tmp: f32 = *el as f32 / batch_size as f32;
                        tmp * tmp
                    })
                })
            })
            .fold(0f32, |acc, el| acc + el as f32),
    )
}
/// Updates a 4-D quantized weight tensor with optional gradient norm clipping.
///
/// If `clip_val > 0`, gradients are scaled so their global L2 norm does not exceed
/// `clip_val`. Otherwise, standard SGD update is applied.
///
/// # Parameters
/// - `weights`: Mutable quantized weight tensor.
/// - `weights_gradient`: Accumulated integer gradients.
/// - `batch_size`: Gradient normalization factor.
/// - `learning_rate`: SGD learning rate.
/// - `clip_val`: Maximum allowed gradient norm.
///
/// # Purpose
/// Prevents exploding gradients in deep or recurrent networks.
pub fn update_weights_clip_4D<
    T: Trainable,
    const BATCHES: usize,
    const ROWS: usize,
    const COLS: usize,
    const CHANS: usize,
    const QUANTS: usize,
>(
    weights: &mut Tensor4D<T, BATCHES, ROWS, COLS, CHANS, QUANTS>,
    weights_gradient: &Buffer4D<i32, BATCHES, ROWS, COLS, CHANS>,
    batch_size: usize,
    learning_rate: f32,
    clip_val: f32,
) {
    if clip_val <= 0f32 {
        for batch in 0..BATCHES {
            for row in 0..ROWS {
                for col in 0..COLS {
                    for chan in 0..CHANS {
                        let tmp: f32 = weights_gradient[batch][(row, col)][chan] as f32;

                        let tmp = learning_rate * tmp / batch_size as f32;
                        weights.buffer[batch][(row, col)][chan] = weights.buffer[batch][(row, col)]
                            [chan]
                            .saturating_sub(T::from_superset(&tmp).unwrap())
                    }
                }
            }
        }
    }
    let norm: f32 = safe_norm_4D(weights_gradient, batch_size);
    let scale = if norm > clip_val {
        // 812f32 / norm //speech
        // 1024f32 / norm //lenet
        // 256f32 / norm //sine
        clip_val / norm
    } else {
        1f32
    };
    for batch in 0..BATCHES {
        for row in 0..ROWS {
            for col in 0..COLS {
                for chan in 0..CHANS {
                    let tmp: f32 = weights_gradient[batch][(row, col)][chan] as f32;

                    let tmp = learning_rate * tmp * scale / batch_size as f32;
                    weights.buffer[batch][(row, col)][chan] = weights.buffer[batch][(row, col)]
                        [chan]
                        .saturating_sub(T::from_superset(&tmp).unwrap())
                }
            }
        }
    }
}
/// Updates a 2-D quantized weight tensor with optional gradient norm clipping.
///
/// Computes the gradient L2 norm and scales gradients if the norm exceeds
/// `clip_val`, then applies SGD update.
///
/// # Parameters
/// - `weights`: Mutable quantized weight tensor.
/// - `weights_gradient`: Accumulated gradients.
/// - `batch_size`: Normalization factor.
/// - `learning_rate`: SGD learning rate.
/// - `clip_val`: Maximum gradient norm allowed.
///
pub fn update_weights_clip_norm_2D<T: Trainable, const ROWS: usize, const COLS: usize>(
    weights: &mut Tensor2D<T, ROWS, COLS, 1>,
    weights_gradient: &Buffer2D<i32, ROWS, COLS>,
    batch_size: usize,
    learning_rate: f32,
    clip_val: f32,
) {
    if clip_val <= 0f32 {
        for row in 0..ROWS {
            for col in 0..COLS {
                let tmp: f32 = weights_gradient[(row, col)] as f32;

                let tmp = learning_rate * tmp / batch_size as f32;
                weights.buffer[(row, col)] =
                    weights.buffer[(row, col)].saturating_sub(T::from_superset(&tmp).unwrap())
            }
        }
    }
    let norm: f32 = safe_norm_2D(weights_gradient, batch_size);
    let scale = if norm > clip_val {
        // 812f32 / norm //speech
        // 1024f32 / norm //lenet
        // 256f32 / norm //sine
        clip_val / norm
    } else {
        1f32
    };
    for row in 0..ROWS {
        for col in 0..COLS {
            let tmp: f32 = weights_gradient[(row, col)] as f32;

            let tmp = learning_rate * tmp * scale / batch_size as f32;
            weights.buffer[(row, col)] =
                weights.buffer[(row, col)].saturating_sub(T::from_superset(&tmp).unwrap())
        }
    }
}
/// Returns a new 2-D buffer with gradients clipped to a maximum L2 norm.
///
/// If the norm exceeds `clip_val`, all elements are scaled proportionally.
///
/// # Parameters
/// - `mat`: Source gradient buffer.
/// - `clip_val`: Maximum allowed L2 norm.
///
/// # Returns
/// A new clipped buffer.
pub fn clip_norm_2D<const ROWS: usize, const COLS: usize>(
    mat: &Buffer2D<i32, ROWS, COLS>,
    clip_val: f32,
) -> Buffer2D<i32, ROWS, COLS> {
    if clip_val <= 0f32 {
        SMatrix::from_fn(|i, j| mat[(i, j)])
    } else {
        let norm: f32 = safe_norm_2D(mat, 1);
        let scale = if norm > clip_val {
            clip_val / norm
        } else {
            1f32
        };
        SMatrix::from_fn(|row, col| {
            let tmp: f32 = mat[(row, col)] as f32;
            i32::from_superset_unchecked(&(tmp * scale))
        })
    }
}
/// Clips a 2-D gradient buffer in place to a maximum L2 norm.
///
/// # Parameters
/// - `mat`: Mutable gradient buffer.
/// - `clip_val`: Maximum allowed L2 norm.
///
/// # Notes
/// Does nothing if `clip_val <= 0` or norm is already within limit.
pub fn clip_norm_2D_mut<const ROWS: usize, const COLS: usize>(
    mat: &mut Buffer2D<i32, ROWS, COLS>,
    clip_val: f32,
) {
    if clip_val <= 0f32 {
        return;
    } else {
        let norm: f32 = safe_norm_2D(mat, 1);
        let scale = if norm > clip_val {
            clip_val / norm
        } else {
            return;
        };
        for row in 0..ROWS {
            for col in 0..COLS {
                let tmp: f32 = mat[(row, col)] as f32;
                mat[(row, col)] = i32::from_superset_unchecked(&(tmp * scale))
            }
        }
    }
}
/// Returns a new 4-D buffer with gradients clipped to a maximum L2 norm.
///
/// # Parameters
/// - `mat`: Source gradient tensor.
/// - `clip_val`: Maximum allowed L2 norm.
///
/// # Returns
/// A new clipped tensor.
pub fn clip_norm_4D<
    const BATCHES: usize,
    const ROWS: usize,
    const COLS: usize,
    const CHANS: usize,
>(
    mat: &Buffer4D<i32, BATCHES, ROWS, COLS, CHANS>,
    clip_val: f32,
) -> Buffer4D<i32, BATCHES, ROWS, COLS, CHANS> {
    if clip_val <= 0f32 {
        core::array::from_fn(|batch| {
            SMatrix::from_fn(|i, j| core::array::from_fn(|chan| mat[batch][(i, j)][chan]))
        })
    } else {
        let norm: f32 = safe_norm_4D(mat, 1);
        let scale = if norm > clip_val {
            clip_val / norm
        } else {
            1f32
        };
        core::array::from_fn(|batch| {
            SMatrix::from_fn(|row, col| {
                core::array::from_fn(|chan| {
                    let tmp: f32 = mat[batch][(row, col)][chan] as f32;
                    i32::from_superset_unchecked(&(tmp * scale))
                })
            })
        })
    }
}
/// Clips a 4-D gradient tensor in place to a maximum L2 norm.
///
/// # Parameters
/// - `mat`: Mutable gradient tensor.
/// - `clip_val`: Maximum allowed L2 norm.
///
/// # Purpose
/// Prevents excessively large updates during training.
pub fn clip_norm_4D_mut<
    const BATCHES: usize,
    const ROWS: usize,
    const COLS: usize,
    const CHANS: usize,
>(
    mat: &mut Buffer4D<i32, BATCHES, ROWS, COLS, CHANS>,
    clip_val: f32,
) {
    if clip_val <= 0f32 {
        return;
    } else {
        let norm: f32 = safe_norm_4D(mat, 1);
        let scale = if norm > clip_val {
            clip_val / norm
        } else {
            return;
        };
        for batch in 0..BATCHES {
            for row in 0..ROWS {
                for col in 0..COLS {
                    for chan in 0..CHANS {
                        let tmp: f32 = mat[batch][(row, col)][chan] as f32;
                        mat[batch][(row, col)][chan] = i32::from_superset_unchecked(&(tmp * scale));
                    }
                }
            }
        }
    }
}
/// Updates floating-point weights using SGD.
///
/// Applies:
/// `weight -= learning_rate * gradient / batch_size`
///
/// # Parameters
/// - `weights`: Mutable floating-point weights.
/// - `weights_gradient`: Floating-point gradients.
/// - `batch_size`: Normalization factor.
/// - `learning_rate`: SGD learning rate.
pub fn update_weights_2D_float<const ROWS: usize, const COLS: usize>(
    weights: &mut Buffer2D<f32, ROWS, COLS>,
    weights_gradient: &Buffer2D<f32, ROWS, COLS>,
    batch_size: usize,
    learning_rate: f32,
) {
    for row in 0..ROWS {
        for col in 0..COLS {
            weights[(row, col)] -= learning_rate * weights_gradient[(row, col)] / batch_size as f32
        }
    }
}
/// Updates a 4-D quantized weight tensor using SGD.
///
/// Each weight is adjusted using normalized integer gradients.
///
/// # Parameters
/// - `weights`: Mutable quantized tensor.
/// - `weights_gradient`: Accumulated gradients.
/// - `batch_size`: Gradient normalization factor.
/// - `learning_rate`: SGD learning rate.
pub fn update_weights_4D<
    T: Trainable,
    const BATCHES: usize,
    const ROWS: usize,
    const COLS: usize,
    const CHANS: usize,
    const QUANTS: usize,
>(
    weights: &mut Tensor4D<T, BATCHES, ROWS, COLS, CHANS, QUANTS>,
    weights_gradient: &Buffer4D<i32, BATCHES, ROWS, COLS, CHANS>,
    batch_size: usize,
    learning_rate: f32,
) {
    for batch in 0..BATCHES {
        for i in 0..ROWS {
            for j in 0..COLS {
                for channel in 0..CHANS {
                    let tmp: f32 = weights_gradient[batch][(i, j)][channel] as f32;
                    weights.buffer[batch][(i, j)][channel] = weights.buffer[batch][(i, j)][channel]
                        .saturating_sub(
                            T::from_superset(&(learning_rate * tmp / batch_size as f32).round())
                                .unwrap(),
                        );
                }
            }
        }
    }
}
/// Updates precomputed constants for a quantized fully connected layer.
///
/// Computes row sums of weights multiplied by the input zero-point,
/// used to accelerate inference.
///
/// # Parameters
/// - `weights`: Quantized weight tensor.
/// - `constants`: Tuple storing scaling constants and offsets.
/// - `input_zero_point`: Quantization zero point of the input.
pub fn update_constants_fully_connected<
    T: Trainable,
    const ROWS: usize,
    const COLS: usize,
    const QUANTS: usize,
>(
    weights: &Tensor2D<T, ROWS, COLS, QUANTS>,
    constants: &mut (Buffer2D<f32, COLS, 1>, f32, Buffer2D<i32, 1, COLS>, i32),
    input_zero_point: T,
) {
    constants.2 = Buffer2D::from(SMatrix::from_rows(&[&weights
        .buffer
        .cast::<i32>()
        .row_sum()
        * i32::from_subset(&input_zero_point)]));
}
/// Accumulates quantized gradients into an integer gradient buffer.
///
/// Converts quantized values into integer representation and adds them.
///
/// # Parameters
/// - `current_gradient`: Newly computed gradient.
/// - `weights_gradient`: Accumulated gradient buffer.
pub fn accumulate_gradient_2D<T: Trainable, const ROWS: usize, const COLS: usize>(
    current_gradient: &Buffer2D<T, ROWS, COLS>,
    weights_gradient: &mut Buffer2D<i32, ROWS, COLS>,
) {
    for row in 0..ROWS {
        for col in 0..COLS {
            let tmp: i32 = current_gradient[(row, col)].to_superset();
            weights_gradient[(row, col)] += tmp;
        }
    }
}
/// Accumulates gradients for a 4-D tensor into an integer buffer.
///
/// # Parameters
/// - `current_gradient`: New quantized gradient tensor.
/// - `weights_gradient`: Accumulated gradient tensor.
pub fn accumulate_gradient_4D<
    T: Trainable,
    const BATCHES: usize,
    const ROWS: usize,
    const COLS: usize,
    const CHANS: usize,
>(
    current_gradient: &Buffer4D<T, BATCHES, ROWS, COLS, CHANS>,
    weights_gradient: &mut Buffer4D<i32, BATCHES, ROWS, COLS, CHANS>,
) {
    for batch in 0..BATCHES {
        for row in 0..ROWS {
            for col in 0..COLS {
                for channel in 0..CHANS {
                    let tmp: i32 = current_gradient[batch][(row, col)][channel].to_superset();
                    weights_gradient[batch][(row, col)][channel] += tmp;
                }
            }
        }
    }
}

pub fn mse_loss<T: Quantized, const ROWS: usize, const COLS: usize>(
    output_p: &Tensor2D<T, ROWS, COLS, 1>,
    output_gt: &Tensor2D<T, ROWS, COLS, 1>,
) -> f32 {
    let difference: Buffer2D<f32, ROWS, COLS> = SMatrix::from_fn(|i, j| {
        let casted_p: f32 = T::to_superset(&output_p.buffer[(i, j)]);
        let casted_gt: f32 = T::to_superset(&output_gt.buffer[(i, j)]);
        output_p.scale[0] * (casted_p - casted_gt)
    });
    0.5f32 * difference.component_mul(&difference).sum()
}
/// Computes gradient of MSE loss for quantized tensors.
///
/// Gradient:
/// `prediction − ground_truth`
///
/// # Parameters
/// - `output_p`: Predicted tensor.
/// - `output_gt`: Ground truth tensor.
///
/// # Returns
/// Integer gradient buffer.
pub fn mse_grad<T: Trainable, const ROWS: usize, const COLS: usize>(
    output_p: &Tensor2D<T, ROWS, COLS, 1>,
    output_gt: &Tensor2D<T, ROWS, COLS, 1>,
) -> Buffer2D<i32, ROWS, COLS> {
    SMatrix::from_fn(|i, j| {
        i32::from_subset(&output_p.buffer[(i, j)]) - i32::from_subset(&output_gt.buffer[(i, j)])
    })
}
/// Computes gradient of cross-entropy loss using softmax output.
///
/// Gradient:
/// `softmax(input) − label`
///
/// # Parameters
/// - `input`: Logits tensor.
/// - `output_scale`: Output quantization scale.
/// - `output_zero_point`: Output quantization zero point.
/// - `label`: Ground truth labels.
///
/// # Returns
/// Integer gradient buffer.
pub fn crossentropy_grad<T: Trainable, const ROWS: usize, const COLS: usize>(
    input: &Tensor2D<T, ROWS, COLS, 1>,
    output_scale: f32,
    output_zero_point: T,
    label: &Tensor2D<T, ROWS, COLS, 1>,
) -> Buffer2D<i32, ROWS, COLS> {
    let softm = softmax_borrow(&input, [output_scale], [output_zero_point]);

    // let scale = output_scale.powi(2) / input.scale[0].powi(2);
    SMatrix::from_fn(|i, j| {
        let tmp1: i32 = T::to_superset(&softm.buffer[(i, j)]);
        let tmp2: i32 = label.buffer[(i, j)].to_superset();
        let diff: i32 = tmp1 - tmp2;
        // T::from_superset(&(output_scale * diff / (input.scale[0].powi(2)))).unwrap()
        // i32::from_superset_unchecked(&(f32::from_subset(&diff) * scale))
        diff
    })
}
pub fn cross_entropy_loss<T: Trainable, const ROWS: usize, const COLS: usize>(
    input: &Tensor2D<T, ROWS, COLS, 1>,
    output_scale: f32,
    output_zero_point: T,
    label: &Tensor2D<T, ROWS, COLS, 1>,
) -> f32 {
    let softm = softmax_borrow(&input, [output_scale], [output_zero_point]);
    let label = label.dequantize();
    label
        .component_mul(&softm.dequantize().map(|el| logf(el)))
        .sum()
}
/// Computes the input tensor index corresponding to a convolution window.
///
/// Accounts for padding mode and stride.
///
/// # Parameters
/// - `view_rows`: Kernel height.
/// - `view_cols`: Kernel width.
/// - `focus`: Output position.
/// - `padding`: Padding mode (Same or Valid).
/// - `strides`: Convolution strides.
///
/// # Returns
/// Input tensor index as `(row, col)`.
pub fn get_input_index(
    view_rows: usize,
    view_cols: usize,
    focus: (usize, usize),
    padding: TensorViewPadding,
    strides: (usize, usize),
) -> (i32, i32) {
    match padding {
        TensorViewPadding::Same => {
            let shift = ((view_rows - 1) / 2, (view_cols - 1) / 2);
            (
                (strides.0 * focus.0) as i32 - (shift.0) as i32,
                (strides.1 * focus.1) as i32 - (shift.1) as i32,
            )
        }
        TensorViewPadding::Valid => ((strides.0 * focus.0) as i32, (strides.1 * focus.1) as i32),
    }
}
/// Determines whether a neuron output is inactive due to fused activation.
///
/// Checks clipping conditions for ReLU and ReLU6.
///
/// # Parameters
/// - `x`: Quantized activation value.
/// - `quantized_6`: Quantized representation of 6.
/// - `zero_point`: Quantization zero point.
/// - `activation`: Activation function.
///
/// # Returns
/// `true` if the value is cut off (inactive), otherwise `false`.
pub fn is_cut_off<T: Trainable>(
    x: T,
    quantized_6: T,
    zero_point: T,
    activation: &FusedActivation,
) -> bool {
    let val = x.saturating_sub(zero_point);
    !(match activation {
        FusedActivation::Relu => val > T::zero(),
        FusedActivation::Relu6 => val > T::zero() && val.saturating_add(zero_point) < quantized_6,
        _ => true,
    })
}
