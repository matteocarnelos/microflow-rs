use crate::{
    activation::FusedActivation,
    buffer::Buffer4D,
    quantize::{quantize, Trainable},
    tensor::{Tensor4D, TensorViewPadding},
    update_layer::{get_input_index, is_cut_off},
};
use core::array;
use nalgebra::SMatrix;
/// Computes the input gradient for a quantized average pooling layer.
///
/// This function propagates gradients from the output of an average pooling
/// operation back to the corresponding input positions. Each output gradient
/// value is distributed evenly across the receptive field defined by the
/// pooling filter size, stride, and padding configuration.
///
/// If a fused activation function (such as ReLU or ReLU6) was applied during
/// the forward pass, gradients are not propagated for output values that were
/// clipped or inactive.
///
/// # Type Parameters
/// - `T`: Quantized numeric type implementing the `Trainable` trait.
/// - `INPUT_ROWS`, `INPUT_COLS`: Spatial dimensions of the input tensor.
/// - `OUTPUT_ROWS`, `OUTPUT_COLS`: Spatial dimensions of the output tensor.
/// - `INPUT_CHANS`: Number of input/output channels.
/// - `FILTER_ROWS`, `FILTER_COLS`: Dimensions of the average pooling filter.
///
/// # Parameters
/// - `input`: Reference to the original input tensor (used for shape and quantization context).
/// - `outputs`: Output tensor produced during the forward average pooling pass.
/// - `output_grad`: Gradient of the loss with respect to the pooling output.
/// - `filter_shape`: Shape of the pooling filter.
/// - `activation`: Fused activation function applied after pooling.
/// - `strides`: Vertical and horizontal stride of the pooling operation.
/// - `padding`: Padding mode (`Same` or `Valid`).
///
/// # Returns
/// A `Buffer4D<i32, ...>` containing accumulated gradients with respect to
/// each input element.
pub fn gradient_average_pool<
    T: Trainable,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const FILTER_ROWS: usize,
    const FILTER_COLS: usize,
>(
    input: &Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    outputs: Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, INPUT_CHANS, 1>,
    output_grad: Buffer4D<i32, 1, OUTPUT_ROWS, OUTPUT_COLS, INPUT_CHANS>,
    _filter_shape: (nalgebra::Const<FILTER_ROWS>, nalgebra::Const<FILTER_COLS>),
    activation: FusedActivation,
    strides: (usize, usize),
    padding: TensorViewPadding,
) -> Buffer4D<i32, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> {
    let mut accum: Buffer4D<i32, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS> =
        array::from_fn(|_| SMatrix::from_fn(|_, _| array::from_fn(|_| 0i32)));
    let quantized_6 = quantize(6f32, outputs.scale[0], outputs.zero_point[0]);
    for output_row in 0..OUTPUT_ROWS {
        for output_col in 0..OUTPUT_COLS {
            let coord = get_input_index(
                FILTER_ROWS,
                FILTER_COLS,
                (output_row, output_col),
                padding,
                strides,
            );
            for output_batch in 0..INPUT_CHANS {
                let val = outputs.buffer[0][(output_row, output_col)][output_batch]
                    .saturating_sub(outputs.zero_point[0]);
                if is_cut_off(val, quantized_6, outputs.zero_point[0], &activation) {
                    continue;
                }
                for filter_row in 0..FILTER_ROWS {
                    if (coord.0 + filter_row as i32) < 0
                        || (coord.0 as usize + filter_row) >= INPUT_ROWS
                    {
                        continue;
                    }
                    for filter_col in 0..FILTER_COLS {
                        if (coord.1 + filter_col as i32) < 0
                            || (coord.1 as usize + filter_col) >= INPUT_COLS
                        {
                            continue;
                        }
                        let cur_coord = (
                            (coord.0 + filter_row as i32) as usize,
                            (coord.1 + filter_col as i32) as usize,
                        );
                        accum[0][cur_coord][output_batch] +=
                            output_grad[0][(output_row, output_col)][output_batch];
                    }
                }
            }
        }
    }
    accum
}
