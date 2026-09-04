use libm::roundf;
use simba::scalar::SupersetOf;

use crate::activation::{relu, relu6, FusedActivation};
use crate::buffer::Buffer2D;
use crate::quantize::Quantized;
use crate::tensor::Tensor2D;

pub struct FullyConnectedOptions {
    pub fused_activation: FusedActivation,
}

/// Performs the FullyConnected operation with per-channel weight quantization.
/// Returns a 2-dimensional output tensor containing the result of the operation.
///
/// `WEIGHTS_QUANTS` scales and zero points are attached to the weights, one per
/// output unit (TFLite converts Keras `Dense` layers per-channel by default).
/// The requant constants are pre-processed by the macro and passed per output
/// column `j`:
/// - `constants.0[j]`: the bias term, already scaled to the output scale;
/// - `constants.1[j]`: the multiplier `scale_x * scale_w[j] / scale_out`;
/// - `constants.2[j]`: the zero-point correction
///   `zp_x * (sum_i w[i][j] - INPUT_COLS * zp_w[j])`.
///
/// A bias dropped by the converter (an all-zero one) is passed by the macro as
/// an explicit zero constant vector.
///
/// # Arguments
/// * `input` - The 2-dimensional input tensor
/// * `weights` - The 2-dimensional tensor representing the weights of the operator
/// * `output_scale` - The scale of the resulting output tensor
/// * `output_zero_point` - The zero point of the resulting output tensor
/// * `options` - Operator's options as a [`FullyConnectedOptions`] struct
/// * `constants` - Constant values coming from the pre-processing phase
///
pub fn fully_connected<
    T: Quantized,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const WEIGHTS_COLS: usize,
    const WEIGHTS_QUANTS: usize,
>(
    input: Tensor2D<T, INPUT_ROWS, INPUT_COLS, 1>,
    weights: &Tensor2D<T, INPUT_COLS, WEIGHTS_COLS, WEIGHTS_QUANTS>,
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
    options: FullyConnectedOptions,
    constants: (
        Buffer2D<f32, WEIGHTS_COLS, 1>,
        Buffer2D<f32, WEIGHTS_COLS, 1>,
        Buffer2D<i32, 1, WEIGHTS_COLS>,
    ),
) -> Tensor2D<T, INPUT_ROWS, WEIGHTS_COLS, 1> {
    // Perform the dot product between the input and the weights
    let dots: Buffer2D<i32, INPUT_ROWS, WEIGHTS_COLS> = Buffer2D::from_fn(|i, j| {
        input
            .buffer
            .row(i)
            .iter()
            .zip(weights.buffer.column(j).iter())
            .fold(0i32, |acc, (i, w)| {
                acc + i32::from_subset(i) * i32::from_subset(w)
            })
    });
    // Perform the row-sum of the input (zero-point correction term)
    let row_sums: Buffer2D<i32, INPUT_ROWS, 1> = Buffer2D::from_fn(|i, _| {
        input
            .buffer
            .row(i)
            .fold(0i32, |acc, e| acc + i32::from_subset(&e))
    });
    // Combine the constant values and the variants to obtain the output
    let output = Buffer2D::from_fn(|i, j| {
        let weights_zero_point = i32::from_subset(
            &weights
                .zero_point
                .get(j)
                .copied()
                .unwrap_or(weights.zero_point[0]),
        );
        // Accumulator of (x - zp_x) . (w - zp_w_j) in input*weights units
        let accumulator = dots[(i, j)] - weights_zero_point * row_sums[i] - constants.2[j];
        let y = T::from_superset_unchecked(&roundf(
            f32::from_subset(&output_zero_point[0])
                + constants.0[j]
                + constants.1[j] * f32::from_subset(&accumulator),
        ));
        // Apply the fused activation function (if any)
        match options.fused_activation {
            FusedActivation::None => y,
            FusedActivation::Relu => relu(y, output_zero_point[0]),
            FusedActivation::Relu6 => relu6(y, output_scale[0], output_zero_point[0]),
        }
    });
    Tensor2D::new(output, output_scale, output_zero_point)
}

#[cfg(test)]
mod tests {
    use nalgebra::{matrix, SMatrix};

    use super::*;

    const INPUT: Tensor2D<i8, 2, 3, 1> = Tensor2D {
        buffer: matrix![
            1, 2, 3;
            4, 5, 6
        ],
        scale: [0.7],
        zero_point: [8],
    };
    const WEIGHTS: Tensor2D<i8, 3, 4, 1> = Tensor2D {
        buffer: matrix![
            9,  10, 11, 12;
            13, 14, 15, 16;
            17, 18, 19, 20
        ],
        scale: [0.21],
        zero_point: [22],
    };
    // Same weights, per-channel quantization with identical per-channel
    // parameters: the results must match the per-tensor case (a plumbing test
    // for WEIGHTS_QUANTS > 1, spec §3.3).
    const WEIGHTS_PER_CHANNEL: Tensor2D<i8, 3, 4, 4> = Tensor2D {
        buffer: matrix![
            9,  10, 11, 12;
            13, 14, 15, 16;
            17, 18, 19, 20
        ],
        scale: [0.21, 0.21, 0.21, 0.21],
        zero_point: [22, 22, 22, 22],
    };
    const OUTPUT_SCALE: [f32; 1] = [0.29];
    const OUTPUT_ZERO_POINT: [i8; 1] = [30];
    const OPTIONS: FullyConnectedOptions = FullyConnectedOptions {
        fused_activation: FusedActivation::Relu,
    };
    // Zero-point correction per column: zp_x * (col_sum - INPUT_COLS * zp_w)
    // = 8 * (39 - 3*22) = -216 for column 0, etc. (the old per-tensor split
    // was c2 = zp_x * col_sum = [312, 336, 360, 384] and c3 = 528).
    const CONSTANTS: (SMatrix<f32, 4, 1>, SMatrix<f32, 4, 1>, SMatrix<i32, 1, 4>) = (
        matrix![-4.655_172_3; -3.724_138; -2.793_103_5; -1.862_069],
        matrix![0.506_896_56; 0.506_896_56; 0.506_896_56; 0.506_896_56],
        matrix![-216, -192, -168, -144],
    );
    const OUTPUT: Tensor2D<i8, 2, 4, 1> = Tensor2D {
        buffer: matrix![
            112, 103, 95, 87;
            70,  67,  63, 60
        ],
        scale: [0.29],
        zero_point: [30],
    };

    #[test]
    fn fully_connected_layer() {
        assert_eq!(
            fully_connected(
                INPUT,
                &WEIGHTS,
                OUTPUT_SCALE,
                OUTPUT_ZERO_POINT,
                OPTIONS,
                CONSTANTS
            ),
            OUTPUT
        )
    }

    #[test]
    fn fully_connected_layer_per_channel() {
        assert_eq!(
            fully_connected(
                INPUT,
                &WEIGHTS_PER_CHANNEL,
                OUTPUT_SCALE,
                OUTPUT_ZERO_POINT,
                OPTIONS,
                CONSTANTS
            ),
            OUTPUT
        )
    }
}
