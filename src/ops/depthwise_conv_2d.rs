use core::array;

use libm::{fmaxf, fminf};
use nalgebra::SMatrix;

use crate::activation::FusedActivation;
use crate::quantize::Quantized;
use crate::tensor::{Tensor2D, Tensor4D};

// TODO: Optimize (constants + quantized op)

pub struct DepthwiseConv2DOptions {
    pub fused_activation: FusedActivation,
    pub padding: DepthwiseConv2DPadding,
    pub strides: (usize, usize),
}

pub enum DepthwiseConv2DPadding {
    SAME,
    VALID,
}

pub fn depthwise_conv_2d<
    T: Quantized,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const WEIGHTS_ROWS: usize,
    const WEIGHTS_COLS: usize,
    const WEIGHTS_CHANS: usize,
    const WEIGHTS_QUANTS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
>(
    input: Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    weights: Tensor4D<T, 1, WEIGHTS_ROWS, WEIGHTS_COLS, WEIGHTS_CHANS, WEIGHTS_QUANTS>,
    biases: Tensor2D<i32, WEIGHTS_CHANS, 1, WEIGHTS_QUANTS>,
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
    options: DepthwiseConv2DOptions,
) -> Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, WEIGHTS_CHANS, 1> {
    let input = input.dequantize();
    let weights = weights.dequantize();
    let biases = biases.dequantize();
    let output = [SMatrix::from_fn(|i, j| {
        array::from_fn(|c| {
            let view: SMatrix<f32, WEIGHTS_ROWS, WEIGHTS_COLS> =
                SMatrix::from_fn(|m, n| match options.padding {
                    DepthwiseConv2DPadding::SAME => {
                        let shift = ((WEIGHTS_ROWS - 1) / 2, (WEIGHTS_COLS - 1) / 2);
                        let index = (
                            if let Some(x) = (options.strides.0 * i + m).checked_sub(shift.0) {
                                x
                            } else {
                                return 0.;
                            },
                            if let Some(x) = (options.strides.1 * j + n).checked_sub(shift.1) {
                                x
                            } else {
                                return 0.;
                            },
                        );
                        if let Some(x) = input[0].get(index) {
                            x.get(c).copied().unwrap_or(x[0])
                        } else {
                            0.
                        }
                    }
                    DepthwiseConv2DPadding::VALID => {
                        let x = input[0][(options.strides.0 * i + m, options.strides.1 * j + n)];
                        x.get(c).copied().unwrap_or(x[0])
                    }
                });
            let y = view.zip_fold(&weights[0], 0f32, |acc, e, a| acc + e * a[c]) + biases[c];
            match options.fused_activation {
                FusedActivation::NONE => y,
                FusedActivation::RELU => fmaxf(y, 0.),
                FusedActivation::RELU6 => fminf(fmaxf(y, 0.), 6.),
            }
        })
    })];
    Tensor4D::quantize(output, output_scale, output_zero_point)
}

#[cfg(test)]
mod tests {
    use nalgebra::matrix;

    use super::*;

    const INPUT: Tensor4D<i8, 1, 2, 3, 2, 1> = Tensor4D {
        buffer: [matrix![
            [1, 2], [3, 4],  [5, 6];
            [7, 8], [9, 10], [11, 12]
        ]],
        scale: [0.13],
        zero_point: [14],
    };
    const WEIGHTS: Tensor4D<i8, 1, 2, 3, 2, 2> = Tensor4D {
        buffer: [matrix![
            [15, 16], [17, 18], [19, 20];
            [21, 22], [23, 24], [25, 26]
        ]],
        scale: [0.27, 0.28],
        zero_point: [29, 30],
    };
    const BIASES: Tensor2D<i32, 2, 1, 2> = Tensor2D {
        buffer: matrix![
            31;
            32
        ],
        scale: [0.33, 0.34],
        zero_point: [35, 36],
    };
    const OUTPUT_SCALE: [f32; 1] = [0.37];
    const OUTPUT_ZERO_POINT: [i8; 1] = [38];
    const OPTIONS: DepthwiseConv2DOptions = DepthwiseConv2DOptions {
        fused_activation: FusedActivation::NONE,
        padding: DepthwiseConv2DPadding::SAME,
        strides: (1, 1),
    };

    #[test]
    fn depthwise_conv_2d_layer() {
        let output: Tensor4D<i8, 1, 2, 3, 2, 1> = depthwise_conv_2d(
            INPUT,
            WEIGHTS,
            BIASES,
            OUTPUT_SCALE,
            OUTPUT_ZERO_POINT,
            OPTIONS,
        );
        assert_eq!(
            output,
            Tensor4D::new(
                [matrix![
                    [66, 63], [82, 78], [65, 62];
                    [47, 45], [52, 49], [44, 42]
                ]],
                [0.37],
                [38]
            )
        );
    }
}
