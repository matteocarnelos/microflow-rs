use core::array;

use libm::{fmaxf, fminf};
use nalgebra::SMatrix;

use crate::activation::FusedActivation;
use crate::buffer::{Buffer2D, Buffer4D};
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
    const D2: usize,
    const D3: usize,
    const D4: usize,
    const D4_OR_1: usize,
    const D5: usize,
    const D6: usize,
    const D7: usize,
    const D8: usize,
>(
    input: Tensor4D<T, 1, D2, D3, D4_OR_1, D4_OR_1>,
    weights: Tensor4D<T, 1, D5, D6, D4, D4_OR_1>,
    biases: Tensor2D<i32, D4, 1>,
    output_scale: [f32; D4_OR_1],
    output_zero_point: [T; D4_OR_1],
    options: DepthwiseConv2DOptions,
) -> Tensor4D<T, 1, D7, D8, D4, D4_OR_1> {
    Tensor4D::quantize(
        convolve(
            input.dequantize(),
            weights.dequantize(),
            biases.dequantize(),
            options.fused_activation,
            options.padding,
            options.strides,
        ),
        output_scale,
        output_zero_point,
    )
}

pub fn convolve<
    const D2: usize,
    const D3: usize,
    const D4: usize,
    const D4_OR_1: usize,
    const D5: usize,
    const D6: usize,
    const D7: usize,
    const D8: usize,
>(
    input: Buffer4D<f32, 1, D2, D3, D4_OR_1>,
    weights: Buffer4D<f32, 1, D5, D6, D4>,
    biases: Buffer2D<f32, D4, 1>,
    fused_activation: FusedActivation,
    padding: DepthwiseConv2DPadding,
    strides: (usize, usize),
) -> Buffer4D<f32, 1, D7, D8, D4> {
    let shift = ((D5 - 1) / 2, (D6 - 1) / 2);
    [SMatrix::from_fn(|i, j| {
        array::from_fn(|c| {
            let view: SMatrix<f32, D5, D6> = SMatrix::from_fn(|m, n| match padding {
                DepthwiseConv2DPadding::SAME => {
                    let index = (
                        if let Some(x) = (strides.0 * i + m).checked_sub(shift.0) {
                            x
                        } else {
                            return 0.;
                        },
                        if let Some(x) = (strides.1 * j + n).checked_sub(shift.1) {
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
                    let x = input[0][(strides.0 * i + m, strides.1 * j + n)];
                    x.get(c).copied().unwrap_or(x[0])
                }
            });
            let y = view.dot(&weights[0].map(|a| a[c])) + biases[c];
            match fused_activation {
                FusedActivation::NONE => y,
                FusedActivation::RELU => fmaxf(y, 0.),
                FusedActivation::RELU6 => fminf(fmaxf(y, 0.), 6.),
            }
        })
    })]
}

#[cfg(test)]
mod tests {
    use nalgebra::matrix;

    use super::*;

    const INPUT: Tensor4D<i8, 1, 2, 3, 2, 2> = Tensor4D {
        buffer: [matrix![
            [1, 2], [3, 4],  [5, 6];
            [7, 8], [9, 10], [11, 12]
        ]],
        scale: [0.13, 0.14],
        zero_point: [15, 16],
    };
    const WEIGHTS: Tensor4D<i8, 1, 2, 3, 2, 2> = Tensor4D {
        buffer: [matrix![
            [17, 18], [19, 20], [21, 22];
            [23, 24], [25, 26], [27, 28]
        ]],
        scale: [0.29, 0.30],
        zero_point: [31, 32],
    };
    const BIASES: Tensor2D<i32, 2, 1> = Tensor2D {
        buffer: matrix![
            33;
            34
        ],
        scale: 0.35,
        zero_point: 36,
    };
    const OUTPUT_SCALE: [f32; 2] = [0.37, 0.38];
    const OUTPUT_ZERO_POINT: [i8; 2] = [39, 40];
    const OPTIONS: DepthwiseConv2DOptions = DepthwiseConv2DOptions {
        fused_activation: FusedActivation::NONE,
        padding: DepthwiseConv2DPadding::SAME,
        strides: (1, 1),
    };

    #[test]
    fn depthwise_conv_2d_layer() {
        let output: Tensor4D<i8, 1, 2, 3, 2, 2> = depthwise_conv_2d(
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
                    [73, 78], [93, 100], [73, 78];
                    [52, 55], [59, 63], [50, 53]
                ]],
                [0.37, 0.38],
                [39, 40]
            )
        );
    }
}
