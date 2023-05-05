use core::array;

use libm::{fmaxf, fminf};
use nalgebra::SMatrix;

use crate::activation::FusedActivation;
use crate::buffer::{Buffer2D, Buffer4D};
use crate::tensor::{QuantizedTensor2D, QuantizedTensor4D};

// TODO: Optimize (constants + quantized op)
// TODO: Implement for `u8`

pub struct DepthwiseConv2DOptions {
    pub fused_activation: FusedActivation,
    pub padding: DepthwiseConv2DPadding,
    pub stride: (usize, usize),
}

pub enum DepthwiseConv2DPadding {
    SAME,
    VALID,
}

pub fn depthwise_conv_2d<
    const D1: usize,
    const D2: usize,
    const D3: usize,
    const D4: usize,
    const D4_OR_1: usize,
    const D5: usize,
    const D6: usize,
    const D7: usize,
    const D8: usize,
>(
    input: QuantizedTensor4D<i8, D1, D2, D3, D4_OR_1, D4_OR_1>,
    weights: QuantizedTensor4D<i8, D1, D5, D6, D4, D4_OR_1>,
    biases: QuantizedTensor2D<i32, D4, 1>,
    output_scale: [f32; D4_OR_1],
    output_zero_point: [i8; D4_OR_1],
    options: DepthwiseConv2DOptions,
) -> QuantizedTensor4D<i8, D1, D7, D8, D4, D4_OR_1> {
    QuantizedTensor4D::quantize(
        convolve(
            input.dequantize(),
            weights.dequantize(),
            biases.dequantize(),
            options.fused_activation,
            options.padding,
            options.stride,
        ),
        output_scale,
        output_zero_point,
    )
}

pub fn convolve<
    const D1: usize,
    const D2: usize,
    const D3: usize,
    const D4: usize,
    const D4_OR_1: usize,
    const D5: usize,
    const D6: usize,
    const D7: usize,
    const D8: usize,
>(
    input: Buffer4D<f32, D1, D2, D3, D4_OR_1>,
    weights: Buffer4D<f32, D1, D5, D6, D4>,
    biases: Buffer2D<f32, D4, 1>,
    fused_activation: FusedActivation,
    padding: DepthwiseConv2DPadding,
    stride: (usize, usize),
) -> Buffer4D<f32, D1, D7, D8, D4> {
    let shift = (D5 / 2, D6 / 2);
    array::from_fn(|b| {
        SMatrix::from_fn(|i, j| {
            array::from_fn(|c| {
                let view: SMatrix<f32, D5, D6> = SMatrix::from_fn(|m, n| match padding {
                    DepthwiseConv2DPadding::SAME => {
                        let index = (
                            if let Some(x) = (stride.0 * i + m).checked_sub(shift.0) {
                                x
                            } else {
                                return 0.;
                            },
                            if let Some(x) = (stride.1 * j + n).checked_sub(shift.1) {
                                x
                            } else {
                                return 0.;
                            },
                        );
                        if let Some(x) = input[b].get(index) {
                            x.get(c).copied().unwrap_or(x[0])
                        } else {
                            0.
                        }
                    }
                    DepthwiseConv2DPadding::VALID => {
                        // TODO: Fallback to input[0]
                        let x = input[b][(stride.0 * i + m, stride.1 * j + n)];
                        x.get(c).copied().unwrap_or(x[0])
                    }
                });
                let y = view.dot(&weights[b].map(|a| a[c])) + biases[c];
                match fused_activation {
                    FusedActivation::NONE => y,
                    FusedActivation::RELU => fmaxf(y, 0.),
                    FusedActivation::RELU6 => fminf(fmaxf(y, 0.), 6.),
                }
            })
        })
    })
}

// TODO: Write proper tests
// #[cfg(test)]
// mod tests {
//     use nalgebra::matrix;
//
//     use super::*;
//
//     extern crate std;
//
//     const INPUT_BUFFER: Buffer4D<f32, 2, 2, 3, 2> = [
//         matrix![
//             [1., 11.], [2., 12.], [3., 13.];
//             [4., 14.], [5., 15.], [6., 16.]
//         ],
//         matrix![
//             [17., 18.], [19., 20.], [21., 22.];
//             [23., 24.], [25., 26.], [27., 28.]
//         ],
//     ];
//     const WEIGHTS_BUFFER: Buffer4D<f32, 2, 5, 3, 2> = [
//         matrix![
//             [7., 29.], [8., 30.], [9., 31.];
//             [39., 40.], [41., 42.], [43., 44.];
//             [51., 52.], [53., 54.], [55., 56.];
//             [63., 64.], [65., 66.], [67., 68.];
//             [69., 70.], [71., 72.], [73., 74.]
//         ],
//         matrix![
//             [7., 29.], [8., 30.], [9., 31.];
//             [39., 40.], [41., 42.], [43., 44.];
//             [51., 52.], [53., 54.], [55., 56.];
//             [63., 64.], [65., 66.], [67., 68.];
//             [69., 70.], [71., 72.], [73., 74.]
//         ],
//     ];
//     const BIASES: Buffer2D<f32, 2, 1> = matrix![10.; 38.];
//
//     #[test]
//     fn depthwise_conv_2d_layer() {
//         let tensor: Buffer4D<f32, 2, 2, 3, 2> = convolve(
//             INPUT_BUFFER,
//             WEIGHTS_BUFFER,
//             BIASES,
//             Activation::NONE,
//             DepthwiseConv2DPadding::SAME,
//             (1, 1),
//         );
//         for m in tensor {
//             std::println!("{}", m.map(|a| std::format!("{:?}", a)));
//         }
//     }
// }
