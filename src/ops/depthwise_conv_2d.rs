use core::array;

use libm::{fmaxf, fminf};
use nalgebra::SMatrix;

use crate::activation::FusedActivation;
use crate::buffer::{Buffer2D, Buffer4D};
use crate::tensor::{QuantizedTensor2D, QuantizedTensor4D};

// TODO: Optimize (constants + quantized op)
// TODO: Implement for `u8`
// TODO: According to Keras, for depthwise conv2d D1 must be = 1

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
    let shift = ((D5 - 1) / 2, (D6 - 1) / 2);
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

#[cfg(test)]
mod tests {
    use nalgebra::matrix;
    use super::*;

    const INPUT: Buffer4D<f32, 2, 2, 3, 2> = [
        matrix![
            [1., 2.], [3., 4.],  [5., 6.];
            [7., 8.], [9., 10.], [11., 12.]
        ],
        matrix![
            [13., 14.], [15., 16.], [17., 18.];
            [19., 20.], [21., 22.], [23., 24.]
        ]
    ];
    const WEIGHTS: Buffer4D<f32, 2, 2, 3, 2> = [
        matrix![
            [25., 26.], [27., 28.], [29., 30.];
            [31., 32.], [33., 34.], [35., 36.]
        ],
        matrix![
            [37., 38.], [39., 40.], [41., 42.];
            [43., 44.], [45., 46.], [47., 48.]
        ]
    ];
    const BIASES: Buffer2D<f32, 2, 1> = matrix![
        49.;
        50.
    ];
    const FUSED_ACTIVATION: FusedActivation = FusedActivation::NONE;
    const PADDING: DepthwiseConv2DPadding = DepthwiseConv2DPadding::SAME;
    const STRIDE: (usize, usize) = (1, 1);

    #[test]
    fn depthwise_conv_2d_layer() {

    }

    #[test]
    fn convolve_op() {
        let output: Buffer4D<f32, 2, 2, 3, 2> = convolve(
            INPUT,
            WEIGHTS,
            BIASES,
            FUSED_ACTIVATION,
            PADDING,
            STRIDE
        );
        assert_eq!(
            output,
            [
                matrix![
                    [709., 858.], [1199., 1422.], [901., 1050.];
                    [499., 574.], [786., 898.], [571., 646.]
                ],
                matrix![
                    [3013., 3258.], [4655., 5022.], [3205., 3450.];
                    [1651., 1774.], [2514., 2698.], [1723., 1846.]
                ],
            ]
        )
    }
}
