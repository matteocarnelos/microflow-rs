use core::array;

use libm::roundf;
use simba::scalar::SupersetOf;

use crate::activation::{relu, relu6, FusedActivation};
use crate::buffer::Buffer2D;
use crate::quantize::Quantized;
use crate::tensor::{Tensor4D, View, ViewPadding};

pub struct DepthwiseConv2DOptions {
    pub fused_activation: FusedActivation,
    pub view_padding: ViewPadding,
    pub strides: (usize, usize),
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
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
    options: DepthwiseConv2DOptions,
    constants: (
        Buffer2D<f32, WEIGHTS_CHANS, 1>,
        Buffer2D<f32, WEIGHTS_QUANTS, 1>,
    ),
) -> Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, WEIGHTS_CHANS, 1> {
    let output = [Buffer2D::from_fn(|i, j| {
        let view: View<T, WEIGHTS_ROWS, WEIGHTS_COLS, INPUT_CHANS> =
            input.view((i, j), 0, options.view_padding, options.strides);
        array::from_fn(|c| {
            let input_zero_point = i32::from_subset(&input.zero_point[0]);
            let weights_zero_point = i32::from_subset(
                &weights
                    .zero_point
                    .get(c)
                    .copied()
                    .unwrap_or(weights.zero_point[0]),
            );
            let x = (
                view.buffer.zip_fold(&weights.buffer[0], 0i32, |acc, v, w| {
                    acc + i32::from_subset(&v.get(c).copied().unwrap_or(v[0]))
                        * i32::from_subset(&w[c])
                }),
                view.buffer.fold(0i32, |acc, a| {
                    acc + i32::from_subset(&a.get(c).copied().unwrap_or(a[0]))
                }) * weights_zero_point,
            );
            let constants = (
                constants.0,
                constants.1,
                input_zero_point
                    * weights.buffer[0].zip_fold(&view.mask, 0i32, |acc, w, m| {
                        if m {
                            acc + i32::from_subset(&w[c])
                        } else {
                            acc
                        }
                    }),
                view.len as i32 * input_zero_point * weights_zero_point,
            );
            let y = T::from_superset_unchecked(&roundf(
                f32::from_subset(&output_zero_point[0])
                    + constants.0[c]
                    + constants.1.get(c).copied().unwrap_or(constants.1[0])
                        * f32::from_subset(&(x.0 - x.1 - constants.2 + constants.3)),
            ));
            match options.fused_activation {
                FusedActivation::None => y,
                FusedActivation::Relu => relu(y, output_zero_point[0]),
                FusedActivation::Relu6 => relu6(y, output_scale[0], output_zero_point[0]),
            }
        })
    })];
    Tensor4D::new(output, output_scale, output_zero_point)
}

#[cfg(test)]
mod tests {
    use nalgebra::matrix;

    use crate::tensor::Tensor2D;

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
    const _BIASES: Tensor2D<i32, 2, 1, 2> = Tensor2D {
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
        fused_activation: FusedActivation::None,
        view_padding: ViewPadding::Same,
        strides: (1, 1),
    };
    const CONSTANTS: (Buffer2D<f32, 2, 1>, Buffer2D<f32, 2, 1>) = (
        matrix![-3.567_567_6; -3.675_675_7],
        matrix![0.094_864_86; 0.098_378_378],
    );
    const OUTPUT: Tensor4D<i8, 1, 2, 3, 2, 1> = Tensor4D {
        buffer: [matrix![
            [66, 63], [82, 78], [65, 62];
            [47, 45], [52, 49], [44, 42]
        ]],
        scale: [0.37],
        zero_point: [38],
    };

    #[test]
    fn depthwise_conv_2d_layer() {
        assert_eq!(
            depthwise_conv_2d(
                INPUT,
                WEIGHTS,
                OUTPUT_SCALE,
                OUTPUT_ZERO_POINT,
                OPTIONS,
                CONSTANTS,
            ),
            OUTPUT
        );
    }
}
