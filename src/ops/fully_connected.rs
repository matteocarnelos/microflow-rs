use libm::roundf;

use crate::activation::{relu, relu6, FusedActivation};
use crate::buffer::Buffer2D;
use crate::tensor::QuantizedTensor2D;

// TODO: Implement for `u8`

pub struct FullyConnectedOptions {
    pub fused_activation: FusedActivation,
}

pub fn fully_connected<const D1: usize, const D2: usize, const D3: usize>(
    input: &QuantizedTensor2D<i8, D1, D2>,
    weights: QuantizedTensor2D<i8, D2, D3>,
    output_scale: f32,
    output_zero_point: i8,
    options: FullyConnectedOptions,
    constants: (Buffer2D<f32, D3, 1>, f32, Buffer2D<i32, 1, D3>, i32),
) -> QuantizedTensor2D<i8, D1, D3> {
    let x = (
        input.buffer.cast::<i32>() * weights.buffer.cast::<i32>(),
        weights.zero_point as i32 * input.buffer.cast::<i32>().column_sum(),
    );
    QuantizedTensor2D::new(
        Buffer2D::from_fn(|i, j| {
            let y = roundf(
                output_zero_point as f32
                    + constants.0[j]
                    + constants.1 * (x.0[(i, j)] - x.1[i] - constants.2[j] + constants.3) as f32,
            ) as i8;
            match options.fused_activation {
                FusedActivation::NONE => y,
                FusedActivation::RELU => relu(y, output_zero_point),
                FusedActivation::RELU6 => relu6(y, output_scale, output_zero_point),
            }
        }),
        output_scale,
        output_zero_point,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::matrix;

    const INPUT_TENSOR: QuantizedTensor2D<i8, 2, 3> = QuantizedTensor2D {
        buffer: matrix![
            1, 2, 3;
            4, 5, 6
        ],
        scale: 0.7,
        zero_point: 8,
    };
    const WEIGHTS_TENSOR: QuantizedTensor2D<i8, 3, 4> = QuantizedTensor2D {
        buffer: matrix![
            9, 10, 11, 12;
            13, 14, 15, 16;
            17, 18, 19, 20
        ],
        scale: 0.21,
        zero_point: 22,
    };
    const _BIASES_TENSOR: QuantizedTensor2D<i8, 4, 1> = QuantizedTensor2D {
        buffer: matrix![
            23; 24; 25; 26
        ],
        scale: 0.27,
        zero_point: 28,
    };
    const OUTPUT_SCALE: f32 = 0.29;
    const OUTPUT_ZERO_POINT: i8 = 30;
    const OPTIONS: FullyConnectedOptions = FullyConnectedOptions {
        fused_activation: FusedActivation::RELU,
    };
    const CONSTANTS: (Buffer2D<f32, 4, 1>, f32, Buffer2D<i32, 1, 4>, i32) = (
        matrix![-4.655_172_3; -3.724_138; -2.793_103_5; -1.862_069],
        0.506_896_56,
        matrix![312, 336, 360, 384],
        528,
    );

    #[test]
    fn fully_connected_layer() {
        assert_eq!(
            fully_connected(
                &INPUT_TENSOR,
                WEIGHTS_TENSOR,
                OUTPUT_SCALE,
                OUTPUT_ZERO_POINT,
                OPTIONS,
                CONSTANTS
            ),
            QuantizedTensor2D::new(
                matrix![
                    112, 103, 95, 87;
                    70, 67, 63, 60
                ],
                0.29,
                30
            )
        )
    }
}
