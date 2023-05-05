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
    constants: (i8, Buffer2D<f32, D3, 1>, f32, Buffer2D<i32, 1, D3>, i32),
) -> QuantizedTensor2D<i8, D1, D3> {
    let x = (
        input.buffer.cast::<i32>() * weights.buffer.cast::<i32>(),
        weights.zero_point as i32 * input.buffer.cast::<i32>().column_sum(),
    );
    QuantizedTensor2D::new(
        Buffer2D::from_fn(|i, j| {
            let y = roundf(
                constants.0 as f32
                    + constants.1[j]
                    + constants.2 * (x.0[(i, j)] - x.1[i] - constants.3[j] + constants.4) as f32,
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

// TODO: Unit tests
