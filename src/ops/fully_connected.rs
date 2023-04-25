use libm::roundf;
use nalgebra::SMatrix;

use crate::activations::{relu, relu6, ActivationType};
use crate::tensor::{Buffer4D, QuantizedTensor2D};

// TODO: Implement ops for u8

pub fn fully_connected<const D1: usize, const D2: usize, const D3: usize>(
    input: &QuantizedTensor2D<i8, D1, D2>,
    weights: QuantizedTensor2D<i8, D2, D3>,
    scale: f32,
    zero_point: i8,
    fused_activation: ActivationType,
    constants: (i8, SMatrix<f32, D3, 1>, f32, SMatrix<i32, 1, D3>, i32),
) -> QuantizedTensor2D<i8, D1, D3> {
    let x = (
        input.buffer.cast::<i32>() * weights.buffer.cast::<i32>(),
        weights.zero_point as i32 * input.buffer.cast::<i32>().column_sum(),
    );

    QuantizedTensor2D::new(
        SMatrix::from_fn(|i, j| {
            let y = roundf(
                constants.0 as f32
                    + constants.1[j]
                    + constants.2 * (x.0[(i, j)] - x.1[i] - constants.3[j] + constants.4) as f32,
            ) as i8;
            match fused_activation {
                ActivationType::NONE => y,
                ActivationType::RELU => relu(y, zero_point),
            }
        }),
        scale,
        zero_point,
    )
}
