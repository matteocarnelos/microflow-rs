use libm::roundf;
use nalgebra::SMatrix;

use crate::activations::{relu, ActivationType};
use crate::tensor::QuantizedTensor2D;

pub fn fully_connected<const M: usize, const P: usize, const N: usize>(
    input: &QuantizedTensor2D<i8, M, P>,
    weights: QuantizedTensor2D<i8, P, N>,
    scale: f32,
    zero_point: i8,
    fused_activation: ActivationType,
    constants: (i8, SMatrix<f32, N, 1>, f32, SMatrix<i32, 1, N>, i32),
) -> QuantizedTensor2D<i8, M, N> {
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
