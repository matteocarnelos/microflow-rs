use libm::roundf;
use nalgebra::SMatrix;

use crate::activations::{relu, Activation};
use crate::tensor::QuantizedTensor;

pub fn fully_connected<const M: usize, const P: usize, const N: usize>(
    input: QuantizedTensor<i8, M, P>,
    weights: QuantizedTensor<i8, P, N>,
    scale: f32,
    zero_point: i8,
    activation: Activation,
    constants: (f32, SMatrix<f32, N, 1>, f32, SMatrix<f32, 1, N>, f32),
) -> QuantizedTensor<i8, M, N> {
    let x = (
        (input.buffer.cast::<i32>() * weights.buffer.cast::<i32>()).cast::<f32>(),
        (weights.zero_point as i32 * input.buffer.cast::<i32>().column_sum()).cast::<f32>(),
    );

    let acc: SMatrix<i8, M, N> = SMatrix::from_fn(|i, j| {
        roundf(
            constants.0
                + constants.1[j]
                + constants.2 * (x.0[(i, j)] - x.1[i] - constants.3[j] + constants.4),
        ) as i8
    });

    match activation {
        Activation::RELU => relu(acc, scale, zero_point),
        Activation::NONE => QuantizedTensor::new(acc, scale, zero_point),
    }
}
