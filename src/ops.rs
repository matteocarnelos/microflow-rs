use libm::roundf;
use nalgebra::SMatrix;

use crate::activations::{relu, Activation};
use crate::tensor::QuantizedTensor;

pub fn fully_connected<const M: usize, const P: usize, const N: usize>(
    input: QuantizedTensor<i8, M, P>,
    weights: QuantizedTensor<i8, P, N>,
    biases: QuantizedTensor<i32, N, 1>,
    scale: f32,
    zero_point: i8,
    activation: Activation,
) -> QuantizedTensor<i8, M, N> {

    // TODO: Preprocess constant values
    let c = (
        zero_point as f32,
        biases.scale / scale * biases.buffer.add_scalar(-biases.zero_point).cast::<f32>(),
        input.scale * weights.scale / scale,
        (input.zero_point as i32 * weights.buffer.cast::<i32>().row_sum()).cast::<f32>(),
        (P as i32 * input.zero_point as i32 * weights.zero_point as i32) as f32,
    );

    let x = (
        (input.buffer.cast::<i32>() * weights.buffer.cast::<i32>()).cast::<f32>(),
        (weights.zero_point as i32 * input.buffer.cast::<i32>().column_sum()).cast::<f32>(),
    );

    let acc: SMatrix<i8, M, N> = SMatrix::from_fn(|i, j| {
        roundf(c.0 + c.1[j] + c.2 * (x.0[(i, j)] - x.1[i] - c.3[j] + c.4)) as i8
    });

    match activation {
        Activation::RELU => relu(acc, scale, zero_point),
        Activation::NONE => QuantizedTensor::new(acc, scale, zero_point),
    }
}
