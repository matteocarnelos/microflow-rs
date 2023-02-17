use core::cmp::max;

use nalgebra::SMatrix;

use crate::tensor::QuantizedTensor;

pub enum Activation {
    NONE,
    RELU,
}

pub fn relu<const R: usize, const C: usize>(
    input: SMatrix<i8, R, C>,
    scale: f32,
    zero_point: i8,
) -> QuantizedTensor<i8, R, C> {
    QuantizedTensor::new(input.map(|x| max(x, zero_point)), scale, zero_point)
}
