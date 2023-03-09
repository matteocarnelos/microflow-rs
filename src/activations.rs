use core::cmp::max;

use crate::tensor::QuantizedTensor;

pub enum Activation {
    NONE,
    RELU,
}

pub fn relu<const R: usize, const C: usize>(input: &mut QuantizedTensor<i8, R, C>) {
    let zero_point = input.zero_point;
    input.matrix.apply(|x| *x = max(*x, zero_point));
}
