use crate::quantize::quantize;
use core::cmp::{max, min};

pub enum ActivationType {
    NONE,
    RELU,
    RELU6,
}

pub fn relu(input: i8, zero_point: i8) -> i8 {
    max(input, zero_point)
}

pub fn relu6(input: i8, scale: f32, zero_point: i8) -> i8 {
    min(relu(input, zero_point), quantize(6., scale, zero_point))
}
