use core::cmp::max;

pub enum ActivationType {
    NONE,
    RELU,
}

pub fn relu(input: i8, zero_point: i8) -> i8 {
    max(input, zero_point)
}
