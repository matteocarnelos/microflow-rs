use crate::quantize::{quantize, Quantized};
use core::cmp::{max, min};
use libm::expf;

pub enum FusedActivation {
    None,
    Relu,
    Relu6,
}

pub fn relu<T: Quantized>(input: T, zero_point: T) -> T {
    max(input, zero_point)
}

pub fn relu6<T: Quantized>(input: T, scale: f32, zero_point: T) -> T {
    min(relu(input, zero_point), quantize(6., scale, zero_point))
}

pub fn softmax<T: Quantized>(input: f32, sum: f32, scale: f32, zero_point: T) -> T {
    quantize(expf(input) / sum, scale, zero_point)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCALE: f32 = 0.1;
    const ZERO_POINT: i8 = 2;

    const RELU_INACTIVE_INPUT: i8 = 1;
    const RELU_ACTIVE_INPUT: i8 = 3;

    const RELU6_SATURATED_INPUT: i8 = 63;
    const RELU6_SATURATION_POINT: i8 = 62;

    const SOFTMAX_INPUT_1: f32 = 1.;
    const SOFTMAX_INPUT_2: f32 = 2.;
    const SOFTMAX_INPUT_3: f32 = 3.;
    const SOFTMAX_SUM: f32 = 30.192_875;
    const SOFTMAX_OUTPUT_1: i8 = 3;
    const SOFTMAX_TOTAL_PROBABILITY: i8 = 16;

    #[test]
    fn relu_inactive() {
        assert_eq!(relu(RELU_INACTIVE_INPUT, ZERO_POINT), ZERO_POINT);
    }

    #[test]
    fn relu_active() {
        assert_eq!(relu(RELU_ACTIVE_INPUT, ZERO_POINT), RELU_ACTIVE_INPUT);
    }

    #[test]
    fn relu6_saturated() {
        assert_eq!(
            relu6(RELU6_SATURATED_INPUT, SCALE, ZERO_POINT),
            RELU6_SATURATION_POINT
        );
    }

    #[test]
    fn softmax_active() {
        assert_eq!(
            softmax(SOFTMAX_INPUT_1, SOFTMAX_SUM, SCALE, ZERO_POINT),
            SOFTMAX_OUTPUT_1
        );
    }

    #[test]
    fn softmax_total() {
        let total = softmax(SOFTMAX_INPUT_1, SOFTMAX_SUM, SCALE, ZERO_POINT)
            + softmax(SOFTMAX_INPUT_2, SOFTMAX_SUM, SCALE, ZERO_POINT)
            + softmax(SOFTMAX_INPUT_3, SOFTMAX_SUM, SCALE, ZERO_POINT);
        assert_eq!(total, SOFTMAX_TOTAL_PROBABILITY);
    }
}
