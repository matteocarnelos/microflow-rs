use libm::roundf;
use nalgebra::Scalar;
use num_traits::{One, Zero};
use simba::scalar::{ClosedAdd, ClosedMul, ClosedSub, SubsetOf, SupersetOf};

pub trait Quantized:
    Scalar + One + Zero + ClosedAdd + ClosedSub + ClosedMul + Copy + Ord + SubsetOf<i32> + SubsetOf<f32>
{
}

impl<
        T: Scalar
            + One
            + Zero
            + ClosedAdd
            + ClosedSub
            + ClosedMul
            + Copy
            + Ord
            + SubsetOf<i32>
            + SubsetOf<f32>,
    > Quantized for T
{
}

pub fn quantize<T: Quantized>(input: f32, scale: f32, zero_point: T) -> T {
    roundf(input / scale + f32::from_subset(&zero_point)).to_subset_unchecked()
}

pub fn dequantize<T: Quantized>(input: T, scale: f32, zero_point: T) -> f32 {
    scale * (f32::from_subset(&input) - f32::from_subset(&zero_point))
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALUE: f32 = 1.;
    const SCALE: f32 = 0.2;
    const ZERO_POINT: i8 = 3;
    const VALUE_QUANTIZED: i8 = 8;
    const VALUE_DEQUANTIZED: f32 = 1.;

    #[test]
    fn quantize_value() {
        assert_eq!(quantize(VALUE, SCALE, ZERO_POINT), VALUE_QUANTIZED);
    }

    #[test]
    fn dequantize_value() {
        assert_eq!(
            dequantize(VALUE_QUANTIZED, SCALE, ZERO_POINT),
            VALUE_DEQUANTIZED
        );
    }
}
