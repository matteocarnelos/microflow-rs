use core::fmt::Debug;

use libm::roundf;
use nalgebra::{convert_unchecked, SMatrix, Scalar};
use simba::scalar::SupersetOf;

#[derive(Debug)]
pub struct QuantizedTensor<T, const R: usize, const C: usize> {
    pub matrix: SMatrix<T, R, C>,
    pub scale: f32,
    pub zero_point: T,
}

impl<T, const R: usize, const C: usize> QuantizedTensor<T, R, C>
where
    T: Scalar,
    i32: SupersetOf<T>,
    f32: SupersetOf<T>,
{
    pub fn new(matrix: SMatrix<T, R, C>, scale: f32, zero_point: T) -> Self {
        Self {
            matrix,
            scale,
            zero_point,
        }
    }

    pub fn quantize(input: SMatrix<f32, R, C>, scale: f32, zero_point: T) -> Self {
        Self {
            matrix: convert_unchecked(
                input.map(|f| roundf(f / scale + f32::from_subset(&zero_point))),
            ),
            scale,
            zero_point,
        }
    }

    pub fn dequantize(self) -> SMatrix<f32, R, C> {
        self.matrix
            .map(|q| self.scale * (f32::from_subset(&q) - f32::from_subset(&self.zero_point)))
    }
}
