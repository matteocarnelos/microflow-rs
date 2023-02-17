use core::fmt::{Debug, Display, Formatter};

use libm::roundf;
use nalgebra::{convert_unchecked, SMatrix, Scalar};
use simba::scalar::SupersetOf;

#[derive(Debug)]
pub struct QuantizedTensor<T, const R: usize, const C: usize> {
    pub(crate) buffer: SMatrix<T, R, C>,
    pub(crate) scale: f32,
    pub(crate) zero_point: T,
}

impl<T, const R: usize, const C: usize> QuantizedTensor<T, R, C>
where
    T: Scalar,
    i32: SupersetOf<T>,
    f32: SupersetOf<T>,
{
    pub fn new(buffer: SMatrix<T, R, C>, scale: f32, zero_point: T) -> Self {
        Self {
            buffer,
            scale,
            zero_point,
        }
    }

    pub fn quantize(input: SMatrix<f32, R, C>, scale: f32, zero_point: T) -> Self {
        Self {
            buffer: convert_unchecked(
                input.map(|f| roundf(f / scale + f32::from_subset(&zero_point))),
            ),
            scale,
            zero_point,
        }
    }

    pub fn dequantize(self) -> SMatrix<f32, R, C> {
        self.buffer
            .map(|q| self.scale * (f32::from_subset(&q) - f32::from_subset(&self.zero_point)))
    }
}

impl<T, const R: usize, const C: usize> Display for QuantizedTensor<T, R, C>
where
    T: Scalar + Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.buffer)
    }
}
