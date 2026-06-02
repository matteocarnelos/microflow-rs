use crate::buffer::{Buffer2D, Buffer4D};
use crate::quantize::Quantized;
use crate::tensor::{Tensor2D, Tensor4D};
use libm::expf;
use simba::scalar::SupersetOf;

pub fn logistic_2d<T: Quantized, const ROWS: usize, const COLS: usize>(
    input: Tensor2D<T, ROWS, COLS, 1>,
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
) -> Tensor2D<T, ROWS, COLS, 1> {
    let output = Buffer2D::from_fn(|i, j| {
        let x = input.scale[0] * (f32::from_subset(&input.buffer[(i, j)]) - f32::from_subset(&input.zero_point[0]));
        let y = 1.0 / (1.0 + expf(-x));
        crate::quantize::quantize(y, output_scale[0], output_zero_point[0])
    });
    Tensor2D::new(output, output_scale, output_zero_point)
}

pub fn logistic_4d<
    T: Quantized,
    const BATCHES: usize,
    const ROWS: usize,
    const COLS: usize,
    const CHANS: usize,
>(
    input: Tensor4D<T, BATCHES, ROWS, COLS, CHANS, 1>,
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
) -> Tensor4D<T, BATCHES, ROWS, COLS, CHANS, 1> {
    let output: Buffer4D<T, BATCHES, ROWS, COLS, CHANS> = core::array::from_fn(|b| {
        Buffer2D::from_fn(|i, j| {
            core::array::from_fn(|c| {
                let x = input.scale[0]
                    * (f32::from_subset(&input.buffer[b][(i, j)][c])
                        - f32::from_subset(&input.zero_point[0]));
                let y = 1.0 / (1.0 + expf(-x));
                crate::quantize::quantize(y, output_scale[0], output_zero_point[0])
            })
        })
    });
    Tensor4D::new(output, output_scale, output_zero_point)
}

#[cfg(test)]
mod tests {
    use nalgebra::matrix;

    use super::*;

    #[test]
    fn logistic_2d_layer() {
        let input = Tensor2D::new(matrix![-2, 2], [0.5], [0]);
        let output: Tensor2D<i8, 1, 2, 1> = logistic_2d(input, [1.0 / 128.0], [0]);
        assert_eq!(output.buffer[(0, 0)] < output.buffer[(0, 1)], true);
    }
}
