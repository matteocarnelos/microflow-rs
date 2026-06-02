use crate::buffer::{Buffer2D, Buffer4D};
use crate::quantize::Quantized;
use crate::tensor::Tensor4D;
use simba::scalar::SupersetOf;

pub fn concatenation_4d<
    T: Quantized,
    const BATCHES: usize,
    const ROWS: usize,
    const COLS: usize,
    const CHANS_A: usize,
    const CHANS_B: usize,
    const OUT_CHANS: usize,
>(
    input: Tensor4D<T, BATCHES, ROWS, COLS, CHANS_A, 1>,
    tail: &Tensor4D<T, BATCHES, ROWS, COLS, CHANS_B, 1>,
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
) -> Tensor4D<T, BATCHES, ROWS, COLS, OUT_CHANS, 1> {
    debug_assert_eq!(OUT_CHANS, CHANS_A + CHANS_B);
    let output: Buffer4D<T, BATCHES, ROWS, COLS, OUT_CHANS> = core::array::from_fn(|b| {
        Buffer2D::from_fn(|i, j| {
            core::array::from_fn(|c| {
                let real = if c < CHANS_A {
                    input.scale[0]
                        * (f32::from_subset(&input.buffer[b][(i, j)][c])
                            - f32::from_subset(&input.zero_point[0]))
                } else {
                    tail.scale[0]
                        * (f32::from_subset(&tail.buffer[b][(i, j)][c - CHANS_A])
                            - f32::from_subset(&tail.zero_point[0]))
                };
                crate::quantize::quantize(real, output_scale[0], output_zero_point[0])
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
    fn concatenation_4d_channels() {
        let input = Tensor4D::new([matrix![[1], [2]]], [0.25], [0]);
        let tail = Tensor4D::new([matrix![[3, 4], [5, 6]]], [0.25], [0]);
        let output: Tensor4D<i8, 1, 1, 2, 3, 1> = concatenation_4d(input, &tail, [0.25], [0]);
        assert_eq!(output.buffer[0], matrix![[1, 3, 4], [2, 5, 6]]);
    }
}
