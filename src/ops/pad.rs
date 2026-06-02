use crate::buffer::{Buffer2D, Buffer4D};
use crate::quantize::Quantized;
use crate::tensor::Tensor4D;

pub fn pad_4d<
    T: Quantized,
    const BATCHES: usize,
    const ROWS: usize,
    const COLS: usize,
    const CHANS: usize,
    const OUT_BATCHES: usize,
    const OUT_ROWS: usize,
    const OUT_COLS: usize,
    const OUT_CHANS: usize,
>(
    input: Tensor4D<T, BATCHES, ROWS, COLS, CHANS, 1>,
    paddings: [(usize, usize); 4],
    constant: T,
) -> Tensor4D<T, OUT_BATCHES, OUT_ROWS, OUT_COLS, OUT_CHANS, 1> {
    debug_assert_eq!(OUT_BATCHES, paddings[0].0 + BATCHES + paddings[0].1);
    debug_assert_eq!(OUT_ROWS, paddings[1].0 + ROWS + paddings[1].1);
    debug_assert_eq!(OUT_COLS, paddings[2].0 + COLS + paddings[2].1);
    debug_assert_eq!(OUT_CHANS, paddings[3].0 + CHANS + paddings[3].1);

    let output: Buffer4D<T, OUT_BATCHES, OUT_ROWS, OUT_COLS, OUT_CHANS> = core::array::from_fn(|b| {
        Buffer2D::from_fn(|i, j| {
            core::array::from_fn(|c| {
                if b >= paddings[0].0
                    && b < paddings[0].0 + BATCHES
                    && i >= paddings[1].0
                    && i < paddings[1].0 + ROWS
                    && j >= paddings[2].0
                    && j < paddings[2].0 + COLS
                    && c >= paddings[3].0
                    && c < paddings[3].0 + CHANS
                {
                    input.buffer[b - paddings[0].0][(i - paddings[1].0, j - paddings[2].0)]
                        [c - paddings[3].0]
                } else {
                    constant
                }
            })
        })
    });

    Tensor4D::new(output, input.scale, input.zero_point)
}

#[cfg(test)]
mod tests {
    use nalgebra::matrix;

    use super::*;

    #[test]
    fn pad_4d_constant() {
        let input = Tensor4D::new([matrix![[1], [2]]], [0.25], [0]);
        let output: Tensor4D<i8, 1, 1, 4, 1, 1> =
            pad_4d(input, [(0, 0), (0, 0), (1, 1), (0, 0)], -1);
        assert_eq!(output.buffer[0], matrix![[-1], [1], [2], [-1]]);
    }
}
