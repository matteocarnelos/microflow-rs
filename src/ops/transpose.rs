use crate::buffer::{Buffer2D, Buffer4D};
use crate::quantize::Quantized;
use crate::tensor::{Tensor2D, Tensor4D};
use core::array;

/// transpose for 2D tensors. `perm` must be a permutation of `[0, 1]`
///  output dimensions stay consistent with the permutation.
pub fn transpose_2d<
    T: Quantized,
    const IN_ROWS: usize,
    const IN_COLS: usize,
    const OUT_ROWS: usize,
    const OUT_COLS: usize,
    const QUANTS: usize,
>(
    input: Tensor2D<T, IN_ROWS, IN_COLS, QUANTS>,
    perm: [usize; 2],
) -> Tensor2D<T, OUT_ROWS, OUT_COLS, QUANTS> {
    debug_assert_eq!(OUT_ROWS, [IN_ROWS, IN_COLS][perm[0]]);
    debug_assert_eq!(OUT_COLS, [IN_ROWS, IN_COLS][perm[1]]);
    let out_buf = Buffer2D::from_fn(|o0, o1| {
        let mut i = [0usize; 2];
        i[perm[0]] = o0;
        i[perm[1]] = o1;
        input.buffer[(i[0], i[1])]
    });
    Tensor2D::new(out_buf, input.scale, input.zero_point)
}

/// transpose for 4D tensors. `perm` must be a permutation of `[0, 1, 2, 3]`
///  output dimensions stay consistent with the permutation.
pub fn transpose_4d<
    T: Quantized,
    const B: usize,
    const R: usize,
    const C: usize,
    const CH: usize,
    const OB: usize,
    const OR: usize,
    const OC: usize,
    const OCH: usize,
    const QUANTS: usize,
>(
    input: Tensor4D<T, B, R, C, CH, QUANTS>,
    perm: [usize; 4],
) -> Tensor4D<T, OB, OR, OC, OCH, QUANTS> {
    debug_assert_eq!(OB, [B, R, C, CH][perm[0]]);
    debug_assert_eq!(OR, [B, R, C, CH][perm[1]]);
    debug_assert_eq!(OC, [B, R, C, CH][perm[2]]);
    debug_assert_eq!(OCH, [B, R, C, CH][perm[3]]);
    let out_buf: Buffer4D<T, OB, OR, OC, OCH> = array::from_fn(|o0| {
        Buffer2D::from_fn(|o1, o2| {
            array::from_fn(|o3| {
                let mut i = [0usize; 4];
                i[perm[0]] = o0;
                i[perm[1]] = o1;
                i[perm[2]] = o2;
                i[perm[3]] = o3;
                input.buffer[i[0]][(i[1], i[2])][i[3]]
            })
        })
    });
    Tensor4D::new(out_buf, input.scale, input.zero_point)
}

#[cfg(test)]
mod tests {
    use nalgebra::matrix;

    use super::*;
    use crate::tensor::Tensor2D;

    #[test]
    fn transpose_2d_01() {
        let t: Tensor2D<i8, 2, 3, 1> = Tensor2D::new(
            matrix![
                1, 2, 3;
                4, 5, 6
            ],
            [0.5],
            [0],
        );
        let out: Tensor2D<i8, 3, 2, 1> = transpose_2d(t, [1, 0]);
        assert_eq!(
            out.buffer,
            matrix![
                1, 4;
                2, 5;
                3, 6
            ]
        );
    }

    #[test]
    fn transpose_4d_identity() {
        let t = Tensor4D::new(
            [
                matrix![
                    [1, 2], [3, 4], [5, 6];
                    [7, 8], [9, 10], [11, 12]
                ],
                matrix![
                    [13, 14], [15, 16], [17, 18];
                    [19, 20], [21, 22], [23, 24]
                ],
            ],
            [1.0],
            [0],
        );
        let out: Tensor4D<i8, 2, 2, 3, 2, 1> = transpose_4d(t, [0, 1, 2, 3]);
        assert_eq!(
            out.buffer,
            [
                matrix![
                    [1, 2], [3, 4], [5, 6];
                    [7, 8], [9, 10], [11, 12]
                ],
                matrix![
                    [13, 14], [15, 16], [17, 18];
                    [19, 20], [21, 22], [23, 24]
                ],
            ]
        );
    }

    #[test]
    fn transpose_4d_batch_channel_swap() {
        // (B,R,C,CH) = (2,1,2,2)  ; test swaps batch (0) and channels (3) -> (2,1,2,2) with permutation [3,1,2,0]
        let t = Tensor4D::new(
            [matrix![[1, 2], [3, 4]], matrix![[5, 6], [7, 8]]],
            [1.0],
            [0],
        );
        let out: Tensor4D<i8, 2, 1, 2, 2, 1> = transpose_4d(t, [3, 1, 2, 0]);
        assert_eq!(out.buffer[0], matrix![[1, 5], [3, 7]]);
        assert_eq!(out.buffer[1], matrix![[2, 6], [4, 8]]);
    }
}
