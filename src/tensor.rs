use core::fmt::Debug;

use crate::buffer::{Buffer2D, Buffer4D};
use crate::quantize::{dequantize, quantize, Quantized};

#[derive(Copy, Clone)]
pub enum ViewPadding {
    Same,
    Valid,
}

pub struct View<T: Quantized, const ROWS: usize, const COLS: usize, const CHANS: usize> {
    pub buffer: Buffer2D<[T; CHANS], ROWS, COLS>,
    pub mask: Buffer2D<bool, ROWS, COLS>,
    pub len: usize,
}

#[derive(Debug, PartialEq)]
pub struct Tensor2D<T: Quantized, const ROWS: usize, const COLS: usize, const QUANTS: usize> {
    pub buffer: Buffer2D<T, ROWS, COLS>,
    pub scale: [f32; QUANTS],
    pub zero_point: [T; QUANTS],
}

#[derive(Debug, PartialEq)]
pub struct Tensor4D<
    T: Quantized,
    const BATCHES: usize,
    const ROWS: usize,
    const COLS: usize,
    const CHANS: usize,
    const QUANTS: usize,
> {
    pub buffer: Buffer4D<T, BATCHES, ROWS, COLS, CHANS>,
    pub scale: [f32; QUANTS],
    pub zero_point: [T; QUANTS],
}

impl<T: Quantized, const ROWS: usize, const COLS: usize, const QUANTS: usize>
    Tensor2D<T, ROWS, COLS, QUANTS>
{
    pub fn new(
        buffer: Buffer2D<T, ROWS, COLS>,
        scale: [f32; QUANTS],
        zero_point: [T; QUANTS],
    ) -> Self {
        Self {
            buffer,
            scale,
            zero_point,
        }
    }

    pub fn quantize(
        input: Buffer2D<f32, ROWS, COLS>,
        scale: [f32; QUANTS],
        zero_point: [T; QUANTS],
    ) -> Self {
        Self::new(
            input.map_with_location(|i, _, f| {
                quantize(
                    f,
                    scale.get(i).copied().unwrap_or(scale[0]),
                    zero_point.get(i).copied().unwrap_or(zero_point[0]),
                )
            }),
            scale,
            zero_point,
        )
    }

    pub fn dequantize(&self) -> Buffer2D<f32, ROWS, COLS> {
        self.buffer.map_with_location(|i, _, q| {
            dequantize(
                q,
                self.scale.get(i).copied().unwrap_or(self.scale[0]),
                self.zero_point
                    .get(i)
                    .copied()
                    .unwrap_or(self.zero_point[0]),
            )
        })
    }
}

impl<
        T: Quantized,
        const BATCHES: usize,
        const ROWS: usize,
        const COLS: usize,
        const CHANS: usize,
        const QUANTS: usize,
        const OUTPUT_COLS: usize,
    > From<Tensor4D<T, BATCHES, ROWS, COLS, CHANS, QUANTS>>
    for Tensor2D<T, BATCHES, OUTPUT_COLS, QUANTS>
{
    fn from(tensor: Tensor4D<T, BATCHES, ROWS, COLS, CHANS, QUANTS>) -> Self {
        Self::new(
            Buffer2D::from_fn(|i, j| {
                tensor.buffer[i][(j / (CHANS * COLS), j / CHANS % COLS)][j % CHANS]
            }),
            tensor.scale,
            tensor.zero_point,
        )
    }
}

impl<
        T: Quantized,
        const BATCHES: usize,
        const ROWS: usize,
        const COLS: usize,
        const CHANS: usize,
        const QUANTS: usize,
    > Tensor4D<T, BATCHES, ROWS, COLS, CHANS, QUANTS>
{
    pub fn new(
        buffer: Buffer4D<T, BATCHES, ROWS, COLS, CHANS>,
        scale: [f32; QUANTS],
        zero_point: [T; QUANTS],
    ) -> Self {
        Self {
            buffer,
            scale,
            zero_point,
        }
    }

    pub fn quantize(
        input: Buffer4D<f32, BATCHES, ROWS, COLS, CHANS>,
        scale: [f32; QUANTS],
        zero_point: [T; QUANTS],
    ) -> Self {
        Self::new(
            input.map(|m| {
                m.map(|a| {
                    let mut iter = 0..CHANS;
                    a.map(|f| {
                        let i = iter.next().unwrap();
                        quantize(
                            f,
                            scale.get(i).copied().unwrap_or(scale[0]),
                            zero_point.get(i).copied().unwrap_or(zero_point[0]),
                        )
                    })
                })
            }),
            scale,
            zero_point,
        )
    }

    pub fn dequantize(&self) -> Buffer4D<f32, BATCHES, ROWS, COLS, CHANS> {
        self.buffer.map(|m| {
            m.map(|a| {
                let mut iter = 0..CHANS;
                a.map(|q| {
                    let i = iter.next().unwrap();
                    dequantize(
                        q,
                        self.scale.get(i).copied().unwrap_or(self.scale[0]),
                        self.zero_point
                            .get(i)
                            .copied()
                            .unwrap_or(self.zero_point[0]),
                    )
                })
            })
        })
    }

    pub fn view<const VIEW_ROWS: usize, const VIEW_COLS: usize>(
        &self,
        focus: (usize, usize),
        batch: usize,
        padding: ViewPadding,
        strides: (usize, usize),
    ) -> View<T, VIEW_ROWS, VIEW_COLS, CHANS> {
        let mut len = VIEW_ROWS * VIEW_COLS;
        let mut mask = Buffer2D::from_element(true);
        View {
            buffer: Buffer2D::from_fn(|m, n| match padding {
                ViewPadding::Same => {
                    let shift = ((VIEW_ROWS - 1) / 2, (VIEW_COLS - 1) / 2);
                    let index = (
                        if let Some(x) = (strides.0 * focus.0 + m).checked_sub(shift.0) {
                            x
                        } else {
                            len -= 1;
                            mask[(m, n)] = false;
                            return [T::from_superset_unchecked(&0); CHANS];
                        },
                        if let Some(x) = (strides.1 * focus.1 + n).checked_sub(shift.1) {
                            x
                        } else {
                            len -= 1;
                            mask[(m, n)] = false;
                            return [T::from_superset_unchecked(&0); CHANS];
                        },
                    );
                    self.buffer[batch].get(index).copied().unwrap_or_else(|| {
                        len -= 1;
                        mask[(m, n)] = false;
                        [T::from_superset_unchecked(&0); CHANS]
                    })
                }
                ViewPadding::Valid => {
                    self.buffer[batch][(strides.0 * focus.0 + m, strides.1 * focus.1 + n)]
                }
            }),
            mask,
            len,
        }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::matrix;

    use super::*;

    const TENSOR_2D_BUFFER: Buffer2D<f32, 2, 3> = matrix![
        1., 2., 3.;
        4., 5., 6.
    ];
    const TENSOR_2D_SCALE: [f32; 1] = [0.7];
    const TENSOR_2D_ZERO_POINT: [i8; 1] = [8];
    const TENSOR_2D_BUFFER_QUANTIZED: Buffer2D<i8, 2, 3> = matrix![
        9,  11, 12;
        14, 15, 17
    ];
    const TENSOR_2D_BUFFER_DEQUANTIZED: Buffer2D<f32, 2, 3> = matrix![
        0.7, 2.1, 2.8;
        4.2, 4.9, 6.2999997
    ];

    const TENSOR_4D_BUFFER: Buffer4D<f32, 2, 2, 3, 2> = [
        matrix![
            [1., 2.], [3., 4.],  [5., 6.];
            [7., 8.], [9., 10.], [11., 12.]
        ],
        matrix![
            [13., 14.], [15., 16.], [17., 18.];
            [19., 20.], [21., 22.], [23., 24.]
        ],
    ];
    const TENSOR_4D_SCALE: [f32; 2] = [0.25, 0.26];
    const TENSOR_4D_ZERO_POINT: [i8; 2] = [27, 28];
    const TENSOR_4D_BUFFER_QUANTIZED: Buffer4D<i8, 2, 2, 3, 2> = [
        matrix![
            [31, 36], [39, 43], [47, 51];
            [55, 59], [63, 66], [71, 74]
        ],
        matrix![
            [79, 82],   [87, 90],   [95, 97];
            [103, 105], [111, 113], [119, 120]
        ],
    ];
    const TENSOR_4D_BUFFER_DEQUANTIZED: Buffer4D<f32, 2, 2, 3, 2> = [
        matrix![
            [1., 2.08],     [3., 3.8999999],  [5., 5.9799995];
            [7., 8.059999], [9., 9.879999],   [11., 11.959999]
        ],
        matrix![
            [13., 14.039999], [15., 16.119999], [17., 17.939999];
            [19., 20.019999], [21., 22.099998], [23., 23.919998]
        ],
    ];
    const TENSOR_4D_VIEW_BUFFER: Buffer2D<[i8; 2], 2, 3> = matrix![
        [55, 59], [63, 66], [71, 74];
        [0, 0],   [0, 0],    [0, 0]
    ];
    const TENSOR_4D_VIEW_MASK: Buffer2D<bool, 2, 3> = matrix![
        true,  true,  true;
        false, false, false
    ];
    const TENSOR_4D_VIEW_LEN: usize = 3;

    const TENSOR_4D_TO_TENSOR_2D_BUFFER: Buffer2D<i8, 2, 12> = matrix![
        31, 36, 39, 43, 47, 51, 55,  59,  63,  66,  71,  74;
        79, 82, 87, 90, 95, 97, 103, 105, 111, 113, 119, 120
    ];

    #[test]
    fn new_tensor_2d() {
        let tensor = Tensor2D::new(
            TENSOR_2D_BUFFER_QUANTIZED,
            TENSOR_2D_SCALE,
            TENSOR_2D_ZERO_POINT,
        );
        assert_eq!(tensor.buffer, TENSOR_2D_BUFFER_QUANTIZED);
        assert_eq!(tensor.scale, TENSOR_2D_SCALE);
        assert_eq!(tensor.zero_point, TENSOR_2D_ZERO_POINT);
    }

    #[test]
    fn quantize_tensor_2d() {
        let tensor = Tensor2D::quantize(TENSOR_2D_BUFFER, TENSOR_2D_SCALE, TENSOR_2D_ZERO_POINT);
        assert_eq!(tensor.buffer, TENSOR_2D_BUFFER_QUANTIZED);
    }

    #[test]
    fn dequantize_tensor_2d() {
        let tensor = Tensor2D::new(
            TENSOR_2D_BUFFER_QUANTIZED,
            TENSOR_2D_SCALE,
            TENSOR_2D_ZERO_POINT,
        );
        assert_eq!(tensor.dequantize(), TENSOR_2D_BUFFER_DEQUANTIZED);
    }

    #[test]
    fn new_tensor_4d() {
        let tensor = Tensor4D::new(
            TENSOR_4D_BUFFER_QUANTIZED,
            TENSOR_4D_SCALE,
            TENSOR_4D_ZERO_POINT,
        );
        assert_eq!(tensor.buffer, TENSOR_4D_BUFFER_QUANTIZED);
        assert_eq!(tensor.scale, TENSOR_4D_SCALE);
        assert_eq!(tensor.zero_point, TENSOR_4D_ZERO_POINT);
    }

    #[test]
    fn quantize_tensor_4d() {
        let tensor = Tensor4D::quantize(TENSOR_4D_BUFFER, TENSOR_4D_SCALE, TENSOR_4D_ZERO_POINT);
        assert_eq!(tensor.buffer, TENSOR_4D_BUFFER_QUANTIZED);
    }

    #[test]
    fn dequantize_tensor_4d() {
        let tensor = Tensor4D::new(
            TENSOR_4D_BUFFER_QUANTIZED,
            TENSOR_4D_SCALE,
            TENSOR_4D_ZERO_POINT,
        );
        assert_eq!(tensor.dequantize(), TENSOR_4D_BUFFER_DEQUANTIZED);
    }

    #[test]
    fn view_tensor_4d() {
        let tensor = Tensor4D::new(
            TENSOR_4D_BUFFER_QUANTIZED,
            TENSOR_4D_SCALE,
            TENSOR_4D_ZERO_POINT,
        );
        let view: View<i8, 2, 3, 2> = tensor.view((1, 1), 0, ViewPadding::Same, (1, 1));
        assert_eq!(view.buffer, TENSOR_4D_VIEW_BUFFER);
        assert_eq!(view.mask, TENSOR_4D_VIEW_MASK);
        assert_eq!(view.len, TENSOR_4D_VIEW_LEN);
    }

    #[test]
    fn tensor_4d_to_tensor_2d() {
        let tensor_4d = Tensor4D::new(
            TENSOR_4D_BUFFER_QUANTIZED,
            TENSOR_4D_SCALE,
            TENSOR_4D_ZERO_POINT,
        );
        let tensor_2d: Tensor2D<i8, 2, 12, 2> = Tensor2D::from(tensor_4d);
        assert_eq!(tensor_2d.buffer, TENSOR_4D_TO_TENSOR_2D_BUFFER);
    }
}
