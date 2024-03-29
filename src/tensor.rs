use core::array;
use core::fmt::Debug;

use crate::buffer::{Buffer2D, Buffer4D};
use crate::quantize::{dequantize, quantize, Quantized};

/// Represents the padding options for the [`TensorView`].
#[derive(Copy, Clone)]
pub enum TensorViewPadding {
    /// In the 'Same' padding, the [`TensorView`] is allowed to exceed the input bounds.
    /// The exceeding values will be replaced by zeros.
    Same,
    /// In the 'Valid' padding, the [`TensorView`] will always remain in the input bounds.
    Valid,
}

/// Represents the tensor view, i.e., the input region extracted from the tensor.
pub struct TensorView<T: Quantized, const ROWS: usize, const COLS: usize, const CHANS: usize> {
    pub buffer: Buffer2D<[T; CHANS], ROWS, COLS>,
    pub mask: Buffer2D<bool, ROWS, COLS>,
    pub len: usize,
}

/// Represents a quantized 2-dimensional tensor.
/// The tensor is composed by a 2-dimensional matrix.
#[derive(Debug, PartialEq)]
pub struct Tensor2D<T: Quantized, const ROWS: usize, const COLS: usize, const QUANTS: usize> {
    pub buffer: Buffer2D<T, ROWS, COLS>,
    pub scale: [f32; QUANTS],
    pub zero_point: [T; QUANTS],
}

/// Represents a quantized 4-dimensional tensor.
/// The tensor is composed by a series of batches containing matrices with multiple channels.
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
    /// Builds a quantized [`Tensor2D`] from the given buffer, scale, and zero point.
    ///
    /// # Arguments
    /// * `buffer` - The tensor buffer as a [`Buffer2D`]
    /// * `scale` - The tensor scale (for quantization)
    /// * `zero_point` - The tensor zero point (for quantization)
    ///
    pub const fn new(
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
}

impl<T: Quantized, const ROWS: usize, const COLS: usize> Tensor2D<T, ROWS, COLS, 1> {
    /// Builds a quantized [`Tensor2D`] from a dequantized [`Buffer2D`].
    ///
    /// # Arguments
    /// * `input` - The input buffer as a [`Buffer2D`]
    /// * `scale` - The quantization scale
    /// * `zero_point` - The quantization zero point
    ///
    pub fn quantize(input: Buffer2D<f32, ROWS, COLS>, scale: [f32; 1], zero_point: [T; 1]) -> Self {
        Self::new(
            input.map(|f| quantize(f, scale[0], zero_point[0])),
            scale,
            zero_point,
        )
    }

    /// Returns a dequantized [`Buffer2D`] from [`Self`].
    pub fn dequantize(&self) -> Buffer2D<f32, ROWS, COLS> {
        self.buffer
            .map(|q| dequantize(q, self.scale[0], self.zero_point[0]))
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
        const ROWS: usize,
        const COLS: usize,
        const QUANTS: usize,
        const OUTPUT_ROWS: usize,
        const OUTPUT_COLS: usize,
        const OUTPUT_CHANS: usize,
    > From<Tensor2D<T, ROWS, COLS, QUANTS>>
    for Tensor4D<T, ROWS, OUTPUT_ROWS, OUTPUT_COLS, OUTPUT_CHANS, QUANTS>
{
    fn from(tensor: Tensor2D<T, ROWS, COLS, QUANTS>) -> Self {
        Self::new(
            array::from_fn(|b| {
                Buffer2D::from_fn(|i, j| {
                    array::from_fn(|c| {
                        tensor.buffer[(b, OUTPUT_CHANS * OUTPUT_COLS * i + OUTPUT_CHANS * j + c)]
                    })
                })
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
    /// Builds a quantized [`Tensor4D`] from the given buffer, scale, and zero point.
    ///
    /// # Arguments
    /// * `buffer` - The tensor buffer as a [`Buffer4D`]
    /// * `scale` - The tensor scale (for quantization)
    /// * `zero_point` - The tensor zero point (for quantization)
    ///
    pub const fn new(
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

    /// Extracts a view from the tensor.
    /// Returns the 4-dimensional tensor view as a [`TensorView`] struct.
    ///
    /// # Arguments
    /// * `focus` - The focus point of the view, i.e., the pseudo-center of the view
    /// * `batch` - The tensor batch from which to extract the view
    /// * `padding` - The view padding as a [`TensorViewPadding`] enum
    /// * `strides` - The view strides on the width and height of the tensor, repectively
    ///
    pub fn view<const VIEW_ROWS: usize, const VIEW_COLS: usize>(
        &self,
        focus: (usize, usize),
        batch: usize,
        padding: TensorViewPadding,
        strides: (usize, usize),
    ) -> TensorView<T, VIEW_ROWS, VIEW_COLS, CHANS> {
        let mut len = VIEW_ROWS * VIEW_COLS;
        let mut mask = Buffer2D::from_element(true);
        TensorView {
            buffer: Buffer2D::from_fn(|m, n| match padding {
                TensorViewPadding::Same => {
                    // Compute the index shift based on the view dimensions
                    let shift = ((VIEW_ROWS - 1) / 2, (VIEW_COLS - 1) / 2);
                    let index = (
                        // If the calculated index falls within the tensor bounds, keep it
                        if let Some(x) = (strides.0 * focus.0 + m).checked_sub(shift.0) {
                            x
                        // Otherwise, return zero (as per "same" padding)
                        } else {
                            len -= 1;
                            mask[(m, n)] = false;
                            return [T::from_superset_unchecked(&0); CHANS];
                        },
                        // Same for the other index value
                        if let Some(x) = (strides.1 * focus.1 + n).checked_sub(shift.1) {
                            x
                        } else {
                            len -= 1;
                            mask[(m, n)] = false;
                            return [T::from_superset_unchecked(&0); CHANS];
                        },
                    );
                    // Extract the view for the computed index
                    self.buffer[batch].get(index).copied().unwrap_or_else(|| {
                        len -= 1;
                        mask[(m, n)] = false;
                        [T::from_superset_unchecked(&0); CHANS]
                    })
                }
                TensorViewPadding::Valid => {
                    // For "valid" paddings, directly extract the view for valid indexes only
                    self.buffer[batch][(strides.0 * focus.0 + m, strides.1 * focus.1 + n)]
                }
            }),
            mask,
            len,
        }
    }
}

impl<
        T: Quantized,
        const BATCHES: usize,
        const ROWS: usize,
        const COLS: usize,
        const CHANS: usize,
    > Tensor4D<T, BATCHES, ROWS, COLS, CHANS, 1>
{
    /// Builds a quantized [`Tensor4D`] from a dequantized [`Buffer4D`].
    ///
    /// # Arguments
    /// * `input` - The input buffer as a [`Buffer4D`]
    /// * `scale` - The quantization scale
    /// * `zero_point` - The quantization zero point
    ///
    pub fn quantize(
        input: Buffer4D<f32, BATCHES, ROWS, COLS, CHANS>,
        scale: [f32; 1],
        zero_point: [T; 1],
    ) -> Self {
        Self::new(
            input.map(|m| m.map(|a| a.map(|f| quantize(f, scale[0], zero_point[0])))),
            scale,
            zero_point,
        )
    }

    /// Returns a dequantized [`Buffer2D`] from [`Self`].
    pub fn dequantize(&self) -> Buffer4D<f32, BATCHES, ROWS, COLS, CHANS> {
        self.buffer
            .map(|m| m.map(|a| a.map(|q| dequantize(q, self.scale[0], self.zero_point[0]))))
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
    const TENSOR_4D_SCALE: [f32; 1] = [0.25];
    const TENSOR_4D_ZERO_POINT: [i8; 1] = [26];
    const TENSOR_4D_BUFFER_QUANTIZED: Buffer4D<i8, 2, 2, 3, 2> = [
        matrix![
            [30, 34], [38, 42], [46, 50];
            [54, 58], [62, 66], [70, 74]
        ],
        matrix![
            [78,  82],  [86,  90],  [94,  98];
            [102, 106], [110, 114], [118, 122]
        ],
    ];
    const TENSOR_4D_VIEW_BUFFER: Buffer2D<[i8; 2], 2, 3> = matrix![
        [54, 58], [62, 66], [70, 74];
        [0,  0],  [0,  0],  [0,  0]
    ];
    const TENSOR_4D_VIEW_MASK: Buffer2D<bool, 2, 3> = matrix![
        true,  true,  true;
        false, false, false
    ];
    const TENSOR_4D_VIEW_LEN: usize = 3;

    const TENSOR_4D_TO_TENSOR_2D_BUFFER: Buffer2D<i8, 2, 12> = matrix![
        30, 34, 38, 42, 46, 50, 54,  58,  62,  66,  70,  74;
        78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122
    ];

    #[test]
    fn tensor_2d_new() {
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
    fn tensor_2d_quantize() {
        let tensor = Tensor2D::quantize(TENSOR_2D_BUFFER, TENSOR_2D_SCALE, TENSOR_2D_ZERO_POINT);
        assert_eq!(tensor.buffer, TENSOR_2D_BUFFER_QUANTIZED);
    }

    #[test]
    fn tensor_2d_dequantize() {
        let tensor = Tensor2D::new(
            TENSOR_2D_BUFFER_QUANTIZED,
            TENSOR_2D_SCALE,
            TENSOR_2D_ZERO_POINT,
        );
        assert_eq!(tensor.dequantize(), TENSOR_2D_BUFFER_DEQUANTIZED);
    }

    #[test]
    fn tensor_2d_to_tensor_4d() {
        let tensor_2d = Tensor2D::new(
            TENSOR_4D_TO_TENSOR_2D_BUFFER,
            TENSOR_4D_SCALE,
            TENSOR_4D_ZERO_POINT,
        );
        let tensor_4d: Tensor4D<i8, 2, 2, 3, 2, 1> = Tensor4D::from(tensor_2d);
        assert_eq!(tensor_4d.buffer, TENSOR_4D_BUFFER_QUANTIZED);
    }

    #[test]
    fn tensor_4d_new() {
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
    fn tensor_4d_quantize() {
        let tensor = Tensor4D::quantize(TENSOR_4D_BUFFER, TENSOR_4D_SCALE, TENSOR_4D_ZERO_POINT);
        assert_eq!(tensor.buffer, TENSOR_4D_BUFFER_QUANTIZED);
    }

    #[test]
    fn tensor_4d_dequantize() {
        let tensor = Tensor4D::new(
            TENSOR_4D_BUFFER_QUANTIZED,
            TENSOR_4D_SCALE,
            TENSOR_4D_ZERO_POINT,
        );
        assert_eq!(tensor.dequantize(), TENSOR_4D_BUFFER);
    }

    #[test]
    fn tensor_4d_view() {
        let tensor = Tensor4D::new(
            TENSOR_4D_BUFFER_QUANTIZED,
            TENSOR_4D_SCALE,
            TENSOR_4D_ZERO_POINT,
        );
        let view: TensorView<i8, 2, 3, 2> = tensor.view((1, 1), 0, TensorViewPadding::Same, (1, 1));
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
        let tensor_2d: Tensor2D<i8, 2, 12, 1> = Tensor2D::from(tensor_4d);
        assert_eq!(tensor_2d.buffer, TENSOR_4D_TO_TENSOR_2D_BUFFER);
    }
}
