use core::fmt::Debug;

use crate::buffer::{Buffer2D, Buffer4D};

use crate::quantize::{dequantize, quantize, Quantized};

#[derive(Debug, PartialEq)]
pub struct QuantizedTensor2D<T: Quantized, const D1: usize, const D2: usize> {
    pub buffer: Buffer2D<T, D1, D2>,
    pub scale: f32,
    pub zero_point: T,
}

#[derive(Debug, PartialEq)]
pub struct QuantizedTensor4D<
    T: Quantized,
    const D1: usize,
    const D2: usize,
    const D3: usize,
    const D4: usize,
    const D4_OR_1: usize,
> {
    pub buffer: Buffer4D<T, D1, D2, D3, D4>,
    pub scale: [f32; D4_OR_1],
    pub zero_point: [T; D4_OR_1],
}

impl<T: Quantized, const D1: usize, const D2: usize> QuantizedTensor2D<T, D1, D2> {
    pub fn new(buffer: Buffer2D<T, D1, D2>, scale: f32, zero_point: T) -> Self {
        Self {
            buffer,
            scale,
            zero_point,
        }
    }

    pub fn quantize(input: Buffer2D<f32, D1, D2>, scale: f32, zero_point: T) -> Self {
        Self::new(
            input.map(|f| quantize(f, scale, zero_point)),
            scale,
            zero_point,
        )
    }

    pub fn dequantize(self) -> Buffer2D<f32, D1, D2> {
        self.buffer
            .map(|q| dequantize(q, self.scale, self.zero_point))
    }
}

impl<
        T: Quantized,
        const D1: usize,
        const D2: usize,
        const D3: usize,
        const D4: usize,
        const D4_OR_1: usize,
        const D2_X_D3_X_D4: usize,
    > From<QuantizedTensor4D<T, D1, D2, D3, D4, D4_OR_1>>
    for QuantizedTensor2D<T, D1, D2_X_D3_X_D4>
{
    fn from(tensor: QuantizedTensor4D<T, D1, D2, D3, D4, D4_OR_1>) -> Self {
        Self::new(
            // TODO: Optimize conversion by removing the transpose
            Buffer2D::from_row_iterator(
                tensor
                    .buffer
                    .map(|m| m.transpose())
                    .iter()
                    .flatten()
                    .flatten()
                    .copied(),
            ),
            tensor.scale[0],
            tensor.zero_point[0],
        )
    }
}

impl<
        T: Quantized,
        const D1: usize,
        const D2: usize,
        const D3: usize,
        const D4: usize,
        const D4_OR_1: usize,
    > QuantizedTensor4D<T, D1, D2, D3, D4, D4_OR_1>
{
    pub fn new(
        buffer: Buffer4D<T, D1, D2, D3, D4>,
        scale: [f32; D4_OR_1],
        zero_point: [T; D4_OR_1],
    ) -> Self {
        Self {
            buffer,
            scale,
            zero_point,
        }
    }

    pub fn quantize(
        input: Buffer4D<f32, D1, D2, D3, D4>,
        scale: [f32; D4_OR_1],
        zero_point: [T; D4_OR_1],
    ) -> Self {
        Self::new(
            input.map(|m| {
                m.map(|a| {
                    let mut iter = 0..D4;
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

    pub fn dequantize(self) -> Buffer4D<f32, D1, D2, D3, D4> {
        self.buffer.map(|m| {
            m.map(|a| {
                let mut iter = 0..D4;
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
}

#[cfg(test)]
mod tests {
    use nalgebra::matrix;

    use super::*;

    const TENSOR_2D_BUFFER: Buffer2D<f32, 2, 3> = matrix![
        1., 2., 3.;
        4., 5., 6.
    ];
    const TENSOR_2D_SCALE: f32 = 0.7;
    const TENSOR_2D_ZERO_POINT: i8 = 8;
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

    const TENSOR_4D_TO_TENSOR_2D_BUFFER: Buffer2D<i8, 2, 12> = matrix![
        31, 36, 39, 43, 47, 51, 55,  59,  63,  66,  71,  74;
        79, 82, 87, 90, 95, 97, 103, 105, 111, 113, 119, 120
    ];

    #[test]
    fn new_tensor_2d() {
        let tensor = QuantizedTensor2D::new(
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
        let tensor =
            QuantizedTensor2D::quantize(TENSOR_2D_BUFFER, TENSOR_2D_SCALE, TENSOR_2D_ZERO_POINT);
        assert_eq!(tensor.buffer, TENSOR_2D_BUFFER_QUANTIZED);
    }

    #[test]
    fn dequantize_tensor_2d() {
        let tensor = QuantizedTensor2D::new(
            TENSOR_2D_BUFFER_QUANTIZED,
            TENSOR_2D_SCALE,
            TENSOR_2D_ZERO_POINT,
        );
        assert_eq!(tensor.dequantize(), TENSOR_2D_BUFFER_DEQUANTIZED);
    }

    #[test]
    fn new_tensor_4d() {
        let tensor = QuantizedTensor4D::new(
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
        let tensor =
            QuantizedTensor4D::quantize(TENSOR_4D_BUFFER, TENSOR_4D_SCALE, TENSOR_4D_ZERO_POINT);
        assert_eq!(tensor.buffer, TENSOR_4D_BUFFER_QUANTIZED);
    }

    #[test]
    fn dequantize_tensor_4d() {
        let tensor = QuantizedTensor4D::new(
            TENSOR_4D_BUFFER_QUANTIZED,
            TENSOR_4D_SCALE,
            TENSOR_4D_ZERO_POINT,
        );
        assert_eq!(tensor.dequantize(), TENSOR_4D_BUFFER_DEQUANTIZED);
    }

    #[test]
    fn tensor_4d_to_tensor_2d() {
        let tensor_4d = QuantizedTensor4D::new(
            TENSOR_4D_BUFFER_QUANTIZED,
            [TENSOR_4D_SCALE[0]],
            [TENSOR_4D_ZERO_POINT[0]],
        );
        let tensor_2d: QuantizedTensor2D<i8, 2, 12> = QuantizedTensor2D::from(tensor_4d);
        assert_eq!(tensor_2d.buffer, TENSOR_4D_TO_TENSOR_2D_BUFFER);
    }
}
