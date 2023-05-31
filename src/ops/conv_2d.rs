use crate::activation::FusedActivation;
use crate::quantize::Quantized;
use crate::tensor::{Tensor2D, Tensor4D, ViewPadding};
use core::array;
use libm::{fmaxf, fminf};
use nalgebra::SMatrix;

// TODO: Optimize (constants + quantized op)

pub struct Conv2DOptions {
    pub fused_activation: FusedActivation,
    pub padding: ViewPadding,
    pub strides: (usize, usize),
}

pub fn conv_2d<
    T: Quantized,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const FILTERS_BATCHES: usize,
    const FILTERS_ROWS: usize,
    const FILTERS_COLS: usize,
    const FILTERS_QUANTS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
>(
    input: Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    filters: Tensor4D<T, FILTERS_BATCHES, FILTERS_ROWS, FILTERS_COLS, INPUT_CHANS, FILTERS_QUANTS>,
    biases: Tensor2D<i32, FILTERS_BATCHES, 1, FILTERS_QUANTS>,
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
    options: Conv2DOptions,
) -> Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, FILTERS_BATCHES, 1> {
    let input = input.dequantize();
    let filters = filters.dequantize();
    let biases = biases.dequantize();
    let output = [SMatrix::from_fn(|i, j| {
        array::from_fn(|b| {
            let view: SMatrix<[f32; INPUT_CHANS], FILTERS_ROWS, FILTERS_COLS> =
                SMatrix::from_fn(|m, n| match options.padding {
                    ViewPadding::SAME => {
                        let shift = ((FILTERS_ROWS - 1) / 2, (FILTERS_COLS - 1) / 2);
                        let index = (
                            if let Some(x) = (options.strides.0 * i + m).checked_sub(shift.0) {
                                x
                            } else {
                                return [0.; INPUT_CHANS];
                            },
                            if let Some(x) = (options.strides.1 * j + n).checked_sub(shift.1) {
                                x
                            } else {
                                return [0.; INPUT_CHANS];
                            },
                        );
                        input[0].get(index).copied().unwrap_or([0.; INPUT_CHANS])
                    }
                    ViewPadding::VALID => {
                        input[0][(options.strides.0 * i + m, options.strides.1 * j + n)]
                    }
                });
            let y = view.zip_fold(&filters[b], 0f32, |acc, a1, a2| {
                acc + a1
                    .iter()
                    .zip(a2.iter())
                    .map(|(e1, e2)| e1 * e2)
                    .sum::<f32>()
            }) + biases[b];
            match options.fused_activation {
                FusedActivation::NONE => y,
                FusedActivation::RELU => fmaxf(y, 0.),
                FusedActivation::RELU6 => fminf(fmaxf(y, 0.), 6.),
            }
        })
    })];
    Tensor4D::quantize(output, output_scale, output_zero_point)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::matrix;

    const INPUT: Tensor4D<i8, 1, 2, 3, 2, 1> = Tensor4D {
        buffer: [matrix![
            [1, 2], [3, 4], [5, 6];
            [7, 8], [9, 10], [11, 12]
        ]],
        scale: [0.13],
        zero_point: [14],
    };
    const FILTERS: Tensor4D<i8, 2, 2, 3, 2, 2> = Tensor4D {
        buffer: [
            matrix![
                [15, 16], [17, 18], [19, 20];
                [21, 22], [23, 24], [25, 26]
            ],
            matrix![
                [27, 28], [29, 30], [31, 32];
                [33, 34], [35, 36], [37, 38]
            ],
        ],
        scale: [0.39, 0.40],
        zero_point: [41, 42],
    };
    const BIASES: Tensor2D<i32, 2, 1, 2> = Tensor2D {
        buffer: matrix![
            43;
            44
        ],
        scale: [0.45, 0.46],
        zero_point: [47, 48],
    };
    const OUTPUT_SCALE: [f32; 1] = [0.49];
    const OUTPUT_ZERO_POINT: [i8; 1] = [50];
    const OPTIONS: Conv2DOptions = Conv2DOptions {
        fused_activation: FusedActivation::NONE,
        padding: ViewPadding::SAME,
        strides: (1, 1),
    };
    const OUTPUT: Tensor4D<i8, 1, 2, 3, 2, 1> = Tensor4D {
        buffer: [matrix![
            [127, 112], [127, 127], [127, 109];
            [100, 72],  [116, 82],  [83, 66]
        ]],
        scale: [0.49],
        zero_point: [50],
    };

    #[test]
    fn conv_2d_layer() {
        assert_eq!(
            conv_2d(
                INPUT,
                FILTERS,
                BIASES,
                OUTPUT_SCALE,
                OUTPUT_ZERO_POINT,
                OPTIONS
            ),
            OUTPUT
        );
    }
}
