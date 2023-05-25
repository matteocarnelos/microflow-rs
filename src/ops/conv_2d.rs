use crate::activation::FusedActivation;
use crate::quantize::Quantized;
use crate::tensor::{Tensor2D, Tensor4D};
use core::array;
use libm::{fmaxf, fminf};
use nalgebra::SMatrix;

pub struct Conv2DOptions {
    pub fused_activation: FusedActivation,
    pub padding: Conv2DPadding,
    pub strides: (usize, usize),
}

pub enum Conv2DPadding {
    SAME,
    VALID,
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
                    Conv2DPadding::SAME => {
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
                    Conv2DPadding::VALID => {
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
