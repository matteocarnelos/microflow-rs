use libm::roundf;
use simba::scalar::SupersetOf;

use crate::activation::{relu, relu6, FusedActivation};
use crate::buffer::{Buffer2D, Buffer4D};
use crate::quantize::Quantized;
use crate::tensor::{Tensor2D, Tensor4D};

pub struct AddOptions {
    pub fused_activation: FusedActivation,
}

pub fn add_2d<T: Quantized, const ROWS: usize, const COLS: usize>(
    input: Tensor2D<T, ROWS, COLS, 1>,
    addend: &Tensor2D<T, ROWS, COLS, 1>,
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
    options: AddOptions,
) -> Tensor2D<T, ROWS, COLS, 1> {
    let output = Buffer2D::from_fn(|i, j| {
        let lhs = f32::from_subset(&input.buffer[(i, j)]); 
        let rhs = f32::from_subset(&addend.buffer[(i, j)]);
        let x = input.scale[0] * (lhs - f32::from_subset(&input.zero_point[0]))
            + addend.scale[0] * (rhs - f32::from_subset(&addend.zero_point[0]));
        let y = T::from_superset_unchecked(&roundf(
            x / output_scale[0] + f32::from_subset(&output_zero_point[0]),
        ));
        match options.fused_activation {
            FusedActivation::None => y,
            FusedActivation::Relu => relu(y, output_zero_point[0]),
            FusedActivation::Relu6 => relu6(y, output_scale[0], output_zero_point[0]),
        }
    });
    Tensor2D::new(output, output_scale, output_zero_point)
}

pub fn add_4d<
    T: Quantized,
    const BATCHES: usize,
    const ROWS: usize,
    const COLS: usize,
    const CHANS: usize,
>(
    input: Tensor4D<T, BATCHES, ROWS, COLS, CHANS, 1>,
    addend: &Tensor4D<T, BATCHES, ROWS, COLS, CHANS, 1>,
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
    options: AddOptions,
) -> Tensor4D<T, BATCHES, ROWS, COLS, CHANS, 1> {
    let output: Buffer4D<T, BATCHES, ROWS, COLS, CHANS> = core::array::from_fn(|b| {
        Buffer2D::from_fn(|i, j| {
            core::array::from_fn(|c| {
                let lhs = f32::from_subset(&input.buffer[b][(i, j)][c]);
                let rhs = f32::from_subset(&addend.buffer[b][(i, j)][c]);
                let x = input.scale[0] * (lhs - f32::from_subset(&input.zero_point[0]))
                    + addend.scale[0] * (rhs - f32::from_subset(&addend.zero_point[0]));
                let y = T::from_superset_unchecked(&roundf(
                    x / output_scale[0] + f32::from_subset(&output_zero_point[0]),
                ));
                match options.fused_activation {
                    FusedActivation::None => y,
                    FusedActivation::Relu => relu(y, output_zero_point[0]),
                    FusedActivation::Relu6 => relu6(y, output_scale[0], output_zero_point[0]),
                }
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
    fn add_2d_layer() {
        let input = Tensor2D::new(matrix![10, 12], [0.5], [10]);
        let addend = Tensor2D::new(matrix![7, 9], [0.5], [8]);
        let output: Tensor2D<i8, 1, 2, 1> = add_2d(
            input,
            &addend,
            [0.5],
            [11],
            AddOptions {
                fused_activation: FusedActivation::None,
            },
        );
        assert_eq!(output.buffer, matrix![10, 14]);
    }
}
