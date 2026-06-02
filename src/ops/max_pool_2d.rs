use core::array;

use nalgebra::Const;
use simba::scalar::SupersetOf;

use crate::activation::{relu, relu6, FusedActivation};
use crate::buffer::Buffer2D;
use crate::quantize::Quantized;
use crate::tensor::{Tensor4D, TensorView, TensorViewPadding};

pub struct MaxPool2DOptions {
    pub fused_activation: FusedActivation,
    pub view_padding: TensorViewPadding,
    pub strides: (usize, usize),
}

pub fn max_pool_2d<
    T: Quantized,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const FILTER_ROWS: usize,
    const FILTER_COLS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
>(
    input: Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    _filter_shape: (Const<FILTER_ROWS>, Const<FILTER_COLS>),
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
    options: MaxPool2DOptions,
) -> Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, INPUT_CHANS, 1> {
    let output = [Buffer2D::from_fn(|i, j| {
        let view: TensorView<T, FILTER_ROWS, FILTER_COLS, INPUT_CHANS> =
            input.view((i, j), 0, options.view_padding, options.strides);
        array::from_fn(|c| {
            let mut x = i32::MIN;
            view.buffer.iter().zip(view.mask.iter()).for_each(|(v, m)| {
                if *m {
                    x = x.max(i32::from_subset(&v[c]));
                }
            });
            let y = T::from_superset_unchecked(&x);
            match options.fused_activation {
                FusedActivation::None => y,
                FusedActivation::Relu => relu(y, output_zero_point[0]),
                FusedActivation::Relu6 => relu6(y, output_scale[0], output_zero_point[0]),
            }
        })
    })];
    Tensor4D::new(output, output_scale, output_zero_point)
}

#[cfg(test)]
mod tests {
    use nalgebra::matrix;

    use super::*;

    #[test]
    fn max_pool_2d_layer() {
        let input: Tensor4D<i8, 1, 2, 3, 2, 1> = Tensor4D::new(
            [matrix![
                [1, 2], [3, 4],  [5, 6];
                [7, 8], [9, 10], [11, 12]
            ]],
            [0.13],
            [0],
        );
        let output: Tensor4D<i8, 1, 2, 3, 2, 1> = max_pool_2d(
            input,
            (Const::<2>, Const::<3>),
            [0.13],
            [0],
            MaxPool2DOptions {
                fused_activation: FusedActivation::None,
                view_padding: TensorViewPadding::Same,
                strides: (1, 1),
            },
        );
        assert_eq!(
            output.buffer[0],
            matrix![
                [9, 10],  [11, 12], [11, 12];
                [9, 10], [11, 12], [11, 12]
            ]
        );
    }
}
