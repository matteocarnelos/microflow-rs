use libm::roundf;
use simba::scalar::SupersetOf;

use crate::activation::{relu, relu6, FusedActivation};
use crate::buffer::Buffer2D;
use crate::quantize::Quantized;
use crate::tensor::Tensor2D;

// TODO: Performance evaluation (fine cast + iters vs bulk cast + nalgebra built-in)
// TODO: See if possible to implement paging here

pub struct FullyConnectedOptions {
    pub fused_activation: FusedActivation,
}

pub fn fully_connected<
    T: Quantized,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const WEIGHTS_COLS: usize,
>(
    input: &Tensor2D<T, INPUT_ROWS, INPUT_COLS, 1>,
    weights: Tensor2D<T, INPUT_COLS, WEIGHTS_COLS, 1>,
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
    options: FullyConnectedOptions,
    constants: (
        Buffer2D<f32, WEIGHTS_COLS, 1>,
        f32,
        Buffer2D<i32, 1, WEIGHTS_COLS>,
        i32,
    ),
) -> Tensor2D<T, INPUT_ROWS, WEIGHTS_COLS, 1> {
    let x = (
        input.buffer.cast::<i32>() * weights.buffer.cast::<i32>(),
        input.buffer.cast::<i32>().column_sum() * i32::from_subset(&weights.zero_point[0]),
    );
    let output = Buffer2D::from_fn(|i, j| {
        let y = T::from_superset_unchecked(&roundf(
            f32::from_subset(&output_zero_point[0])
                + constants.0[j]
                + constants.1
                    * f32::from_subset(&(x.0[(i, j)] - x.1[i] - constants.2[j] + constants.3)),
        ));
        match options.fused_activation {
            FusedActivation::NONE => y,
            FusedActivation::RELU => relu(y, output_zero_point[0]),
            FusedActivation::RELU6 => relu6(y, output_scale[0], output_zero_point[0]),
        }
    });
    Tensor2D::new(output, output_scale, output_zero_point)
}

#[cfg(test)]
mod tests {
    use nalgebra::matrix;

    use super::*;

    const INPUT: Tensor2D<i8, 2, 3, 1> = Tensor2D {
        buffer: matrix![
            1, 2, 3;
            4, 5, 6
        ],
        scale: [0.7],
        zero_point: [8],
    };
    const WEIGHTS: Tensor2D<i8, 3, 4, 1> = Tensor2D {
        buffer: matrix![
            9, 10, 11, 12;
            13, 14, 15, 16;
            17, 18, 19, 20
        ],
        scale: [0.21],
        zero_point: [22],
    };
    const _BIASES: Tensor2D<i32, 4, 1, 1> = Tensor2D {
        buffer: matrix![
            23; 24; 25; 26
        ],
        scale: [0.27],
        zero_point: [28],
    };
    const OUTPUT_SCALE: [f32; 1] = [0.29];
    const OUTPUT_ZERO_POINT: [i8; 1] = [30];
    const OPTIONS: FullyConnectedOptions = FullyConnectedOptions {
        fused_activation: FusedActivation::RELU,
    };
    const CONSTANTS: (Buffer2D<f32, 4, 1>, f32, Buffer2D<i32, 1, 4>, i32) = (
        matrix![-4.655_172_3; -3.724_138; -2.793_103_5; -1.862_069],
        0.506_896_56,
        matrix![312, 336, 360, 384],
        528,
    );
    const OUTPUT: Tensor2D<i8, 2, 4, 1> = Tensor2D {
        buffer: matrix![
            112, 103, 95, 87;
            70, 67, 63, 60
        ],
        scale: [0.29],
        zero_point: [30],
    };

    #[test]
    fn fully_connected_layer() {
        assert_eq!(
            fully_connected(
                &INPUT,
                WEIGHTS,
                OUTPUT_SCALE,
                OUTPUT_ZERO_POINT,
                OPTIONS,
                CONSTANTS
            ),
            OUTPUT
        )
    }
}
