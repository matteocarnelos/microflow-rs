use core::array;
use libm::{floorf, roundf};

use simba::scalar::SupersetOf;

use crate::buffer::Buffer2D;
use crate::quantize::Quantized;
use crate::tensor::Tensor4D;

pub struct ResizeNearestNeighborOptions {
    pub align_corners: bool,
    pub half_pixel_centers: bool,
}

/// Performs the ResizeNearestNeighbor operation.
/// Returns a 4-dimensional output tensor containing the result of the operation.
///
/// # Arguments
/// * `input` - The 4-dimensional input tensor
/// * `output_scale` - The scale of the resulting output tensor
/// * `output_zero_point` - The zero point of the resulting output tensor
/// * `options` - Operator's options as a [`ResizeNearestNeighborOptions`] struct
/// * `constants` - Constant values coming from the pre-processing phase
///
pub fn resize_nearest_neighbor<
    T: Quantized,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
>(
    input: Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
    options: ResizeNearestNeighborOptions,
    constants: (f32, f32),
) -> Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, INPUT_CHANS, 1> {
    let output = [Buffer2D::from_fn(|i, j| {
        // Map the output coordinates to the nearest input ones
        let y = input_index(i, INPUT_ROWS, OUTPUT_ROWS, &options);
        let x = input_index(j, INPUT_COLS, OUTPUT_COLS, &options);
        // Requantize each channel
        array::from_fn(|c| {
            let z = f32::from_subset(&input.buffer[0][(y, x)][c]);
            T::from_superset_unchecked(&roundf(constants.0 * z + constants.1))
        })
    })];
    Tensor4D::new(output, output_scale, output_zero_point)
}

/// Maps an output index to the nearest input index, clamped to the input bounds.
///
/// # Arguments
/// * `index` - The index along the output dimension
/// * `input_len` - The length of the input dimension
/// * `output_len` - The length of the output dimension
/// * `options` - Operator's options as a [`ResizeNearestNeighborOptions`] struct
///
fn input_index(
    index: usize,
    input_len: usize,
    output_len: usize,
    options: &ResizeNearestNeighborOptions,
) -> usize {
    let scale = if options.align_corners && output_len > 1 {
        (input_len - 1) as f32 / (output_len - 1) as f32
    } else {
        input_len as f32 / output_len as f32
    };
    let offset = if options.half_pixel_centers { 0.5 } else { 0. };
    let coord = (index as f32 + offset) * scale;
    let nearest = if options.align_corners {
        roundf(coord)
    } else {
        floorf(coord)
    };
    (nearest as usize).min(input_len - 1)
}

#[cfg(test)]
mod tests {
    use nalgebra::matrix;

    use super::*;

    const INPUT: Tensor4D<i8, 1, 2, 2, 1, 1> = Tensor4D {
        buffer: [matrix![
            [10], [20];
            [30], [40]
        ]],
        scale: [1.],
        zero_point: [0],
    };
    const OUTPUT_SCALE: [f32; 1] = [1.];
    const OUTPUT_ZERO_POINT: [i8; 1] = [0];
    const OPTIONS: ResizeNearestNeighborOptions = ResizeNearestNeighborOptions {
        align_corners: false,
        half_pixel_centers: false,
    };
    const CONSTANTS: (f32, f32) = (1., 0.);

    #[test]
    fn upscale() {
        let output: Tensor4D<i8, 1, 4, 4, 1, 1> = Tensor4D {
            buffer: [matrix![
                [10], [10], [20], [20];
                [10], [10], [20], [20];
                [30], [30], [40], [40];
                [30], [30], [40], [40]
            ]],
            scale: [1.],
            zero_point: [0],
        };
        assert_eq!(
            resize_nearest_neighbor(INPUT, OUTPUT_SCALE, OUTPUT_ZERO_POINT, OPTIONS, CONSTANTS),
            output
        );
    }

    #[test]
    fn downscale() {
        let input: Tensor4D<i8, 1, 4, 4, 1, 1> = Tensor4D {
            buffer: [matrix![
                [10],  [20],  [30],  [40];
                [50],  [60],  [70],  [80];
                [90],  [100], [110], [120];
                [127], [127], [127], [127]
            ]],
            scale: [1.],
            zero_point: [0],
        };
        let output: Tensor4D<i8, 1, 2, 2, 1, 1> = Tensor4D {
            buffer: [matrix![
                [10], [30];
                [90], [110]
            ]],
            scale: [1.],
            zero_point: [0],
        };
        assert_eq!(
            resize_nearest_neighbor(input, OUTPUT_SCALE, OUTPUT_ZERO_POINT, OPTIONS, CONSTANTS),
            output
        );
    }

    #[test]
    fn align_corners() {
        let options = ResizeNearestNeighborOptions {
            align_corners: true,
            half_pixel_centers: false,
        };
        let output: Tensor4D<i8, 1, 3, 3, 1, 1> = Tensor4D {
            buffer: [matrix![
                [10], [20], [20];
                [30], [40], [40];
                [30], [40], [40]
            ]],
            scale: [1.],
            zero_point: [0],
        };
        assert_eq!(
            resize_nearest_neighbor(INPUT, OUTPUT_SCALE, OUTPUT_ZERO_POINT, options, CONSTANTS),
            output
        );
    }

    #[test]
    fn half_pixel_centers() {
        let options = ResizeNearestNeighborOptions {
            align_corners: false,
            half_pixel_centers: true,
        };
        let output: Tensor4D<i8, 1, 3, 3, 1, 1> = Tensor4D {
            buffer: [matrix![
                [10], [20], [20];
                [30], [40], [40];
                [30], [40], [40]
            ]],
            scale: [1.],
            zero_point: [0],
        };
        assert_eq!(
            resize_nearest_neighbor(INPUT, OUTPUT_SCALE, OUTPUT_ZERO_POINT, options, CONSTANTS),
            output
        );
    }

    #[test]
    fn multi_channel() {
        let input: Tensor4D<i8, 1, 2, 2, 2, 1> = Tensor4D {
            buffer: [matrix![
                [10, 1], [20, 2];
                [30, 3], [40, 4]
            ]],
            scale: [1.],
            zero_point: [0],
        };
        let output: Tensor4D<i8, 1, 3, 3, 2, 1> = Tensor4D {
            buffer: [matrix![
                [10, 1], [10, 1], [20, 2];
                [10, 1], [10, 1], [20, 2];
                [30, 3], [30, 3], [40, 4]
            ]],
            scale: [1.],
            zero_point: [0],
        };
        assert_eq!(
            resize_nearest_neighbor(input, OUTPUT_SCALE, OUTPUT_ZERO_POINT, OPTIONS, CONSTANTS),
            output
        );
    }
}
