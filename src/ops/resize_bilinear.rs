use core::array;
use libm::{floorf, roundf};

use simba::scalar::SupersetOf;

use crate::buffer::Buffer2D;
use crate::quantize::Quantized;
use crate::tensor::Tensor4D;

pub struct ResizeBilinearOptions {
    pub align_corners: bool,
    pub half_pixel_centers: bool,
}

/// Performs the ResizeBilinear operation.
/// Returns a 4-dimensional output tensor containing the result of the operation.
///
/// # Arguments
/// * `input` - The 4-dimensional input tensor
/// * `output_scale` - The scale of the resulting output tensor
/// * `output_zero_point` - The zero point of the resulting output tensor
/// * `options` - Operator's options as a [`ResizeBilinearOptions`] struct
/// * `constants` - Constant values coming from the pre-processing phase
///
pub fn resize_bilinear<
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
    options: ResizeBilinearOptions,
    constants: (f32, f32),
) -> Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, INPUT_CHANS, 1> {
    let output = [Buffer2D::from_fn(|i, j| {
        // Map the output coordinates to the input ones
        let y = input_coord(i, INPUT_ROWS, OUTPUT_ROWS, &options);
        let x = input_coord(j, INPUT_COLS, OUTPUT_COLS, &options);
        // Find the four nearest neighbors and their interpolation weights
        let (y0, x0) = (floorf(y) as usize, floorf(x) as usize);
        let (y1, x1) = ((y0 + 1).min(INPUT_ROWS - 1), (x0 + 1).min(INPUT_COLS - 1));
        let (dy, dx) = (y - y0 as f32, x - x0 as f32);
        // Interpolate each channel
        array::from_fn(|c| {
            let p = |i, j| f32::from_subset(&input.buffer[0][(i, j)][c]);
            let z = p(y0, x0) * (1. - dx) * (1. - dy)
                + p(y0, x1) * dx * (1. - dy)
                + p(y1, x0) * (1. - dx) * dy
                + p(y1, x1) * dx * dy;
            T::from_superset_unchecked(&roundf(constants.0 * z + constants.1))
        })
    })];
    Tensor4D::new(output, output_scale, output_zero_point)
}

/// Maps an output index to the corresponding input coordinate, clamped to the input bounds.
///
/// # Arguments
/// * `index` - The index along the output dimension
/// * `input_len` - The length of the input dimension
/// * `output_len` - The length of the output dimension
/// * `options` - Operator's options as a [`ResizeBilinearOptions`] struct
///
fn input_coord(
    index: usize,
    input_len: usize,
    output_len: usize,
    options: &ResizeBilinearOptions,
) -> f32 {
    let coord = if options.half_pixel_centers {
        (index as f32 + 0.5) * (input_len as f32 / output_len as f32) - 0.5
    } else if options.align_corners && output_len > 1 {
        index as f32 * (input_len - 1) as f32 / (output_len - 1) as f32
    } else {
        index as f32 * input_len as f32 / output_len as f32
    };
    coord.clamp(0., (input_len - 1) as f32)
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
    const OPTIONS: ResizeBilinearOptions = ResizeBilinearOptions {
        align_corners: false,
        half_pixel_centers: false,
    };
    const CONSTANTS: (f32, f32) = (1., 0.);

    #[test]
    fn upscale() {
        let output: Tensor4D<i8, 1, 4, 4, 1, 1> = Tensor4D {
            buffer: [matrix![
                [10], [15], [20], [20];
                [20], [25], [30], [30];
                [30], [35], [40], [40];
                [30], [35], [40], [40]
            ]],
            scale: [1.],
            zero_point: [0],
        };
        assert_eq!(
            resize_bilinear(INPUT, OUTPUT_SCALE, OUTPUT_ZERO_POINT, OPTIONS, CONSTANTS),
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
            resize_bilinear(input, OUTPUT_SCALE, OUTPUT_ZERO_POINT, OPTIONS, CONSTANTS),
            output
        );
    }

    #[test]
    fn align_corners() {
        let options = ResizeBilinearOptions {
            align_corners: true,
            half_pixel_centers: false,
        };
        let output: Tensor4D<i8, 1, 3, 3, 1, 1> = Tensor4D {
            buffer: [matrix![
                [10], [15], [20];
                [20], [25], [30];
                [30], [35], [40]
            ]],
            scale: [1.],
            zero_point: [0],
        };
        assert_eq!(
            resize_bilinear(INPUT, OUTPUT_SCALE, OUTPUT_ZERO_POINT, options, CONSTANTS),
            output
        );
    }

    #[test]
    fn half_pixel_centers() {
        let options = ResizeBilinearOptions {
            align_corners: false,
            half_pixel_centers: true,
        };
        let output: Tensor4D<i8, 1, 3, 3, 1, 1> = Tensor4D {
            buffer: [matrix![
                [10], [15], [20];
                [20], [25], [30];
                [30], [35], [40]
            ]],
            scale: [1.],
            zero_point: [0],
        };
        assert_eq!(
            resize_bilinear(INPUT, OUTPUT_SCALE, OUTPUT_ZERO_POINT, options, CONSTANTS),
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
                [10, 1], [17, 2], [20, 2];
                [23, 2], [30, 3], [33, 3];
                [30, 3], [37, 4], [40, 4]
            ]],
            scale: [1.],
            zero_point: [0],
        };
        assert_eq!(
            resize_bilinear(input, OUTPUT_SCALE, OUTPUT_ZERO_POINT, OPTIONS, CONSTANTS),
            output
        );
    }
}
