use crate::buffer::Buffer2D;
use crate::quantize::Quantized;
use crate::tensor::Tensor4D;
use libm::{floorf, roundf};
use simba::scalar::SupersetOf;

#[derive(Clone, Copy)]
pub struct ResizeBilinearOptions {
    pub align_corners: bool,
    pub half_pixel_centers: bool,
}

/// Performs the ResizeBilinear operation.
/// Returns a 4-dimensional output tensor containing the resized input.
///
/// # Arguments
/// * `input` - The 4-dimensional input tensor passed by value.
/// * `output_scale` - The scale of the resulting output tensor.
/// * `output_zero_point` - The zero point of the resulting output tensor.
/// * `options` - Operator's options as a [`ResizeBilinearOptions`] struct.
/// * `constants` - A tuple of pre-calculated f32 values for requantization.
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
    let output = [Buffer2D::from_fn(|r, c| {

        // Calculate the corresponding row and column in the input tensor
        let in_r = if options.half_pixel_centers {
            ((r as f32 + 0.5) * (INPUT_ROWS as f32 / OUTPUT_ROWS as f32)) - 0.5
        } else if options.align_corners && OUTPUT_ROWS > 1 {
            r as f32 * (INPUT_ROWS - 1) as f32 / (OUTPUT_ROWS - 1) as f32
        } else {
            r as f32 * INPUT_ROWS as f32 / OUTPUT_ROWS as f32
        };
        
        let in_c = if options.half_pixel_centers {
            ((c as f32 + 0.5) * (INPUT_COLS as f32 / OUTPUT_COLS as f32)) - 0.5
        } else if options.align_corners && OUTPUT_COLS > 1 {
            c as f32 * (INPUT_COLS - 1) as f32 / (OUTPUT_COLS - 1) as f32
        } else {
            c as f32 * INPUT_COLS as f32 / OUTPUT_COLS as f32
        };

        // Squeeze the coordinates to be within the input tensor's bounds
        let in_r_squeezed = in_r.max(0.).min(INPUT_ROWS as f32 - 1.);
        let in_c_squeezed = in_c.max(0.).min(INPUT_COLS as f32 - 1.);

        // Get the four nearest neighbors
        let r1 = floorf(in_r_squeezed) as usize;
        let c1 = floorf(in_c_squeezed) as usize;
        let r2 = (r1 + 1).min(INPUT_ROWS - 1);
        let c2 = (c1 + 1).min(INPUT_COLS - 1);

        // Calculate the interpolation weights
        let yr = in_r_squeezed - r1 as f32;
        let xr = in_c_squeezed - c1 as f32;

        let mut out_channels = [T::from_superset_unchecked(&0); INPUT_CHANS];
        for i in 0..INPUT_CHANS {
            let p1 = f32::from_subset(&input.buffer[0][(r1, c1)][i]);
            let p2 = f32::from_subset(&input.buffer[0][(r1, c2)][i]);
            let p3 = f32::from_subset(&input.buffer[0][(r2, c1)][i]);
            let p4 = f32::from_subset(&input.buffer[0][(r2, c2)][i]);

            // Perform the bilinear interpolation
            let out = p1 * (1. - xr) * (1. - yr)
                + p2 * xr * (1. - yr)
                + p3 * (1. - xr) * yr
                + p4 * xr * yr;
            
            // Requantize the output value
            let requantized = roundf(constants.0 * out + constants.1);

            // Validate for NaN/infinity and use safe conversion
            let clamped = if requantized.is_nan() || requantized.is_infinite() {
                0.0f32 // Default to zero for invalid results
            } else {
                requantized
            };

            // Use safe conversion with fallback to zero
            out_channels[i] = T::from_superset(&clamped).unwrap_or_else(|| T::from_superset_unchecked(&0.0f32));
        }
        out_channels
    })];
    Tensor4D::new(output, output_scale, output_zero_point)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::matrix;

    const INPUT: Tensor4D<i8, 1, 2, 2, 1, 1> = Tensor4D {
        buffer: [matrix![[10], [20]; [30], [40]]],
        scale: [1.0],
        zero_point: [0],
    };
    const OUTPUT_SCALE: [f32; 1] = [1.0];
    const OUTPUT_ZERO_POINT: [i8; 1] = [0];
    // For 1:1 scale and zero point, constants are (1.0, 0.0)
    const CONSTANTS: (f32, f32) = (1.0, 0.0);

    #[test]
    fn test_upscaling() {
        let options = ResizeBilinearOptions {
            align_corners: false,
            half_pixel_centers: false,
        };
        let output: Tensor4D<i8, 1, 4, 4, 1, 1> = resize_bilinear(
            INPUT,
            OUTPUT_SCALE,
            OUTPUT_ZERO_POINT,
            options,
            CONSTANTS,
        );
        let expected = [
            [10, 15, 20, 20],
            [20, 25, 30, 30],
            [30, 35, 40, 40],
            [30, 35, 40, 40],
        ];
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(output.buffer[0][(r, c)][0], expected[r][c]);
            }
        }
    }

    #[test]
    fn test_downscaling() {
        let input: Tensor4D<i8, 1, 4, 4, 1, 1> = Tensor4D {
            buffer: [matrix![
                [10], [20], [30], [40];
                [50], [60], [70], [80];
                [90], [100], [110], [120];
                [127], [127], [127], [127]
            ]],
            scale: [1.0],
            zero_point: [0],
        };
        let options = ResizeBilinearOptions {
            align_corners: false,
            half_pixel_centers: false,
        };
        let output: Tensor4D<i8, 1, 2, 2, 1, 1> = resize_bilinear(
            input,
            OUTPUT_SCALE,
            OUTPUT_ZERO_POINT,
            options,
            CONSTANTS,
        );
        assert_eq!(output.buffer[0][(0, 0)][0], 10);
        assert_eq!(output.buffer[0][(0, 1)][0], 30);
        assert_eq!(output.buffer[0][(1, 0)][0], 90);
        assert_eq!(output.buffer[0][(1, 1)][0], 110);
    }
    
    #[test]
    fn test_align_corners() {
        let options = ResizeBilinearOptions {
            align_corners: true,
            half_pixel_centers: false,
        };
        let output: Tensor4D<i8, 1, 3, 3, 1, 1> = resize_bilinear(
            INPUT,
            OUTPUT_SCALE,
            OUTPUT_ZERO_POINT,
            options,
            CONSTANTS,
        );
        let expected = [[10, 15, 20], [20, 25, 30], [30, 35, 40]];
        for r in 0..3 {
            for c in 0..3 {
                assert_eq!(output.buffer[0][(r, c)][0], expected[r][c]);
            }
        }
    }

    #[test]
    fn test_half_pixel_centers() {
        let options = ResizeBilinearOptions {
            align_corners: false,
            half_pixel_centers: true,
        };
        let output: Tensor4D<i8, 1, 3, 3, 1, 1> = resize_bilinear(
            INPUT,
            OUTPUT_SCALE,
            OUTPUT_ZERO_POINT,
            options,
            CONSTANTS,
        );
        // The center of the output (1,1) should map to the center of the input (0.5, 0.5)
        // This should be an average of all 4 input pixels: (10+20+30+40)/4 = 25
        assert_eq!(output.buffer[0][(1, 1)][0], 25);
    }
    
    #[test]
    fn test_multi_channel() {
        let input: Tensor4D<i8, 1, 2, 2, 2, 1> = Tensor4D {
            buffer: [matrix![
                [10, 1], [20, 2];
                [30, 3], [40, 4]
            ]],
            scale: [1.0],
            zero_point: [0],
        };
        let options = ResizeBilinearOptions {
            align_corners: false,
            half_pixel_centers: false,
        };
        let output: Tensor4D<i8, 1, 3, 3, 2, 1> = resize_bilinear(
            input,
            OUTPUT_SCALE,
            OUTPUT_ZERO_POINT,
            options,
            CONSTANTS,
        );
        // Check interpolated value for both channels at (1,1)
        // With default resize (no align_corners, no half_pixel_centers):
        // Position (1,1) maps to input coordinates (1.33, 1.33)
        // This gives more weight to the (1,1) corner than an equal average
        assert_eq!(output.buffer[0][(1, 1)][0], 30); // Actual interpolated value
        assert_eq!(output.buffer[0][(1, 1)][1], 3);  // Actual interpolated value
    }
}