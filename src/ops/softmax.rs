use crate::activation;
use crate::quantize::Quantized;
use crate::tensor::Tensor2D;
use libm::expf;
use simba::scalar::SupersetOf;

/// Performs the Softmax activation function as an operator.
/// Returns a 2-dimensional output tensor containing the result of the operation.
///
/// # Arguments
/// * `input` - The 2-dimensional input tensor
/// * `output_scale` - The scale of the resulting output tensor
/// * `output_zero_point` - The zero point of the resulting output tensor
///
pub fn softmax<T: Quantized, const ROWS: usize, const COLS: usize>(
    input: Tensor2D<T, ROWS, COLS, 1>,
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
) -> Tensor2D<T, ROWS, COLS, 1> {
    let exp = input.buffer.map(|e| f32::from_subset(&e) * input.scale[0]);
    let sum = exp.map(expf).sum();
    Tensor2D::new(
        exp.map(|e| activation::softmax(e, sum, output_scale[0], output_zero_point[0])),
        output_scale,
        output_zero_point,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::matrix;

    const INPUT: Tensor2D<i8, 2, 3, 1> = Tensor2D {
        buffer: matrix![
            1, 2, 3;
            4, 5, 6
        ],
        scale: [0.7],
        zero_point: [8],
    };
    const OUTPUT_SCALE: [f32; 1] = [0.9];
    const OUTPUT_ZERO_POINT: [i8; 1] = [10];
    const OUTPUT: Tensor2D<i8, 2, 3, 1> = Tensor2D {
        buffer: matrix![
            10, 10, 10;
            10, 10, 11
        ],
        scale: OUTPUT_SCALE,
        zero_point: OUTPUT_ZERO_POINT,
    };

    #[test]
    fn softmax_layer() {
        assert_eq!(softmax(INPUT, OUTPUT_SCALE, OUTPUT_ZERO_POINT), OUTPUT);
    }
}
