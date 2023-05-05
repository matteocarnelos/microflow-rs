use crate::activation;
use crate::tensor::QuantizedTensor2D;
use libm::expf;

pub fn softmax<const D1: usize, const D2: usize>(
    input: QuantizedTensor2D<i8, D1, D2>,
    output_scale: f32,
    output_zero_point: i8,
) -> QuantizedTensor2D<i8, D1, D2> {
    let exp = input.buffer.map(|e| e as f32 * input.scale);
    let sum = exp.map(expf).sum();
    QuantizedTensor2D::new(
        exp.map(|e| activation::softmax(e, sum, output_scale, output_zero_point)),
        output_scale,
        output_zero_point,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::matrix;

    const INPUT_TENSOR: QuantizedTensor2D<i8, 2, 3> = QuantizedTensor2D {
        buffer: matrix![
            1, 2, 3;
            4, 5, 6
        ],
        scale: 0.7,
        zero_point: 8,
    };
    const OUTPUT_SCALE: f32 = 0.9;
    const OUTPUT_ZERO_POINT: i8 = 10;
    const OUTPUT_TENSOR: QuantizedTensor2D<i8, 2, 3> = QuantizedTensor2D {
        buffer: matrix![
            10, 10, 10;
            10, 10, 11
        ],
        scale: OUTPUT_SCALE,
        zero_point: OUTPUT_ZERO_POINT,
    };

    #[test]
    fn softmax_layer() {
        assert_eq!(
            softmax(INPUT_TENSOR, OUTPUT_SCALE, OUTPUT_ZERO_POINT),
            OUTPUT_TENSOR
        );
    }
}
