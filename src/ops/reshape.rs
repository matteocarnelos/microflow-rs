/// Performs the Reshape operator.
/// Returns the correspondig output tensor.
pub fn reshape<InputT, OutputT>(input: InputT) -> OutputT
where
    InputT: Into<OutputT>,
{
    input.into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Tensor2D, Tensor4D};
    use nalgebra::matrix;

    const INPUT: Tensor2D<i8, 2, 3, 1> = Tensor2D {
        buffer: matrix![
            1, 2, 3;
            4, 5, 6
        ],
        scale: [0.7],
        zero_point: [8],
    };
    const OUTPUT: Tensor4D<i8, 2, 1, 3, 1, 1> = Tensor4D {
        buffer: [matrix![[1], [2], [3]], matrix![[4], [5], [6]]],
        scale: [0.7],
        zero_point: [8],
    };

    #[test]
    fn reshape_layer() {
        let output: Tensor4D<i8, 2, 1, 3, 1, 1> = reshape(INPUT);
        assert_eq!(output, OUTPUT);
    }
}
