use flatbuffers::{ForwardsUOffset, Vector};
use nalgebra::{convert_ref, DMatrix};
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};
use simba::scalar::SupersetOf;

use crate::activation::TokenFusedActivation;
use crate::buffer::TokenBuffer2D;
use crate::quantize::TokenQuantized;
use crate::tensor::TokenTensor2D;
use crate::tflite_flatbuffers::tflite::{Buffer, Operator, Tensor, TensorType};

pub(crate) struct TokenFullyConnected<T: TokenQuantized> {
    pub(crate) input: TokenTensor2D<T>,
    pub(crate) weights: TokenTensor2D<T>,
    pub(crate) output: TokenTensor2D<T>,
    pub(crate) fused_activation: TokenFusedActivation,
    pub(crate) constants: (TokenBuffer2D<f32>, f32, TokenBuffer2D<i32>, i32),
}

pub(crate) fn parse(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
    buffers: Vector<ForwardsUOffset<Buffer>>,
) -> Box<dyn ToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(TokenFullyConnected::<i8>::new(operator, tensors, buffers)),
        TensorType::UINT8 => Box::new(TokenFullyConnected::<u8>::new(operator, tensors, buffers)),
        _ => unimplemented!(),
    }
}

impl<T: TokenQuantized> TokenFullyConnected<T> {
    pub(crate) fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
        buffers: Vector<ForwardsUOffset<Buffer>>,
    ) -> Self {
        let inputs = operator.inputs().unwrap();
        let input = TokenTensor2D::from_empty_tensor(tensors.get(inputs.get(0) as usize));
        let weights =
            TokenTensor2D::from_buffered_tensor(tensors.get(inputs.get(1) as usize), buffers);
        let biases =
            TokenTensor2D::from_buffered_tensor(tensors.get(inputs.get(2) as usize), buffers);
        let output = TokenTensor2D::from_empty_tensor(
            tensors.get(operator.outputs().unwrap().get(0) as usize),
        );
        let options = operator
            .builtin_options_as_fully_connected_options()
            .unwrap();
        let constants = Self::preprocess(&input, &weights, &biases, &output);
        Self {
            input,
            weights,
            output,
            fused_activation: options.fused_activation_function().into(),
            constants,
        }
    }

    fn preprocess(
        input: &TokenTensor2D<T>,
        weights: &TokenTensor2D<T>,
        biases: &TokenTensor2D<i32>,
        output: &TokenTensor2D<T>,
    ) -> (TokenBuffer2D<f32>, f32, TokenBuffer2D<i32>, i32) {
        (
            TokenBuffer2D::from(
                biases.scale[0] / output.scale[0]
                    * biases
                        .buffer
                        .add_scalar(-biases.zero_point[0])
                        .cast::<f32>(),
            ),
            input.scale[0] * weights.scale[0] / output.scale[0],
            TokenBuffer2D::from(DMatrix::from_rows(&[
                convert_ref::<DMatrix<T>, DMatrix<i32>>(&weights.buffer).row_sum()
                    * i32::from_subset(&input.zero_point[0]),
            ])),
            input.shape[1] as i32
                * i32::from_subset(&input.zero_point[0])
                * i32::from_subset(&weights.zero_point[0]),
        )
    }
}

impl<T: TokenQuantized> ToTokens for TokenFullyConnected<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let input_shape = &self.input.shape;
        let output_shape = &self.output.shape;
        let weights = &self.weights;
        let output_scale = self.output.scale[0];
        let output_zero_point = self.output.zero_point[0];
        let fused_activation = self.fused_activation;
        let (constants_0, constants_1, constants_2, constants_3) = &self.constants;

        let output = quote! {
            let input: microflow::tensor::Tensor2D<_, #(#input_shape),*, 1usize> = input.into();
            let input: microflow::tensor::Tensor2D<_, #(#output_shape),*, 1usize> = microflow::ops::fully_connected(
                input,
                #weights,
                [#output_scale],
                [#output_zero_point],
                microflow::ops::FullyConnectedOptions {
                    fused_activation: #fused_activation,
                },
                (#constants_0, #constants_1, #constants_2, #constants_3)
            );
        };
        output.to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::dmatrix;

    use super::*;

    fn setup() -> TokenFullyConnected<i8> {
        TokenFullyConnected {
            input: TokenTensor2D {
                buffer: TokenBuffer2D::new(),
                shape: vec![1, 2],
                scale: vec![0.1],
                zero_point: vec![2],
            },
            weights: TokenTensor2D {
                buffer: TokenBuffer2D::from(dmatrix![
                    3, 4, 5;
                    6, 7, 8
                ]),
                shape: vec![2, 3],
                scale: vec![0.9],
                zero_point: vec![10],
            },
            output: TokenTensor2D {
                buffer: TokenBuffer2D::new(),
                shape: vec![1, 3],
                scale: vec![0.11],
                zero_point: vec![12],
            },
            fused_activation: TokenFusedActivation::Relu,
            constants: (
                TokenBuffer2D::from(dmatrix![13., 14.]),
                15.,
                TokenBuffer2D::from(dmatrix![16, 17]),
                18,
            ),
        }
    }

    #[test]
    fn fully_connected_preprocess() {
        let layer = setup();
        let biases = TokenTensor2D {
            buffer: TokenBuffer2D::from(dmatrix![
                19;
                20;
                21
            ]),
            shape: vec![3, 1],
            scale: vec![0.22],
            zero_point: vec![23],
        };
        let constants =
            TokenFullyConnected::preprocess(&layer.input, &layer.weights, &biases, &layer.output);
        assert_eq!(constants.0 .0, Some(dmatrix![-8.0; -6.0; -4.0]));
        assert_eq!(constants.1, 0.8181818);
        assert_eq!(constants.2 .0, Some(dmatrix![18, 22, 26]));
        assert_eq!(constants.3, 40);
    }

    #[test]
    fn fully_connected_to_tokens() {
        let layer = setup();
        let weights = &layer.weights;
        let fused_activation = layer.fused_activation;
        let constants_0 = &layer.constants.0;
        let constants_2 = &layer.constants.2;
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                let input: microflow::tensor::Tensor2D<_, 1usize, 2usize, 1usize> = input.into();
                let input: microflow::tensor::Tensor2D<_, 1usize, 3usize, 1usize> = microflow::ops::fully_connected(
                    input,
                    #weights,
                    [0.11f32],
                    [12i8],
                    microflow::ops::FullyConnectedOptions {
                        fused_activation: #fused_activation,
                    },
                    (#constants_0, 15f32, #constants_2, 18i32)
                );
            }
            .to_string()
        );
    }
}
