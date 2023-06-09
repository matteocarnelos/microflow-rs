use flatbuffers::{ForwardsUOffset, Vector};
use nalgebra::{convert_ref, DMatrix};
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote, ToTokens};
use simba::scalar::SupersetOf;

use crate::activation::TokenFusedActivation;
use crate::buffer::TokenBuffer2D;
use crate::quantize::TokenQuantized;
use crate::tensor::TokenTensor2D;
use crate::tflite_flatbuffers::tflite::{Buffer, Operator, Tensor, TensorType};

pub(crate) struct TokenFullyConnected<T: TokenQuantized> {
    pub(crate) weights: TokenTensor2D<T>,
    pub(crate) output: TokenTensor2D<T>,
    pub(crate) fused_activation: TokenFusedActivation,
    pub(crate) constants: (TokenBuffer2D<f32>, f32, TokenBuffer2D<i32>, i32),
    pub(crate) index: usize,
    pub(crate) reshape: bool,
}

pub(crate) fn parse(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
    buffers: Vector<ForwardsUOffset<Buffer>>,
    index: usize,
) -> Box<dyn ToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(TokenFullyConnected::<i8>::new(
            operator, tensors, buffers, index,
        )),
        TensorType::UINT8 => Box::new(TokenFullyConnected::<u8>::new(
            operator, tensors, buffers, index,
        )),
        _ => unimplemented!(),
    }
}

impl<T: TokenQuantized> TokenFullyConnected<T> {
    pub(crate) fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
        buffers: Vector<ForwardsUOffset<Buffer>>,
        index: usize,
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
            weights,
            output,
            fused_activation: options.fused_activation_function().into(),
            reshape: input.shape.len() != 2,
            constants,
            index,
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
        let reshape = if self.reshape {
            quote!(.into())
        } else {
            quote!()
        };
        let weights_ident = format_ident!("weights_{}", self.index);
        let weights_type = self.weights.type_tokens();
        let weights = &self.weights;
        let output_shape = &self.output.shape;
        let output_scale = self.output.scale[0];
        let output_zero_point = self.output.zero_point[0];
        let fused_activation = self.fused_activation;
        let (constants_0, constants_1, constants_2, constants_3) = &self.constants;

        let ts = quote! {
            const #weights_ident: #weights_type = #weights;
            let input: microflow::tensor::Tensor2D<_, #(#output_shape),*, 1usize> =
                microflow::ops::fully_connected(
                    input #reshape,
                    &#weights_ident,
                    [#output_scale],
                    [#output_zero_point],
                    microflow::ops::FullyConnectedOptions {
                        fused_activation: #fused_activation,
                    },
                    (#constants_0, #constants_1, #constants_2, #constants_3)
            );
        };
        ts.to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::dmatrix;

    use super::*;

    fn setup() -> TokenFullyConnected<i8> {
        TokenFullyConnected {
            weights: TokenTensor2D {
                buffer: TokenBuffer2D::from(dmatrix![
                    1, 2, 3;
                    4, 5, 6
                ]),
                shape: vec![2, 3],
                scale: vec![0.7],
                zero_point: vec![8],
            },
            output: TokenTensor2D {
                buffer: TokenBuffer2D::new(),
                shape: vec![1, 3],
                scale: vec![0.9],
                zero_point: vec![10],
            },
            fused_activation: TokenFusedActivation::Relu,
            constants: (
                TokenBuffer2D::from(dmatrix![11., 12.]),
                13.,
                TokenBuffer2D::from(dmatrix![14, 15]),
                16,
            ),
            index: 0,
            reshape: false,
        }
    }

    #[test]
    fn fully_connected_preprocess() {
        let layer = setup();
        let input = TokenTensor2D {
            buffer: TokenBuffer2D::new(),
            shape: vec![1, 2],
            scale: vec![0.17],
            zero_point: vec![18],
        };
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
            TokenFullyConnected::preprocess(&input, &layer.weights, &biases, &layer.output);
        assert_eq!(
            constants.0 .0,
            Some(dmatrix![-0.9777778; -0.73333335; -0.4888889])
        );
        assert_eq!(constants.1, 0.13222224);
        assert_eq!(constants.2 .0, Some(dmatrix![90, 126, 162]));
        assert_eq!(constants.3, 288);
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
                const weights_0: microflow::tensor::Tensor2D<i8, 2usize, 3usize, 1usize> = #weights;
                let input: microflow::tensor::Tensor2D<_, 1usize, 3usize, 1usize> =
                    microflow::ops::fully_connected(
                        input,
                        &weights_0,
                        [0.9f32],
                        [10i8],
                        microflow::ops::FullyConnectedOptions {
                            fused_activation: #fused_activation,
                        },
                        (#constants_0, 13f32, #constants_2, 16i32)
                );
            }
            .to_string()
        );
    }
}
