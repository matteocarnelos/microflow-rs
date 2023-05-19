use flatbuffers::{ForwardsUOffset, Vector};
use nalgebra::{convert_ref, dmatrix, DMatrix};
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};
use simba::scalar::SupersetOf;

use crate::activation::TokenFusedActivation;
use crate::buffer::TokenBuffer2D;
use crate::quantize::TokenQuantized;
use crate::tensor::TokenTensor2D;
use crate::tflite_flatbuffers::tflite::{Buffer, Operator, Tensor, TensorType};

pub(crate) struct FullyConnected<T1: TokenQuantized, T2: TokenQuantized> {
    pub(crate) weights: TokenTensor2D<T1>,
    pub(crate) output: TokenTensor2D<T1>,
    pub(crate) fused_activation: TokenFusedActivation,
    pub(crate) constants: (TokenBuffer2D<f32>, f32, TokenBuffer2D<T2>, T2),
    pub(crate) capacity: Option<usize>,
}

pub(crate) fn parse(operator: Operator, tensors: Vector<ForwardsUOffset<Tensor>>, buffers: Vector<ForwardsUOffset<Buffer>>, capacity: Option<usize>) -> Box<dyn ToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    let input = match input_type {
        TensorType::INT8 => TokenTensor2D::<i8>::from_empty_tensor(tensors.get(inputs.get(0) as usize)),
        TensorType::UINT8 => TokenTensor2D::<u8>::from_empty_tensor(tensors.get(inputs.get(0) as usize))
    };
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
    Box::new(FullyConnected::<i8, i32>::new(operator, tensors, buffers, capacity))
}

impl<T1: TokenQuantized, T2: TokenQuantized> FullyConnected<T1, T2>
where
    T2: SupersetOf<T1>,
{
    pub(crate) fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
        buffers: Vector<ForwardsUOffset<Buffer>>,
        capacity: Option<usize>,
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
            fused_activation: TokenFusedActivation(options.fused_activation_function()),
            constants,
            capacity,
        }
    }

    fn preprocess(
        input: &TokenTensor2D<T1>,
        weights: &TokenTensor2D<T1>,
        biases: &TokenTensor2D<T2>,
        output: &TokenTensor2D<T1>,
    ) -> (TokenBuffer2D<f32>, f32, TokenBuffer2D<T2>, T2) {
        (
            TokenBuffer2D::from(
                biases.scale / output.scale
                    * biases.buffer.map::<T2, _>(|e| e - biases.zero_point).cast::<f32>(),
            ),
            input.scale * weights.scale / output.scale,
            TokenBuffer2D::from(DMatrix::from_rows(&[
                convert_ref::<DMatrix<T1>, DMatrix<T2>>(&weights.buffer).row_sum()
                    * T2::from_subset(&input.zero_point),
            ])),
            T2::from_subset(&input.shape[1])
                * T2::from_subset(&input.zero_point)
                * T2::from_subset(&weights.zero_point),
        )
    }
}

impl<T1: TokenQuantized, T2: TokenQuantized> ToTokens for FullyConnected<T1, T2> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let weights = &self.weights;
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        let fused_activation = &self.fused_activation;
        let (constant_0, constant_1, constant_2, constant_3) = &self.constants;

        let output = if self.capacity.is_some() && self.capacity.unwrap() < weights.buffer.nrows() {
            let weights_vec: Vec<_> = weights
                .buffer
                .column_iter()
                .map(|c| TokenTensor2D {
                    buffer: TokenBuffer2D::from(DMatrix::from_iterator(
                        c.nrows(),
                        c.ncols(),
                        c.iter().copied(),
                    )),
                    shape: vec![c.nrows(), c.ncols()],
                    scale: weights.scale,
                    zero_point: weights.zero_point,
                })
                .collect();
            let constant_0_vec: Vec<_> = constant_0
                .iter()
                .map(|c| TokenBuffer2D::from(dmatrix![*c]))
                .collect();
            let constant_2_vec: Vec<_> = constant_2
                .iter()
                .map(|c| TokenBuffer2D::from(dmatrix![*c]))
                .collect();
            quote! {
                let output = microflow::tensor::Tensor2D::new(
                    microflow::buffer::Buffer2D::from_columns(
                        &[
                            #(
                                microflow::ops::fully_connected(
                                    &output.into(),
                                    #weights_vec,
                                    #output_scale,
                                    #output_zero_point,
                                    microflow::ops::FullyConnectedOptions {
                                        fused_activation: #fused_activation
                                    },
                                    (#constant_0_vec, #constant_1, #constant_2_vec, #constant_3)
                                ).buffer
                            ),*
                        ]
                    ),
                    #output_scale,
                    #output_zero_point
                );
            }
        } else {
            quote! {
                let output = microflow::ops::fully_connected(
                    &output.into(),
                    #weights,
                    #output_scale,
                    #output_zero_point,
                    microflow::ops::FullyConnectedOptions {
                        fused_activation: #fused_activation
                    },
                    (#constant_0, #constant_1, #constant_2, #constant_3)
                );
            }
        };
        output.to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tflite_flatbuffers::tflite::ActivationFunctionType;
    use nalgebra::dmatrix;

    #[test]
    fn fully_connected_preprocess() {
        let input = TokenTensor2D {
            buffer: TokenBuffer2D::new(),
            shape: vec![1, 2],
            scale: 0.1,
            zero_point: 2,
        };
        let weights = TokenTensor2D {
            buffer: TokenBuffer2D::from(dmatrix![
                3, 4, 5;
                6, 7, 8
            ]),
            shape: vec![2, 3],
            scale: 0.9,
            zero_point: 10,
        };
        let biases = TokenTensor2D {
            buffer: TokenBuffer2D::from(dmatrix![
                11; 12; 13
            ]),
            shape: vec![1, 3],
            scale: 0.14,
            zero_point: 15,
        };
        let output = TokenTensor2D {
            buffer: TokenBuffer2D::new(),
            shape: vec![1, 3],
            scale: 0.16,
            zero_point: 17,
        };
        let constants = FullyConnected::preprocess(&input, &weights, &biases, &output);
        assert_eq!(constants.0 .0, Some(dmatrix![-3.5; -2.625; -1.75]));
        assert_eq!(constants.1, 0.5625);
        assert_eq!(constants.2 .0, Some(dmatrix![18, 22, 26]));
        assert_eq!(constants.3, 40);
    }

    #[test]
    fn fully_connected_to_tokens() {
        let layer = FullyConnected {
            weights: TokenTensor2D {
                buffer: TokenBuffer2D::from(dmatrix![
                    1i8, 2i8, 3i8;
                    4i8, 5i8, 6i8
                ]),
                shape: vec![2, 3],
                scale: 0.7,
                zero_point: 8,
            },
            output: TokenTensor2D {
                buffer: TokenBuffer2D::new(),
                shape: vec![2, 2],
                scale: 0.9,
                zero_point: 10,
            },
            fused_activation: TokenFusedActivation(ActivationFunctionType::RELU),
            constants: (
                TokenBuffer2D::from(dmatrix![11., 12.]),
                13.,
                TokenBuffer2D::from(dmatrix![14, 15]),
                16,
            ),
            capacity: None,
        };
        let weights = &layer.weights;
        let fused_activation = &layer.fused_activation;
        let constants_0 = &layer.constants.0;
        let constants_2 = &layer.constants.2;
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                let output = microflow::ops::fully_connected(
                    &output.into(),
                    #weights,
                    0.9f32,
                    10i8,
                    microflow::ops::FullyConnectedOptions {
                        fused_activation: #fused_activation
                    },
                    (#constants_0, 13f32, #constants_2, 16i32)
                );
            }
            .to_string()
        );
    }

    #[test]
    fn fully_connected_with_capactiy_to_tokens() {
        let layer = FullyConnected {
            weights: TokenTensor2D {
                buffer: TokenBuffer2D::from(dmatrix![
                    1i8, 2i8;
                    3i8, 4i8
                ]),
                shape: vec![2, 2],
                scale: 0.5,
                zero_point: 6,
            },
            output: TokenTensor2D {
                buffer: TokenBuffer2D::new(),
                shape: vec![2, 2],
                scale: 0.7,
                zero_point: 8,
            },
            fused_activation: TokenFusedActivation(ActivationFunctionType::RELU),
            constants: (
                TokenBuffer2D::from(dmatrix![9., 10.]),
                11.,
                TokenBuffer2D::from(dmatrix![12, 13]),
                14,
            ),
            capacity: Some(1),
        };
        let weights_0 = TokenTensor2D {
            buffer: TokenBuffer2D::from(dmatrix![1i8; 3i8]),
            shape: vec![2, 1],
            scale: 0.5,
            zero_point: 6,
        };
        let weights_1 = TokenTensor2D {
            buffer: TokenBuffer2D::from(dmatrix![2i8; 4i8]),
            shape: vec![2, 1],
            scale: 0.5,
            zero_point: 6,
        };
        let fused_activation = &layer.fused_activation;
        let constants_0_0 = TokenBuffer2D::from(dmatrix![9f32]);
        let constants_0_1 = TokenBuffer2D::from(dmatrix![10f32]);
        let constants_2_0 = TokenBuffer2D::from(dmatrix![12i32]);
        let constants_2_1 = TokenBuffer2D::from(dmatrix![13i32]);
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                let output = microflow::tensor::Tensor2D::new(
                    microflow::buffer::Buffer2D::from_columns(
                        &[
                            microflow::ops::fully_connected(
                                &output.into(),
                                #weights_0,
                                0.7f32,
                                8i8,
                                microflow::ops::FullyConnectedOptions {
                                    fused_activation: #fused_activation
                                },
                                (#constants_0_0, 11f32, #constants_2_0, 14i32)
                            ).buffer,
                            microflow::ops::fully_connected(
                                &output.into(),
                                #weights_1,
                                0.7f32,
                                8i8,
                                microflow::ops::FullyConnectedOptions {
                                    fused_activation: #fused_activation
                                },
                                (#constants_0_1, 11f32, #constants_2_1, 14i32)
                            ).buffer
                        ]
                    ),
                    0.7f32,
                    8i8
                );
            }
            .to_string()
        );
    }
}
