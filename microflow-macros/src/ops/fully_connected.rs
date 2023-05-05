use flatbuffers::{ForwardsUOffset, Vector};
use nalgebra::{convert_ref, dmatrix, DMatrix};
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};

use crate::activation::TokenFusedActivation;
use crate::buffer::TokenBuffer2D;
use crate::tensor::TokenTensor2D;
use crate::tflite_flatbuffers::tflite::{Buffer, Operator, Tensor};

pub(crate) struct FullyConnected {
    pub(crate) weights: TokenTensor2D<i8>,
    pub(crate) output: TokenTensor2D<i8>,
    pub(crate) fused_activation: TokenFusedActivation,
    pub(crate) constants: (i8, TokenBuffer2D<f32>, f32, TokenBuffer2D<i32>, i32),
    pub(crate) capacity: Option<usize>,
}

impl FullyConnected {
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
        input: &TokenTensor2D<i8>,
        weights: &TokenTensor2D<i8>,
        biases: &TokenTensor2D<i32>,
        output: &TokenTensor2D<i8>,
    ) -> (i8, TokenBuffer2D<f32>, f32, TokenBuffer2D<i32>, i32) {
        (
            output.zero_point,
            (biases.scale / output.scale
                * biases.buffer.add_scalar(-biases.zero_point).cast::<f32>())
            .into(),
            input.scale * weights.scale / output.scale,
            DMatrix::from_rows(&[(input.zero_point as i32
                * convert_ref::<DMatrix<i8>, DMatrix<i32>>(&weights.buffer).row_sum())])
            .into(),
            input.shape[1] as i32 * input.zero_point as i32 * weights.zero_point as i32,
        )
    }
}

impl ToTokens for FullyConnected {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let weights = &self.weights;
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        let fused_activation = &self.fused_activation;
        let (constant_0, constant_1, constant_2, constant_3, constant_4) = &self.constants;

        let output = if self.capacity.is_some() && self.capacity.unwrap() < weights.buffer.nrows() {
            let weights_vec: Vec<_> = weights
                .buffer
                .column_iter()
                .map(|c| TokenTensor2D {
                    buffer: DMatrix::from_iterator(c.nrows(), c.ncols(), c.iter().copied()).into(),
                    shape: vec![c.nrows(), c.ncols()],
                    scale: weights.scale,
                    zero_point: weights.zero_point,
                })
                .collect();
            let constant_1_vec: Vec<_> = constant_1
                .iter()
                .map(|c| TokenBuffer2D::from(dmatrix![*c]))
                .collect();
            let constant_3_vec: Vec<_> = constant_3
                .iter()
                .map(|c| TokenBuffer2D::from(dmatrix![*c]))
                .collect();
            quote! {
                let output = microflow::tensor::QuantizedTensor2D::new(
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
                                    (#constant_0, #constant_1_vec, #constant_2, #constant_3_vec, #constant_4),
                                ).buffer
                            ),*
                        ]
                    ),
                    #output_scale,
                    #output_zero_point,
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
                    (#constant_0, #constant_1, #constant_2, #constant_3, #constant_4)
                );
            }
        };
        output.to_tokens(tokens);
    }
}

// TODO: Unit tests
