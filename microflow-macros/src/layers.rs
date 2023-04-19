use flatbuffers::{ForwardsUOffset, Vector};
use nalgebra::{convert_ref, dmatrix, DMatrix};
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};

use crate::matrix::TokenMatrix;
use crate::tensor::TokenTensor;
use crate::tflite_flatbuffers::tflite::{
    ActivationFunctionType, Buffer, BuiltinOperator, Operator, Tensor,
};

pub const SUPPORTED_OPS: [BuiltinOperator; 3] = [
    BuiltinOperator::QUANTIZE,
    BuiltinOperator::DEQUANTIZE,
    BuiltinOperator::FULLY_CONNECTED,
];

pub struct FullyConnected {
    pub(crate) weights: TokenTensor<i8>,
    pub(crate) output: TokenTensor<i8>,
    pub(crate) fused_activation: ActivationFunctionType,
    pub(crate) constants: (i8, TokenMatrix<f32>, f32, TokenMatrix<i32>, i32),
    pub(crate) capacity: Option<usize>,
}

impl FullyConnected {
    pub fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
        buffers: Vector<ForwardsUOffset<Buffer>>,
        capacity: Option<usize>,
    ) -> Self {
        let inputs = operator.inputs().unwrap();
        let input = tensors.get(inputs.get(0) as usize).into();
        let weights =
            TokenTensor::from_buffered_tensor(tensors.get(inputs.get(1) as usize), buffers);
        let biases =
            TokenTensor::from_buffered_tensor(tensors.get(inputs.get(2) as usize), buffers);
        let output = tensors
            .get(operator.outputs().unwrap().get(0) as usize)
            .into();
        let activation = operator
            .builtin_options_as_fully_connected_options()
            .unwrap()
            .fused_activation_function();
        let constants = Self::preprocess(&input, &weights, &biases, &output);
        Self {
            weights,
            output,
            fused_activation: activation,
            constants,
            capacity,
        }
    }

    pub fn preprocess(
        input: &TokenTensor<i8>,
        weights: &TokenTensor<i8>,
        biases: &TokenTensor<i32>,
        output: &TokenTensor<i8>,
    ) -> (i8, TokenMatrix<f32>, f32, TokenMatrix<i32>, i32) {
        (
            output.zero_point,
            (biases.scale / output.scale
                * biases.matrix.add_scalar(-biases.zero_point).cast::<f32>())
            .into(),
            input.scale * weights.scale / output.scale,
            DMatrix::from_rows(&[(input.zero_point as i32
                * convert_ref::<DMatrix<i8>, DMatrix<i32>>(&weights.matrix).row_sum())])
            .into(),
            input.matrix.shape().1 as i32 * input.zero_point as i32 * weights.zero_point as i32,
        )
    }
}

impl ToTokens for FullyConnected {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let weights = &self.weights;
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        let (constant_0, constant_1, constant_2, constant_3, constant_4) = &self.constants;

        let activation = match self.fused_activation {
            ActivationFunctionType::RELU => quote! { microflow::activations::ActivationType::RELU },
            ActivationFunctionType::NONE => quote! { microflow::activations::ActivationType::NONE },
            _ => unimplemented!(),
        };

        let output = if self.capacity.is_some() && self.capacity.unwrap() < weights.matrix.nrows() {
            let weights_vec: Vec<_> = weights
                .matrix
                .column_iter()
                .map(|c| {
                    TokenTensor::new(
                        DMatrix::from_iterator(c.nrows(), c.ncols(), c.iter().copied()).into(),
                        weights.scale,
                        weights.zero_point,
                    )
                })
                .collect();
            let constant_1_vec: Vec<_> = constant_1
                .iter()
                .map(|c| TokenMatrix::from(dmatrix![*c]))
                .collect();
            let constant_3_vec: Vec<_> = constant_3
                .iter()
                .map(|c| TokenMatrix::from(dmatrix![*c]))
                .collect();
            quote! {
                let output = microflow::tensor::QuantizedTensor::new(
                    nalgebra::SMatrix::from_columns(
                        &[
                            #(
                                microflow::ops::fully_connected(
                                    &output,
                                    #weights_vec,
                                    #output_scale,
                                    #output_zero_point,
                                    #activation,
                                    (#constant_0, #constant_1_vec, #constant_2, #constant_3_vec, #constant_4),
                                ).matrix
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
                    &output,
                    #weights,
                    #output_scale,
                    #output_zero_point,
                    #activation,
                    (#constant_0, #constant_1, #constant_2, #constant_3, #constant_4)
                );
            }
        };
        output.to_tokens(tokens);
    }
}
