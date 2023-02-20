use flatbuffers::{ForwardsUOffset, Vector};
use quote::ToTokens;

use crate::tflite_flatbuffers::tflite::{ActivationFunctionType, Buffer, Operator, Tensor};
use crate::{quote, ParsedTensor, TokenStream2};

pub struct FullyConnected {
    pub(crate) weights: ParsedTensor<i8>,
    pub(crate) biases: ParsedTensor<i32>,
    pub(crate) output: ParsedTensor<i8>,
    pub(crate) activation: ActivationFunctionType,
}

impl FullyConnected {
    pub fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
        buffers: Vector<ForwardsUOffset<Buffer>>,
    ) -> Self {
        let weights = tensors.get(operator.inputs().unwrap().get(1) as usize);
        let biases = tensors.get(operator.inputs().unwrap().get(2) as usize);
        let output = tensors.get(operator.outputs().unwrap().get(0) as usize);
        Self {
            weights: ParsedTensor::new_with_data(weights, buffers),
            biases: ParsedTensor::new_with_data(biases, buffers),
            output: ParsedTensor::new_empty(output),
            activation: operator
                .builtin_options_as_fully_connected_options()
                .unwrap()
                .fused_activation_function(),
        }
    }
}

impl ToTokens for FullyConnected {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let weights = &self.weights;
        let biases = &self.biases;
        let output_scale = self.output.scale;
        let output_zero_point = self.output.zero_point;
        let activation = match self.activation {
            ActivationFunctionType::RELU => quote! { microflow::activations::Activation::RELU },
            ActivationFunctionType::NONE => quote! { microflow::activations::Activation::NONE },
            _ => unimplemented!(),
        };
        let layer = quote! {
            let weights = #weights;
            let biases = #biases;
            let output = microflow::ops::fully_connected(output, weights, biases, #output_scale, #output_zero_point, #activation);
        };
        layer.to_tokens(tokens);
    }
}
