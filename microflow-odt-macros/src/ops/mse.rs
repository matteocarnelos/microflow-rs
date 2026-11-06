use crate::quantize::{TokenQuantized, TrainToTokens};
use crate::tensor::TokenTensor2D;
use crate::tflite_flatbuffers::tflite::{Operator, Tensor, TensorType};
use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote, ToTokens};

/// Represents the tokenized version of the `Softmax` operator.
pub(crate) struct TokenMse<T: TokenQuantized> {
    pub(crate) output: TokenTensor2D<T>,
    pub(crate) layer_index: i32,
}

/// Parses the [`TokenSoftmax`] struct from the given operator.
///
/// # Arguments
/// * `operator` - The model operator as an [`Operator`]
/// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
///
pub(crate) fn parse_indexed(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
    layer_index: i32,
) -> Box<dyn ToTokens> {
    let inputs = operator.outputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(TokenMse::<i8>::new(operator, tensors, layer_index)),
        TensorType::UINT8 => Box::new(TokenMse::<u8>::new(operator, tensors, layer_index)),
        _ => unimplemented!(),
    }
}
impl<T: TokenQuantized> TokenMse<T> {
    /// Builds the [`TokenSoftmax`] operator from the given model operator and tensors.
    ///
    /// # Arguments
    /// * `operator` - The model operator as an [`Operator`]
    /// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
    ///
    pub(crate) fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
        layer_index: i32,
    ) -> Self {
        let output = TokenTensor2D::from_empty_tensor(
            tensors.get(operator.outputs().unwrap().get(0) as usize),
        );
        Self {
            output,
            layer_index,
        }
    }
}

impl<T: TokenQuantized> ToTokens for TokenMse<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let output_name = format_ident!("input{}", self.layer_index as usize);
        print!("input{}", (self.layer_index - 1));
        let ts = quote! {
            let backward_gradient =
                microflow::update_layer::mse_grad(& #output_name, output_gt);
            // println!("mse grad = {}", backward_gradient);
        };
        ts.to_tokens(tokens);
    }
}
