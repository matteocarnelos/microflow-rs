use crate::quantize::{TokenQuantized, TrainToTokens};
use crate::tensor::TokenTensor2D;
use crate::tflite_flatbuffers::tflite::{Operator, Tensor, TensorType};
use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote, ToTokens};
use syn::parse_quote;

/// Represents the tokenized version of the `Softmax` operator.
pub(crate) struct TokenSoftmax<T: TokenQuantized> {
    pub(crate) output: TokenTensor2D<T>,
    pub(crate) layer_index: i32,
    pub(crate) train: bool,
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
) -> Box<dyn TrainToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(TokenSoftmax::<i8>::new(operator, tensors, layer_index)),
        TensorType::UINT8 => Box::new(TokenSoftmax::<u8>::new(operator, tensors, layer_index)),
        _ => unimplemented!(),
    }
}

/// Parses the [`TokenSoftmax`] struct from the given operator.
///
/// # Arguments
/// * `operator` - The model operator as an [`Operator`]
/// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
///
pub(crate) fn parse(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
) -> Box<dyn ToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(TokenSoftmax::<i8>::new(operator, tensors, -1)),
        TensorType::UINT8 => Box::new(TokenSoftmax::<u8>::new(operator, tensors, -1)),
        _ => unimplemented!(),
    }
}

impl<T: TokenQuantized> TokenSoftmax<T> {
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
            train: false,
        }
    }
}
impl<T: TokenQuantized> TrainToTokens for TokenSoftmax<T> {
    fn add_attrs(&self, _: &mut syn::ItemStruct) {}
    fn define_members(&self, _: &mut TokenStream2) {}
    fn switch_train(&mut self) {
        self.train = !self.train;
    }
    fn train_ops(&self, definitions: &mut TokenStream2) {}
    fn update_ops(&self, updates: &mut TokenStream2) {}
}

impl<T: TokenQuantized> ToTokens for TokenSoftmax<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let output_shape = &self.output.shape;
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        let output_name = if self.layer_index >= 0 && self.train {
            format_ident!("input{}", self.layer_index as usize)
        } else {
            format_ident!("input")
        };
        let input_name = if self.layer_index > 0 && self.train {
            format_ident!("input{}", (self.layer_index - 1) as usize)
        } else {
            format_ident!("input")
        };
        let func_name: syn::Path = if self.layer_index >= 0 && self.train {
            parse_quote!(microflow::ops::softmax_borrow)
        } else {
            parse_quote!(microflow::ops::softmax)
        };
        let reference_tok = if self.layer_index >= 0 && self.train {
            quote! {&}
        } else {
            quote! {}
        };
        print!("input{}", (self.layer_index - 1));
        let ts = quote! {
            let #output_name: microflow::tensor::Tensor2D<_, #(#output_shape),*, 1usize> =
                #func_name(#reference_tok #input_name, [#(#output_scale),*], [#(#output_zero_point),*]);
        };
        ts.to_tokens(tokens);
    }
}
