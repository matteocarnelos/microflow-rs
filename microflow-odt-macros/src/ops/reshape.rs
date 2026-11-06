use crate::{
    quantize::TrainToTokens,
    tflite_flatbuffers::tflite::{Operator, Tensor},
};
use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote, ToTokens};

/// Represents the tokenized version of the `Reshape` operator.
pub(crate) struct TokenReshape {
    pub(crate) input_shape: Vec<usize>,
    pub(crate) output_shape: Vec<usize>,
    pub(crate) layer_index: i32,
    pub(crate) train: bool,
}

/// Parses the [`TokenReshape`] struct from the given operator.
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
    Box::new(TokenReshape::new(operator, tensors, layer_index))
}

/// Parses the [`TokenReshape`] struct from the given operator.
///
/// # Arguments
/// * `operator` - The model operator as an [`Operator`]
/// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
///
pub(crate) fn parse(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
) -> Box<dyn ToTokens> {
    Box::new(TokenReshape::new(operator, tensors, -1))
}

impl TokenReshape {
    /// Builds the [`TokenReshape`] operator from the given model operator and tensors.
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
        let input_shape: Vec<_> = tensors
            .get(operator.inputs().unwrap().get(0) as usize)
            .shape()
            .unwrap()
            .iter()
            .map(|e| e as usize)
            .collect();
        let output_shape: Vec<_> = tensors
            .get(operator.outputs().unwrap().get(0) as usize)
            .shape()
            .unwrap()
            .iter()
            .map(|e| e as usize)
            .collect();
        Self {
            input_shape,
            output_shape,
            layer_index,
            train: false,
        }
    }
}
impl TrainToTokens for TokenReshape {
    fn add_attrs(&self, _: &mut syn::ItemStruct) {}
    fn define_members(&self, _: &mut TokenStream2) {}
    fn switch_train(&mut self) {
        self.train = !self.train;
    }
    fn train_ops(&self, backward: &mut TokenStream2) {
        let input_shape = &self.input_shape;
        let input_tensor = match input_shape.len() {
            2 => quote!(Tensor2D),
            4 => quote!(Tensor4D),
            _ => unimplemented!(),
        };
        let output_shape = &self.output_shape;
        let output_tensor = match output_shape.len() {
            2 => quote!(Tensor2D),
            4 => quote!(Tensor4D),
            _ => unimplemented!(),
        };
        let output_ident = if self.layer_index >= 0 {
            format_ident!("input{}", self.layer_index as usize)
        } else {
            format_ident!("input")
        };
        let input_ident = if self.layer_index > 0 {
            format_ident!("input{}", (self.layer_index - 1) as usize)
        } else {
            format_ident!("input")
        };
        let prepend = quote! {
            let backward_gradient = microflow::tensor::#output_tensor::new(backward_gradient, [1f32],[0]);
            let backward_gradient: microflow::tensor::#input_tensor<_, #(#input_shape),*, 1usize> =
            microflow::ops::reshape(backward_gradient);
            let backward_gradient = backward_gradient.buffer;
            let #input_ident: microflow::tensor::#input_tensor<_, #(#input_shape),*, 1usize> =
                microflow::ops::reshape(#output_ident);
        };
        let mut ts = TokenStream2::new();
        prepend.to_tokens(&mut ts);
        ts.extend(backward.clone());
        *backward = ts;
    }
    fn update_ops(&self, updates: &mut TokenStream2) {}
}

impl ToTokens for TokenReshape {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let output_shape = &self.output_shape;
        let output_tensor = match output_shape.len() {
            2 => quote!(Tensor2D),
            4 => quote!(Tensor4D),
            _ => unimplemented!(),
        };
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
        let ts = quote! {
            let #output_name: microflow::tensor::#output_tensor<_, #(#output_shape),*, 1usize> =
                microflow::ops::reshape(#input_name);
        };
        ts.to_tokens(tokens)
    }
}
