use crate::activation::TokenFusedActivation;
use crate::quantize::{TokenQuantized, TrainToTokens};
use crate::tensor::{TokenTensor4D, TokenTensorViewPadding};
use crate::tflite_flatbuffers::tflite::{Operator, Tensor, TensorType};
use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote, ToTokens};
use simba::scalar::SupersetOf;
use syn::{parse_quote, ItemStruct};

/// Represents the tokenized version of the `AveragePool2D` operator.
pub(crate) struct TokenAveragePool2D<T: TokenQuantized> {
    pub(crate) filter_shape: (usize, usize),
    pub(crate) output: TokenTensor4D<T>,
    pub(crate) fused_activation: TokenFusedActivation,
    pub(crate) view_padding: TokenTensorViewPadding,
    pub(crate) strides: (usize, usize),
    pub(crate) constants: (f32, f32),
    pub(crate) layer_index: i32,
    pub(crate) train: bool,
}

/// Parses the [`TokenAveragePool2D`] struct from the given operator.
///
/// # Arguments
/// * `operator` - The model operator as an [`Operator`]
/// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
/// * `index` - index associated to the layer. Used for trained layers
///
pub(crate) fn parse_indexed(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
    index: i32,
) -> Box<dyn TrainToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(TokenAveragePool2D::<i8>::new(operator, tensors, index)),
        TensorType::UINT8 => Box::new(TokenAveragePool2D::<u8>::new(operator, tensors, index)),
        _ => unimplemented!(),
    }
}
/// Parses the [`TokenAveragePool2D`] struct from the given operator.
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
        TensorType::INT8 => Box::new(TokenAveragePool2D::<i8>::new(operator, tensors, -1)),
        TensorType::UINT8 => Box::new(TokenAveragePool2D::<u8>::new(operator, tensors, -1)),
        _ => unimplemented!(),
    }
}

impl<T: TokenQuantized> TokenAveragePool2D<T> {
    /// Builds the [`TokenAveragePool2D`] operator from the given model operator and tensors.
    ///
    /// # Arguments
    /// * `operator` - The model operator as an [`Operator`]
    /// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
    ///
    pub(crate) fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
        index: i32,
    ) -> Self {
        let inputs = operator.inputs().unwrap();
        let input = TokenTensor4D::from_empty_tensor(tensors.get(inputs.get(0) as usize));
        let output = TokenTensor4D::from_empty_tensor(
            tensors.get(operator.outputs().unwrap().get(0) as usize),
        );
        let options = operator.builtin_options_as_pool_2_doptions().unwrap();
        let constants = Self::preprocess(&input, &output);
        Self {
            filter_shape: (
                options.filter_height() as usize,
                options.filter_width() as usize,
            ),
            output,
            fused_activation: options.fused_activation_function().into(),
            view_padding: options.padding().into(),
            strides: (options.stride_h() as usize, options.stride_w() as usize),
            constants,
            layer_index: index,
            train: false,
        }
    }

    /// Pre-processes the operator, returning the tuple of constants.
    ///
    /// # Arguments
    /// * `input` - The input of the operator as a [`TokenTensor2D`]
    /// * `output` - The output of the operator as a [`TokenTensor2D`]
    ///
    fn preprocess(input: &TokenTensor4D<T>, output: &TokenTensor4D<T>) -> (f32, f32) {
        (
            input.scale[0] / output.scale[0],
            f32::from_subset(&output.zero_point[0])
                - (input.scale[0] * f32::from_subset(&input.zero_point[0])) / output.scale[0],
        )
    }
}

impl<T: TokenQuantized> ToTokens for TokenAveragePool2D<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let (filter_shape_0, filter_shape_1) = self.filter_shape;
        let output_shape = &self.output.shape;
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        let fused_activation = self.fused_activation;
        let view_padding = self.view_padding;
        let (strides_0, strides_1) = self.strides;
        let (constants_0, constants_1) = self.constants;
        let reference_tok = if self.layer_index >= 0 && self.train {
            quote! {&}
        } else {
            quote! {}
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
        let func_name: syn::Path = if self.layer_index >= 0 && self.train {
            parse_quote!(microflow::ops::average_pool_2d_borrow)
        } else {
            parse_quote!(microflow::ops::average_pool_2d)
        };
        let ts = quote! {
            let #output_name: microflow::tensor::Tensor4D<_, #(#output_shape),*, 1usize> =
                #func_name(
                    #reference_tok #input_name,
                    (nalgebra::Const::<#filter_shape_0>, nalgebra::Const::<#filter_shape_1>),
                    [#(#output_scale),*],
                    [#(#output_zero_point),*],
                    microflow::ops::AveragePool2DOptions {
                        fused_activation: #fused_activation,
                        view_padding: #view_padding,
                        strides: (#strides_0, #strides_1),
                    },
                    (#constants_0, #constants_1)
                );
        };
        ts.to_tokens(tokens);
    }
}

impl<T: TokenQuantized> TrainToTokens for TokenAveragePool2D<T> {
    fn add_attrs(&self, _: &mut ItemStruct) {}
    fn define_members(&self, _: &mut proc_macro2::TokenStream) {}
    fn switch_train(&mut self) {
        self.train = !self.train;
    }
    fn train_ops(&self, backward: &mut TokenStream2) {
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
        let activation = self.fused_activation;
        let (shape_0, shape_1) = self.filter_shape;
        let (stride_0, stride_1) = self.strides;
        let padding = self.view_padding;
        let prepend = quote! {
            let backward_gradient = microflow::gradient_average_pool::gradient_average_pool(
                &#input_ident,
                #output_ident,
                backward_gradient,
                (nalgebra::Const::<#shape_0>, nalgebra::Const::<#shape_1>),
                #activation,
                (#stride_0, #stride_1),
                #padding,
            );
            // println!("gradient input average pool: {}",backward_gradient[0].map(|el|el.iter().fold(String::new(),|sum, el1|sum +" " +&el1.to_string())));
        };
        let mut ts = TokenStream2::new();
        prepend.to_tokens(&mut ts);
        ts.extend(backward.clone());
        *backward = ts;
    }
    fn update_ops(&self, updates: &mut TokenStream2) {}
}
