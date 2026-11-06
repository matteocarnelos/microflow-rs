use crate::activation::TokenFusedActivation;
use crate::buffer::TokenBuffer2D;
use crate::quantize::{TokenQuantized, TrainToTokens};
use crate::tensor::{TokenTensor2D, TokenTensor4D, TokenTensorViewPadding};
use crate::tflite_flatbuffers::tflite::{Buffer, Operator, Tensor, TensorType};
use flatbuffers::{ForwardsUOffset, Vector};
use nalgebra::DMatrix;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote, ToTokens};
use syn::{parse_quote, ItemStruct};

/// Represents the tokenized version of the `DepthwiseConv2D` operator.
pub(crate) struct TokenDepthwiseConv2D<T: TokenQuantized> {
    pub(crate) weights: TokenTensor4D<T>,
    pub(crate) output: TokenTensor4D<T>,
    pub(crate) fused_activation: TokenFusedActivation,
    pub(crate) view_padding: TokenTensorViewPadding,
    pub(crate) strides: (usize, usize),
    pub(crate) constants: (TokenBuffer2D<f32>, TokenBuffer2D<f32>),
    pub(crate) constants_gradient: (TokenBuffer2D<f32>, TokenBuffer2D<f32>),
    pub(crate) index: usize,
    pub(crate) layer_index: i32,
    pub(crate) scale_bias: Vec<f32>,
    pub(crate) train: bool,
    pub(crate) back_norm: f32,
    pub(crate) gradient_norm: f32,
}

/// Parses the [`TokenDepthwiseConv2D`] struct from the given operator.
///
/// # Arguments
/// * `operator` - The model operator as an [`Operator`]
/// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
/// * `buffers` - The model buffers as a [`Vector<ForwardsUOffset<Buffer>>`]
/// * `index` - The operator index
///
pub(crate) fn parse_indexed(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
    buffers: Vector<ForwardsUOffset<Buffer>>,
    index: usize,
    layer_index: i32,
    back_norm: f32,
    gradient_norm: f32,
) -> Box<dyn TrainToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(TokenDepthwiseConv2D::<i8>::new(
            operator,
            tensors,
            buffers,
            index,
            layer_index,
            back_norm,
            gradient_norm,
        )),
        TensorType::UINT8 => Box::new(TokenDepthwiseConv2D::<u8>::new(
            operator,
            tensors,
            buffers,
            index,
            layer_index,
            back_norm,
            gradient_norm,
        )),
        _ => unimplemented!(),
    }
}

/// Parses the [`TokenDepthwiseConv2D`] struct from the given operator.
///
/// # Arguments
/// * `operator` - The model operator as an [`Operator`]
/// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
/// * `buffers` - The model buffers as a [`Vector<ForwardsUOffset<Buffer>>`]
/// * `index` - The operator index
///
pub(crate) fn parse(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
    buffers: Vector<ForwardsUOffset<Buffer>>,
    index: usize,
) -> Box<dyn ToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(TokenDepthwiseConv2D::<i8>::new(
            operator, tensors, buffers, index, -1, -1f32, -1f32,
        )),
        TensorType::UINT8 => Box::new(TokenDepthwiseConv2D::<u8>::new(
            operator, tensors, buffers, index, -1, -1f32, -1f32,
        )),
        _ => unimplemented!(),
    }
}

impl<T: TokenQuantized> TokenDepthwiseConv2D<T> {
    /// Builds the [`TokenDepthwiseConv2D`] operator from the given model operator and tensors.
    ///
    /// # Arguments
    /// * `operator` - The model operator as an [`Operator`]
    /// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
    /// * `buffers` - The model buffers as a [`Vector<ForwardsUOffset<Buffer>>`]
    /// * `index` - The operator index
    ///
    pub(crate) fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
        buffers: Vector<ForwardsUOffset<Buffer>>,
        index: usize,
        layer_index: i32,
        back_norm: f32,
        gradient_norm: f32,
    ) -> Self {
        let inputs = operator.inputs().unwrap();
        let input = TokenTensor4D::from_empty_tensor(tensors.get(inputs.get(0) as usize));
        let weights =
            TokenTensor4D::from_buffered_tensor(tensors.get(inputs.get(1) as usize), buffers);
        let biases =
            TokenTensor2D::from_buffered_tensor(tensors.get(inputs.get(2) as usize), buffers);
        let output = TokenTensor4D::from_empty_tensor(
            tensors.get(operator.outputs().unwrap().get(0) as usize),
        );
        let options = operator
            .builtin_options_as_depthwise_conv_2_doptions()
            .unwrap();
        let constants = Self::preprocess(&input, &weights, &biases, &output);
        let constants_gradient = Self::gen_zero_constants(&weights);
        Self {
            weights,
            constants_gradient,
            output,
            fused_activation: options.fused_activation_function().into(),
            view_padding: options.padding().into(),
            strides: (options.stride_h() as usize, options.stride_w() as usize),
            constants,
            index,
            layer_index,
            train: false,
            scale_bias: biases.scale,
            back_norm,
            gradient_norm,
        }
    }

    /// Pre-processes the operator, returning the tuple of constants.
    ///
    /// # Arguments
    /// * `input` - The input of the operator as a [`TokenTensor2D`]
    /// * `weights` - The weights of the operator as a [`TokenTensor2D`]
    /// * `biases` - The biases of the operator as a [`TokenTensor2D`]
    /// * `output` - The output of the operator as a [`TokenTensor2D`]
    ///
    fn preprocess(
        input: &TokenTensor4D<T>,
        weights: &TokenTensor4D<T>,
        biases: &TokenTensor2D<i32>,
        output: &TokenTensor4D<T>,
    ) -> (TokenBuffer2D<f32>, TokenBuffer2D<f32>) {
        (
            TokenBuffer2D::from(DMatrix::from_fn(weights.shape[3], 1, |c, _| {
                biases.scale.get(c).copied().unwrap_or(biases.scale[0]) / output.scale[0]
                    * (biases.buffer[c]
                        - biases
                            .zero_point
                            .get(c)
                            .copied()
                            .unwrap_or(biases.zero_point[0])) as f32
            })),
            TokenBuffer2D::from(DMatrix::from_fn(weights.scale.len(), 1, |c, _| {
                input.scale[0] * weights.scale[c] / output.scale[0]
            })),
        )
    }
    fn gen_zero_constants(weights: &TokenTensor4D<T>) -> (TokenBuffer2D<f32>, TokenBuffer2D<f32>) {
        (
            TokenBuffer2D::from(DMatrix::from_fn(weights.shape[3], 1, |_, _| 0f32)),
            TokenBuffer2D::from(DMatrix::from_fn(weights.scale.len(), 1, |_, _| 0f32)),
        )
    }
}

impl<T: TokenQuantized> TrainToTokens for TokenDepthwiseConv2D<T> {
    fn add_attrs(&self, attrs: &mut ItemStruct) {
        let filters_ident = format_ident!("weights{}", self.layer_index as usize);
        let filters_gradient_ident = format_ident!("weights{}_gradient", self.layer_index as usize);
        let filters_shape = &self.weights.shape;
        let filters_shape_0 = filters_shape[0];
        let filters_shape_1 = filters_shape[1];
        let filters_shape_2 = filters_shape[2];
        let filters_shape_3 = filters_shape[3];
        let filters_type = self.weights.type_tokens();
        let constants_field_name = format_ident!("constants{}", self.layer_index as usize);
        let constants_gradient_field_name =
            format_ident!("constants{}_gradient", self.layer_index as usize);
        let dim00 = self.constants.0.shape().0;
        let dim01 = self.constants.0.shape().1;
        let dim10 = self.constants.1.shape().0;
        let dim11 = self.constants.1.shape().1;
        let constants_field_type: syn::Type = parse_quote! {
            ( microflow::buffer::Buffer2D<f32, #dim00, #dim01>, microflow::buffer::Buffer2D<f32, #dim10, #dim11>,)
        };

        let constants_field: syn::Field = syn::parse_quote! {
            #constants_field_name: #constants_field_type
        };
        let filters_field: syn::Field = syn::parse_quote! {
            #filters_ident: #filters_type
        };
        let constants_field_gradient: syn::Field = syn::parse_quote! {
            #constants_gradient_field_name: #constants_field_type
        };
        let filters_field_gradient: syn::Field = syn::parse_quote! {
            #filters_gradient_ident: microflow::buffer::Buffer4D<i32,#filters_shape_0,#filters_shape_1,#filters_shape_2,#filters_shape_3>
        };
        match &mut attrs.fields {
            syn::Fields::Named(ref mut fields_named) => {
                fields_named.named.push(constants_field);
                fields_named.named.push(filters_field);
                fields_named.named.push(constants_field_gradient);
                fields_named.named.push(filters_field_gradient);
            }
            _ => panic!("add_fields only works with structs with named fields"),
        }
    }
    fn define_members(&self, declarations: &mut TokenStream2) {
        let constants_field_name = format_ident!("constants{}", self.layer_index as usize);
        let filters_ident = format_ident!("weights{}", self.layer_index as usize);
        let constants_gradient_field_name =
            format_ident!("constants{}_gradient", self.layer_index as usize);
        let filters_gradient_ident = format_ident!("weights{}_gradient", self.layer_index as usize);
        let filters = &self.weights;
        let (constants_0, constants_1) = &self.constants;
        let (constants_0_gradient, constants_1_gradient) = &self.constants_gradient;
        let ts = quote! {
            #filters_ident : #filters,
            #constants_field_name : (#constants_0, #constants_1),
            #filters_gradient_ident : core::array::from_fn(|_|SMatrix::from_fn(|_,_|core::array::from_fn(|_|0i32))),
            #constants_gradient_field_name : (#constants_0_gradient, #constants_1_gradient),
        };
        ts.to_tokens(declarations);
    }
    fn switch_train(&mut self) {
        self.train = !self.train;
    }
    fn train_ops(&self, backward: &mut TokenStream2) {
        let weights_ident: syn::Expr = {
            let field_ident = format_ident!("weights{}", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        };
        let weights_gradient_ident: syn::Expr = {
            let field_ident = format_ident!("weights{}_gradient", self.layer_index as usize);
            parse_quote!(self.#field_ident)
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
        let constants_ident: syn::Expr = {
            let field_ident = format_ident!("constants{}", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        };
        let constants_gradient_ident: syn::Expr = {
            let field_ident = format_ident!("constants{}_gradient", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        };
        let activation = self.fused_activation;
        let view_padding = self.view_padding;
        let (strides_0, strides_1) = self.strides;
        let bias_scale = self
            .scale_bias
            .iter()
            .map(|x| quote! { #x })
            .collect::<Vec<_>>();
        let backward_norm = self.back_norm;
        let prepend = quote! {
            let backward_gradient = microflow::gradient_depthwise_conv_2d::update_grad_depthwise_conv_2d(
                &#input_ident,
                &#weights_ident,
                &mut #weights_gradient_ident,
                &#constants_ident,
                &mut #constants_gradient_ident,
                #output_ident,
                backward_gradient,
                #activation,
                (#strides_0,#strides_1),
                #view_padding,
                [#(#bias_scale),*],
                learning_rate,
                #backward_norm
            );
            // println!("gradient weights: {}",weight_gradient.iter().fold(String::new(), |accarr, batch|accarr + &batch.map(|el|el.iter().fold(String::new(),|sum, el1|sum +" "+ &el1.to_string())).to_string()));
            // println!("gradient input: {}",backward_gradient[0].map(|el|el.iter().fold(String::new(),|sum, el1|sum +" " +&el1.to_string())));
            // println!("mean gradient: {}",backward_gradient[0].map(|el|el.iter().fold(0f32,|sum, el1|sum+(*el1 as f32).abs() / el.len() as f32)).mean());
        };
        let mut ts = TokenStream2::new();
        prepend.to_tokens(&mut ts);
        ts.extend(backward.clone());
        *backward = ts;
    }
    fn update_ops(&self, updates: &mut TokenStream2) {
        let weights_ident: syn::Expr = {
            let field_ident = format_ident!("weights{}", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        };
        let weights_gradient_ident: syn::Expr = {
            let field_ident = format_ident!("weights{}_gradient", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        };
        let constants_ident: syn::Expr = {
            let field_ident = format_ident!("constants{}", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        };
        let constants_gradient_ident: syn::Expr = {
            let field_ident = format_ident!("constants{}_gradient", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        };
        let gradient_norm = self.gradient_norm;
        let update = quote! {
            microflow::update_layer::update_weights_clip_4D(
                &mut #weights_ident,
                &#weights_gradient_ident,
                batch_size,
                learning_rate,
                #gradient_norm
            );
            microflow::update_layer::update_weights_2D_float(
                &mut #constants_ident.0,
                &#constants_gradient_ident.0,
                batch_size,
                learning_rate,
            );
            #weights_gradient_ident = core::array::from_fn(|_|SMatrix::from_fn(|_,_|core::array::from_fn(|_|0i32)));
            #constants_gradient_ident.0 = SMatrix::zeros();
        };
        update.to_tokens(updates);
    }
}
impl<T: TokenQuantized> ToTokens for TokenDepthwiseConv2D<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let weights_ident: syn::Expr = if self.layer_index >= 0 {
            let field_ident = format_ident!("weights{}", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        } else {
            let field_ident = format_ident!("weights{}", self.index);
            parse_quote!(#field_ident)
        };
        let constants_ident: syn::Expr = if self.layer_index >= 0 {
            let field_ident = format_ident!("constants{}", self.layer_index as usize);
            parse_quote!(self.#field_ident)
        } else {
            let (constants_0, constants_1) = &self.constants;
            parse_quote!((#constants_0, #constants_1))
        };
        let weights_type = self.weights.type_tokens();
        let weights = &self.weights;
        let weights_declaration = if self.layer_index < 0 {
            quote! {const #weights_ident: #weights_type = #weights;}
        } else {
            quote! {}
        };
        let output_shape = &self.output.shape;
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        let fused_activation = self.fused_activation;
        let view_padding = self.view_padding;
        let (strides_0, strides_1) = self.strides;
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
            parse_quote!(microflow::ops::depthwise_conv_2d_borrow)
        } else {
            parse_quote!(microflow::ops::depthwise_conv_2d)
        };
        let ts = quote! {
        #weights_declaration
        let #output_name: microflow::tensor::Tensor4D<_, #(#output_shape),*, 1usize> =
            #func_name(
                #reference_tok (#input_name),
                &#weights_ident,
                [#(#output_scale),*],
                [#(#output_zero_point),*],
                microflow::ops::DepthwiseConv2DOptions {
                    fused_activation: #fused_activation,
                    view_padding: #view_padding,
                    strides: (#strides_0, #strides_1),
                },
                #constants_ident
        );
        };
        ts.to_tokens(tokens);
    }
}
