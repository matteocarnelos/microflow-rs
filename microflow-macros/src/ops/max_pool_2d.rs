use crate::activation::TokenFusedActivation;
use crate::quantize::TokenQuantized;
use crate::tensor::{TokenTensor4D, TokenTensorViewPadding};
use crate::tflite_flatbuffers::tflite::{Operator, Tensor, TensorType};
use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro2::TokenStream as TokenStream2;
use proc_macro_error::abort_call_site;
use quote::{quote, ToTokens};

pub(crate) struct TokenMaxPool2D<T: TokenQuantized> {
    pub(crate) filter_shape: (usize, usize),
    pub(crate) output: TokenTensor4D<T>,
    pub(crate) fused_activation: TokenFusedActivation,
    pub(crate) view_padding: TokenTensorViewPadding,
    pub(crate) strides: (usize, usize),
}

pub(crate) fn parse(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
) -> Box<dyn ToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(TokenMaxPool2D::<i8>::new(operator, tensors)),
        TensorType::UINT8 => Box::new(TokenMaxPool2D::<u8>::new(operator, tensors)),
        input_type => abort_call_site!(
            "MaxPool2D supports only INT8/UINT8 input tensors, got {:?}",
            input_type
        ),
    }
}

impl<T: TokenQuantized> TokenMaxPool2D<T> {
    pub(crate) fn new(operator: Operator, tensors: Vector<ForwardsUOffset<Tensor>>) -> Self {
        let output = TokenTensor4D::from_empty_tensor(
            tensors.get(operator.outputs().unwrap().get(0) as usize),
        );
        let options = operator.builtin_options_as_pool_2_doptions().unwrap();
        Self {
            filter_shape: (
                options.filter_height() as usize,
                options.filter_width() as usize,
            ),
            output,
            fused_activation: options.fused_activation_function().into(),
            view_padding: options.padding().into(),
            strides: (options.stride_h() as usize, options.stride_w() as usize),
        }
    }
}

impl<T: TokenQuantized> ToTokens for TokenMaxPool2D<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let (filter_shape_0, filter_shape_1) = self.filter_shape;
        let output_shape = &self.output.shape;
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        let fused_activation = self.fused_activation;
        let view_padding = self.view_padding;
        let (strides_0, strides_1) = self.strides;

        quote! {
            let input: microflow::tensor::Tensor4D<_, #(#output_shape),*, 1usize> =
                microflow::ops::max_pool_2d(
                    input,
                    (nalgebra::Const::<#filter_shape_0>, nalgebra::Const::<#filter_shape_1>),
                    [#(#output_scale),*],
                    [#(#output_zero_point),*],
                    microflow::ops::MaxPool2DOptions {
                        fused_activation: #fused_activation,
                        view_padding: #view_padding,
                        strides: (#strides_0, #strides_1),
                    },
            );
        }
        .to_tokens(tokens);
    }
}
