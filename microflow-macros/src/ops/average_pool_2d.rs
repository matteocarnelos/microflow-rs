use crate::activation::TokenFusedActivation;
use crate::quantize::TokenQuantized;
use crate::tensor::TokenTensor4D;
use crate::tflite_flatbuffers::tflite::{Operator, Padding, Tensor, TensorType};
use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro2::TokenStream;
use quote::{quote, ToTokens};

pub(crate) struct TokenAveragePool2D<T: TokenQuantized> {
    pub(crate) filter_shape: (usize, usize),
    pub(crate) output: TokenTensor4D<T>,
    pub(crate) fused_activation: TokenFusedActivation,
    pub(crate) padding: Padding,
    pub(crate) strides: (usize, usize),
}

pub(crate) fn parse(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
) -> Box<dyn ToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(TokenAveragePool2D::<i8>::new(operator, tensors)),
        TensorType::UINT8 => Box::new(TokenAveragePool2D::<u8>::new(operator, tensors)),
        _ => unimplemented!(),
    }
}

impl<T: TokenQuantized> TokenAveragePool2D<T> {
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
            fused_activation: TokenFusedActivation(options.fused_activation_function()),
            padding: options.padding(),
            strides: (options.stride_h() as usize, options.stride_w() as usize),
        }
    }
}

impl<T: TokenQuantized> ToTokens for TokenAveragePool2D<T> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let mut output_shape = self.output.shape.clone();
        output_shape.push(self.output.scale.len());
        let (filter_shape_0, filter_shape_1) = self.filter_shape;
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        let fused_activation = &self.fused_activation;
        let padding = match self.padding {
            Padding::SAME => quote!(microflow::ops::AveragePool2DPadding::SAME),
            Padding::VALID => quote!(microflow::ops::AveragePool2DPadding::VALID),
            _ => unreachable!(),
        };
        let (strides_0, strides_1) = self.strides;

        let output = quote! {
            let output: microflow::tensor::Tensor4D<_, #(#output_shape),*> = microflow::ops::average_pool_2d(
                output.into(),
                (nalgebra::Const::<#filter_shape_0>, nalgebra::Const::<#filter_shape_1>),
                [#(#output_scale),*],
                [#(#output_zero_point),*],
                microflow::ops::AveragePool2DOptions {
                    fused_activation: #fused_activation,
                    padding: #padding,
                    strides: (#strides_0, #strides_1),
                }
            );
        };
        output.to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::TokenBuffer4D;
    use crate::tflite_flatbuffers::tflite::ActivationFunctionType;

    #[test]
    fn average_pool_2d_to_tokens() {
        let layer = TokenAveragePool2D {
            filter_shape: (2, 3),
            output: TokenTensor4D {
                buffer: TokenBuffer4D::new(),
                shape: vec![1, 2, 3, 2],
                scale: vec![0.1],
                zero_point: vec![2i8],
            },
            fused_activation: TokenFusedActivation(ActivationFunctionType::NONE),
            padding: Padding::SAME,
            strides: (1, 1),
        };
        let fused_activation = &layer.fused_activation;
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                let output: microflow::tensor::Tensor4D<_, 1usize, 2usize, 3usize, 2usize, 1usize> = microflow::ops::average_pool_2d(
                    output.into(),
                    (nalgebra::Const::<2usize>, nalgebra::Const::<3usize>),
                    [0.1f32],
                    [2i8],
                    microflow::ops::AveragePool2DOptions {
                        fused_activation: #fused_activation,
                        padding: microflow::ops::AveragePool2DPadding::SAME,
                        strides: (1usize, 1usize),
                    }
                );
            }
            .to_string()
        );
    }
}
