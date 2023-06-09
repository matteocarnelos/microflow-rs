use crate::activation::TokenFusedActivation;
use crate::quantize::TokenQuantized;
use crate::tensor::{TokenTensor4D, TokenViewPadding};
use crate::tflite_flatbuffers::tflite::{Operator, Tensor, TensorType};
use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};
use simba::scalar::SupersetOf;

pub(crate) struct TokenAveragePool2D<T: TokenQuantized> {
    pub(crate) filter_shape: (usize, usize),
    pub(crate) output: TokenTensor4D<T>,
    pub(crate) fused_activation: TokenFusedActivation,
    pub(crate) view_padding: TokenViewPadding,
    pub(crate) strides: (usize, usize),
    pub(crate) constants: (f32, f32),
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
        }
    }

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

        let ts = quote! {
            let input: microflow::tensor::Tensor4D<_, #(#output_shape),*, 1usize> =
                microflow::ops::average_pool_2d(
                    input,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::TokenBuffer4D;

    fn setup() -> TokenAveragePool2D<i8> {
        TokenAveragePool2D {
            filter_shape: (2, 3),
            output: TokenTensor4D {
                buffer: TokenBuffer4D::new(),
                shape: vec![1, 2, 3, 2],
                scale: vec![0.1],
                zero_point: vec![2],
            },
            fused_activation: TokenFusedActivation::None,
            view_padding: TokenViewPadding::Same,
            strides: (1, 1),
            constants: (3., 4.),
        }
    }

    #[test]
    fn average_pool_2d_preprocess() {
        let layer = setup();
        let input = TokenTensor4D {
            buffer: TokenBuffer4D::new(),
            shape: vec![1, 2, 3, 2],
            scale: vec![0.5],
            zero_point: vec![6],
        };
        let constants = TokenAveragePool2D::preprocess(&input, &layer.output);
        assert_eq!(constants.0, 5.);
        assert_eq!(constants.1, -28.);
    }

    #[test]
    fn average_pool_2d_to_tokens() {
        let layer = setup();
        let fused_activation = layer.fused_activation;
        let view_padding = layer.view_padding;
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                let input: microflow::tensor::Tensor4D<_, 1usize, 2usize, 3usize, 2usize, 1usize> =
                    microflow::ops::average_pool_2d(
                        input,
                        (nalgebra::Const::<2usize>, nalgebra::Const::<3usize>),
                        [0.1f32],
                        [2i8],
                        microflow::ops::AveragePool2DOptions {
                            fused_activation: #fused_activation,
                            view_padding: #view_padding,
                            strides: (1usize, 1usize),
                        },
                        (3f32, 4f32)
                );
            }
            .to_string()
        );
    }
}
