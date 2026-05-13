use crate::activation::TokenFusedActivation;
use crate::quantize::TokenQuantized;
use crate::tensor::{TokenTensor2D, TokenTensor4D, TokenTensorViewPadding};
use crate::tflite_flatbuffers::tflite::{Operator, Tensor, TensorType};
use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens};
use simba::scalar::SupersetOf;

/// Represents the tokenized version of the `Quantize` operator.
enum DynamicTokenTensor<T : TokenQuantized> {
    _2D(TokenTensor2D<T>),
    _4D(TokenTensor4D<T>),
}
pub(crate) struct TokenQuantize<T: TokenQuantized> {
    pub(crate) output: DynamicTokenTensor<T>,
    pub(crate) constants: (f32, f32, f32),
}

/// Parses the [`TokenQuantize`] struct from the given operator.
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
        TensorType::INT8 => Box::new(TokenQuantize::<i8>::new(operator, tensors)),
        TensorType::UINT8 => Box::new(TokenQuantize::<u8>::new(operator, tensors)),
        _ => unimplemented!(),
    }
}

impl<T: TokenQuantized> TokenQuantize<T> {
    /// Builds the [`TokenQuantize`] operator from the given model operator and tensors.
    ///
    /// # Arguments
    /// * `operator` - The model operator as an [`Operator`]
    /// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
    ///
    pub(crate) fn new(operator: Operator, tensors: Vector<ForwardsUOffset<Tensor>>) -> Self {
        let inputs = operator.inputs().unwrap();
        let input_shape : Vec<_> = tensors.get(inputs.get(0) as usize).shape().unwrap().iter().map(|e| e as usize).collect();
        let input = match input_shape.len() {
                2 => DynamicTokenTensor::_2D(TokenTensor2D::from_empty_tensor(tensors.get(inputs.get(0) as usize))),
                4 => DynamicTokenTensor::_4D(TokenTensor4D::from_empty_tensor(tensors.get(inputs.get(0) as usize))),
                _ => unimplemented!(),
            };
        let outputs = operator.outputs().unwrap();
        let output_shape : Vec<_> = tensors.get(outputs.get(0) as usize).shape().unwrap().iter().map(|e| e as usize).collect();
        let output = match output_shape.len() {
                2 => DynamicTokenTensor::_2D(TokenTensor2D::from_empty_tensor(tensors.get(outputs.get(0) as usize))),
                4 => DynamicTokenTensor::_4D(TokenTensor4D::from_empty_tensor(tensors.get(outputs.get(0) as usize))),
                _ => unimplemented!(),
            };
        let options = operator.builtin_options_as_quantize_options().unwrap();
        let constants = Self::preprocess(&input, &output);
        Self {
            output,
            constants
        }
    }

    /// Pre-processes the operator, returning the tuple of constants.
    ///
    /// # Arguments
    /// * `input` - The input of the operator as a [`TokenTensor2D`]
    /// * `output` - The output of the operator as a [`TokenTensor2D`]
    ///
    fn preprocess(input: &DynamicTokenTensor<T>, output: &DynamicTokenTensor<T>) -> (f32, f32, f32) {
        match (input, output){
            (DynamicTokenTensor::_2D(input), DynamicTokenTensor::_2D(output)) => (
            input.scale[0] / output.scale[0],
            f32::from_subset(&output.zero_point[0])/output.scale[0],
            input.scale[0]
        ),
            (DynamicTokenTensor::_4D(input), DynamicTokenTensor::_4D(output)) => (
            input.scale[0] / output.scale[0],
            f32::from_subset(&output.zero_point[0])/output.scale[0],
            input.scale[0]
        ),
            _ => unimplemented!()

        }
    }
}

impl<T: TokenQuantized> ToTokens for TokenQuantize<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
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

    fn setup() -> TokenQuantize<i8> {
        TokenQuantize {
            filter_shape: (2, 3),
            output: TokenTensor4D {
                buffer: TokenBuffer4D::new(),
                shape: vec![1, 2, 3, 2],
                scale: vec![0.1],
                zero_point: vec![2],
            },
            fused_activation: TokenFusedActivation::None,
            view_padding: TokenTensorViewPadding::Same,
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
        let constants = TokenQuantize::preprocess(&input, &layer.output);
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
