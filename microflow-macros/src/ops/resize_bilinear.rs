use crate::quantize::TokenQuantized;
use crate::tensor::TokenTensor4D;
use crate::tflite_flatbuffers::tflite::{Operator, Tensor, TensorType};
use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro2::TokenStream as TokenStream2;
use proc_macro_error::abort_call_site;
use quote::{quote, ToTokens};
use simba::scalar::SupersetOf;

/// Represents the tokenized version of the `ResizeBilinear` operator.
pub(crate) struct TokenResizeBilinear<T: TokenQuantized> {
    pub(crate) output: TokenTensor4D<T>,
    pub(crate) align_corners: bool,
    pub(crate) half_pixel_centers: bool,
    pub(crate) constants: (f32, f32),
}

/// Parses the [`TokenResizeBilinear`] struct from the given operator.
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
        TensorType::INT8 => Box::new(TokenResizeBilinear::<i8>::new(operator, tensors)),
        TensorType::UINT8 => Box::new(TokenResizeBilinear::<u8>::new(operator, tensors)),
        input_type => abort_call_site!(
            "ResizeBilinear supports only INT8/UINT8 input tensors, got {:?}",
            input_type
        ),
    }
}

impl<T: TokenQuantized> TokenResizeBilinear<T> {
    /// Builds the [`TokenResizeBilinear`] operator from the given model operator and tensors.
    ///
    /// # Arguments
    /// * `operator` - The model operator as an [`Operator`]
    /// * `tensors` - The model tensors as a [`Vector<ForwardsUOffset<Tensor>>`]
    ///
    pub(crate) fn new(operator: Operator, tensors: Vector<ForwardsUOffset<Tensor>>) -> Self {
        let inputs = operator.inputs().unwrap();
        let input = TokenTensor4D::from_empty_tensor(tensors.get(inputs.get(0) as usize));
        let output = TokenTensor4D::from_empty_tensor(
            tensors.get(operator.outputs().unwrap().get(0) as usize),
        );
        let options = operator
            .builtin_options_as_resize_bilinear_options()
            .unwrap();
        let constants = Self::preprocess(&input, &output);
        Self {
            output,
            align_corners: options.align_corners(),
            half_pixel_centers: options.half_pixel_centers(),
            constants,
        }
    }

    /// Pre-processes the operator, returning the tuple of constants.
    ///
    /// # Arguments
    /// * `input` - The input of the operator as a [`TokenTensor4D`]
    /// * `output` - The output of the operator as a [`TokenTensor4D`]
    ///
    fn preprocess(input: &TokenTensor4D<T>, output: &TokenTensor4D<T>) -> (f32, f32) {
        (
            input.scale[0] / output.scale[0],
            f32::from_subset(&output.zero_point[0])
                - (input.scale[0] * f32::from_subset(&input.zero_point[0])) / output.scale[0],
        )
    }
}

impl<T: TokenQuantized> ToTokens for TokenResizeBilinear<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let output_shape = &self.output.shape;
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        let align_corners = self.align_corners;
        let half_pixel_centers = self.half_pixel_centers;
        let (constants_0, constants_1) = self.constants;

        let ts = quote! {
            let input: microflow::tensor::Tensor4D<_, #(#output_shape),*, 1usize> =
                microflow::ops::resize_bilinear(
                    input,
                    [#(#output_scale),*],
                    [#(#output_zero_point),*],
                    microflow::ops::ResizeBilinearOptions {
                        align_corners: #align_corners,
                        half_pixel_centers: #half_pixel_centers,
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

    fn setup() -> TokenResizeBilinear<i8> {
        TokenResizeBilinear {
            output: TokenTensor4D {
                buffer: TokenBuffer4D::new(),
                shape: vec![1, 4, 6, 2],
                scale: vec![0.1],
                zero_point: vec![2],
            },
            align_corners: false,
            half_pixel_centers: true,
            constants: (3., 4.),
        }
    }

    #[test]
    fn resize_bilinear_preprocess() {
        let layer = setup();
        let input = TokenTensor4D {
            buffer: TokenBuffer4D::new(),
            shape: vec![1, 2, 3, 2],
            scale: vec![0.5],
            zero_point: vec![6],
        };
        let constants = TokenResizeBilinear::preprocess(&input, &layer.output);
        assert_eq!(constants.0, 5.);
        assert_eq!(constants.1, -28.);
    }

    #[test]
    fn resize_bilinear_to_tokens() {
        let layer = setup();
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                let input: microflow::tensor::Tensor4D<_, 1usize, 4usize, 6usize, 2usize, 1usize> =
                    microflow::ops::resize_bilinear(
                        input,
                        [0.1f32],
                        [2i8],
                        microflow::ops::ResizeBilinearOptions {
                            align_corners: false,
                            half_pixel_centers: true,
                        },
                        (3f32, 4f32)
                );
            }
            .to_string()
        );
    }
}
