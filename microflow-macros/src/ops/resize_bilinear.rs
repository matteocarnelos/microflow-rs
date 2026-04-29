use crate::quantize::TokenQuantized;
use crate::tensor::TokenTensor4D;
use crate::tflite_flatbuffers::tflite::{Buffer, Operator, Tensor, TensorType};
use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro2::TokenStream as TokenStream2;
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
/// * `operator` - The model operator as an [`Operator`].
/// * `tensors` - A `Vector` of all tensors in the model.
/// * `buffers` - A `Vector` of all buffers in the model.
/// * `index` - The index of the current operator.
///
/// # Returns
/// A `Box<dyn ToTokens>` which can be converted to Rust code.
pub(crate) fn parse(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
    buffers: Vector<ForwardsUOffset<Buffer>>,
    index: usize,
) -> Box<dyn ToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => {
            Box::new(TokenResizeBilinear::<i8>::new(operator, tensors, buffers, index))
        }
        TensorType::UINT8 => {
            Box::new(TokenResizeBilinear::<u8>::new(operator, tensors, buffers, index))
        }
        _ => unimplemented!(),
    }
}

impl<T: TokenQuantized> TokenResizeBilinear<T> {
    /// Creates a new [`TokenResizeBilinear`] struct.
    pub(crate) fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
        _buffers: Vector<ForwardsUOffset<Buffer>>,
        _index: usize,
    ) -> Self {
        let inputs = operator.inputs().unwrap();
        let outputs = operator.outputs().unwrap();

        if inputs.len() != 2 {
            panic!("ResizeBilinear operator expects 2 inputs");
        }

        let input_tensor = tensors.get(inputs.get(0) as usize);
        let _size_tensor = tensors.get(inputs.get(1) as usize); // Size tensor (target dimensions)
        let output_tensor = tensors.get(outputs.get(0) as usize);

        let options = operator
            .builtin_options_as_resize_bilinear_options()
            .unwrap();

        let input = TokenTensor4D::<T>::from_empty_tensor(input_tensor);
        let output = TokenTensor4D::<T>::from_empty_tensor(output_tensor);

        let constants = Self::preprocess(&input, &output);

        Self {
            output,
            align_corners: options.align_corners(),
            half_pixel_centers: options.half_pixel_centers(),
            constants,
        }
    }

    /// Pre-processes the operator, returning the tuple of constants for requantization.
    /// # Arguments
    /// * `input` - The input tensor as a [`TokenTensor4D`]
    /// * `output` - The output tensor as a [`TokenTensor4D`]
    ///
    fn preprocess(
        input: &TokenTensor4D<T>, 
        output: &TokenTensor4D<T>
    ) -> (f32, f32) {
        let input_scale = input.scale.get(0).copied().unwrap_or(1.0);
        let output_scale = output.scale.get(0).copied().unwrap_or(1.0);
        let input_zero_point = input.zero_point.get(0).copied().unwrap_or(T::from_superset_unchecked(&0));
        let output_zero_point = output.zero_point.get(0).copied().unwrap_or(T::from_superset_unchecked(&0));

        let scale_ratio = input_scale / output_scale;
        let zero_point_offset = f32::from_subset(&output_zero_point) - scale_ratio * f32::from_subset(&input_zero_point);

        (scale_ratio, zero_point_offset)
    }
}

impl<T: TokenQuantized> ToTokens for TokenResizeBilinear<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let output_dims = &self.output.shape;
        let output_rows = output_dims[1];
        let output_cols = output_dims[2];
        let output_chans = output_dims[3];

        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;

        let align_corners = self.align_corners;
        let half_pixel_centers = self.half_pixel_centers;
        let (constants_0, constants_1) = self.constants;

        let resize_bilinear = quote! {
            let input: microflow::tensor::Tensor4D<_, 1, #output_rows, #output_cols, #output_chans, 1> = microflow::ops::resize_bilinear(
                input,
                [#(#output_scale),*],
                [#(#output_zero_point),*],
                microflow::ops::ResizeBilinearOptions {
                    align_corners: #align_corners,
                    half_pixel_centers: #half_pixel_centers,
                },
                (#constants_0, #constants_1),
            );
        };
        tokens.extend(resize_bilinear);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::ToTokens;

    #[test]
    fn test_to_tokens_i8() {
        let op = TokenResizeBilinear::<i8> {
            output: TokenTensor4D {
                buffer: TokenBuffer4D::new(),
                shape: vec![1, 224, 224, 3],
                scale: vec![0.0039],
                zero_point: vec![-128],
            },
            align_corners: false,
            half_pixel_centers: false,
            constants: (1.0, 0.5),
        };

        let expected = quote! {
            let input: microflow::tensor::Tensor4D<_, 1, 224usize, 224usize, 3usize, 1> = microflow::ops::resize_bilinear(
                input,
                [0.0039f32],
                [-128i8],
                microflow::ops::ResizeBilinearOptions {
                    align_corners: false,
                    half_pixel_centers: false,
                },
                (1f32, 0.5f32),
            );
        };
        assert_eq!(op.to_token_stream().to_string(), expected.to_string());
    }
}