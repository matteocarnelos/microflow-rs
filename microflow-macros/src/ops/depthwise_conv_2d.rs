use crate::activation::TokenFusedActivation;
use crate::quantize::TokenQuantized;
use crate::tensor::{TokenTensor2D, TokenTensor4D};
use crate::tflite_flatbuffers::tflite::{Buffer, Operator, Padding, Tensor};
use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro2::TokenStream;
use quote::{quote, ToTokens};
use simba::scalar::SupersetOf;

pub(crate) struct DepthwiseConv2D<T1: TokenQuantized, T2: TokenQuantized> {
    pub(crate) weights: TokenTensor4D<T1>,
    pub(crate) biases: TokenTensor2D<T2>,
    pub(crate) output: TokenTensor4D<T1>,
    pub(crate) fused_activation: TokenFusedActivation,
    pub(crate) padding: Padding,
    pub(crate) strides: (usize, usize),
}

impl<T1: TokenQuantized, T2: TokenQuantized> DepthwiseConv2D<T1, T2>
where
    T2: SupersetOf<T1>,
{
    pub(crate) fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
        buffers: Vector<ForwardsUOffset<Buffer>>,
    ) -> Self {
        let inputs = operator.inputs().unwrap();
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
        Self {
            weights,
            biases,
            output,
            fused_activation: TokenFusedActivation(options.fused_activation_function()),
            padding: options.padding(),
            // TODO: Check if swap is needed
            strides: (options.stride_h() as usize, options.stride_w() as usize),
        }
    }
}

impl<T1: TokenQuantized, T2: TokenQuantized> ToTokens for DepthwiseConv2D<T1, T2> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let mut output_shape = self.output.shape.clone();
        output_shape.push(self.output.scale.len());
        let weights = &self.weights;
        let biases = &self.biases;
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        let fused_activation = &self.fused_activation;
        let padding = match self.padding {
            Padding::SAME => quote!(microflow::ops::DepthwiseConv2DPadding::SAME),
            Padding::VALID => quote!(microflow::ops::DepthwiseConv2DPadding::VALID),
            _ => unreachable!(),
        };
        let (strides_0, strides_1) = &self.strides;

        let output = quote! {
            let output: microflow::tensor::Tensor4D<i8, #(#output_shape),*> = microflow::ops::depthwise_conv_2d(
                output.into(),
                #weights,
                #biases,
                [#(#output_scale),*],
                [#(#output_zero_point),*],
                microflow::ops::DepthwiseConv2DOptions {
                    fused_activation: #fused_activation,
                    padding: #padding,
                    strides: (#strides_0, #strides_1)
                }
            );
        };
        output.to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{TokenBuffer2D, TokenBuffer4D};
    use crate::tflite_flatbuffers::tflite::ActivationFunctionType;
    use nalgebra::dmatrix;

    #[test]
    fn depthwise_conv_2d_to_tokens() {
        let layer = DepthwiseConv2D {
            weights: TokenTensor4D {
                buffer: TokenBuffer4D::from(vec![
                    dmatrix![
                        vec![1i8, 2i8], vec![3i8, 4i8],  vec![5i8, 6i8];
                        vec![7i8, 8i8], vec![9i8, 10i8], vec![11i8, 12i8]
                    ],
                    dmatrix![
                        vec![13i8, 14i8], vec![15i8, 16i8], vec![17i8, 18i8];
                        vec![19i8, 20i8], vec![21i8, 22i8], vec![23i8, 24i8]
                    ],
                ]),
                shape: vec![2, 2, 3, 2],
                scale: vec![0.25, 0.26],
                zero_point: vec![27, 28],
            },
            biases: TokenTensor2D {
                buffer: TokenBuffer2D::from(dmatrix![29i32; 30i32]),
                shape: vec![2, 1],
                scale: 0.31,
                zero_point: 32,
            },
            output: TokenTensor4D {
                buffer: TokenBuffer4D::new(),
                shape: vec![2, 2, 3, 2],
                scale: vec![0.33, 0.34],
                zero_point: vec![35, 36],
            },
            fused_activation: TokenFusedActivation(ActivationFunctionType::RELU6),
            padding: Padding::SAME,
            strides: (1, 1),
        };
        let weights = &layer.weights;
        let biases = &layer.biases;
        let fused_activation = &layer.fused_activation;
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                let output: microflow::tensor::Tensor4D<i8, 2usize, 2usize, 3usize, 2usize, 2usize> = microflow::ops::depthwise_conv_2d(
                    output.into(),
                    #weights,
                    #biases,
                    [0.33f32, 0.34f32],
                    [35i8, 36i8],
                    microflow::ops::DepthwiseConv2DOptions {
                        fused_activation: #fused_activation,
                        padding: microflow::ops::DepthwiseConv2DPadding::SAME,
                        strides: (1usize, 1usize)
                    }
                );
            }.to_string()
        );
    }
}
