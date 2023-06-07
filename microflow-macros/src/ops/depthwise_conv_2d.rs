use crate::activation::TokenFusedActivation;
use crate::buffer::TokenBuffer2D;
use crate::quantize::TokenQuantized;
use crate::tensor::{TokenTensor2D, TokenTensor4D, TokenViewPadding};
use crate::tflite_flatbuffers::tflite::{Buffer, Operator, Tensor, TensorType};
use flatbuffers::{ForwardsUOffset, Vector};
use nalgebra::DMatrix;
use proc_macro2::TokenStream;
use quote::{quote, ToTokens};

pub(crate) struct TokenDepthwiseConv2D<T: TokenQuantized> {
    pub(crate) input: TokenTensor4D<T>,
    pub(crate) weights: TokenTensor4D<T>,
    pub(crate) output: TokenTensor4D<T>,
    pub(crate) fused_activation: TokenFusedActivation,
    pub(crate) view_padding: TokenViewPadding,
    pub(crate) strides: (usize, usize),
    pub(crate) constants: (TokenBuffer2D<f32>, TokenBuffer2D<f32>),
}

pub(crate) fn parse(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
    buffers: Vector<ForwardsUOffset<Buffer>>,
) -> Box<dyn ToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(TokenDepthwiseConv2D::<i8>::new(operator, tensors, buffers)),
        TensorType::UINT8 => Box::new(TokenDepthwiseConv2D::<u8>::new(operator, tensors, buffers)),
        _ => unimplemented!(),
    }
}

impl<T: TokenQuantized> TokenDepthwiseConv2D<T> {
    pub(crate) fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
        buffers: Vector<ForwardsUOffset<Buffer>>,
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
        Self {
            input,
            weights,
            output,
            fused_activation: options.fused_activation_function().into(),
            view_padding: options.padding().into(),
            strides: (options.stride_h() as usize, options.stride_w() as usize),
            constants,
        }
    }

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
}

impl<T: TokenQuantized> ToTokens for TokenDepthwiseConv2D<T> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let input_shape = &self.input.shape;
        let output_shape = &self.output.shape;
        let weights = &self.weights;
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        let fused_activation = self.fused_activation;
        let view_padding = self.view_padding;
        let (strides_0, strides_1) = self.strides;
        let (constants_0, constants_1) = &self.constants;

        let output = quote! {
            let input: microflow::tensor::Tensor4D<_, #(#input_shape),*, 1usize> = input.into();
            let input: microflow::tensor::Tensor4D<_, #(#output_shape),*, 1usize> = microflow::ops::depthwise_conv_2d(
                input,
                #weights,
                [#(#output_scale),*],
                [#(#output_zero_point),*],
                microflow::ops::DepthwiseConv2DOptions {
                    fused_activation: #fused_activation,
                    view_padding: #view_padding,
                    strides: (#strides_0, #strides_1),
                },
                (#constants_0, #constants_1)
            );
        };
        output.to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{TokenBuffer2D, TokenBuffer4D};
    use nalgebra::dmatrix;

    fn setup() -> TokenDepthwiseConv2D<i8> {
        TokenDepthwiseConv2D {
            input: TokenTensor4D {
                buffer: TokenBuffer4D::new(),
                shape: vec![1, 2, 3, 2],
                scale: vec![0.1],
                zero_point: vec![2],
            },
            weights: TokenTensor4D {
                buffer: TokenBuffer4D::from(vec![dmatrix![
                    vec![3, 4],  vec![5,  6],  vec![7,  8];
                    vec![9, 10], vec![11, 12], vec![13, 14]
                ]]),
                shape: vec![1, 2, 3, 2],
                scale: vec![0.15, 0.16],
                zero_point: vec![17, 18],
            },
            output: TokenTensor4D {
                buffer: TokenBuffer4D::new(),
                shape: vec![1, 2, 3, 2],
                scale: vec![0.19],
                zero_point: vec![20],
            },
            fused_activation: TokenFusedActivation::Relu6,
            view_padding: TokenViewPadding::Same,
            strides: (1, 1),
            constants: (
                TokenBuffer2D::from(dmatrix![21., 22.]),
                TokenBuffer2D::from(dmatrix![23., 24.]),
            ),
        }
    }

    #[test]
    fn depthwise_conv_2d_preprocess() {
        let layer = setup();
        let biases = TokenTensor2D {
            buffer: TokenBuffer2D::from(dmatrix![
                25;
                26
            ]),
            shape: vec![2, 1],
            scale: vec![0.27, 0.28],
            zero_point: vec![29, 30],
        };
        let constants =
            TokenDepthwiseConv2D::preprocess(&layer.input, &layer.weights, &biases, &layer.output);
        assert_eq!(constants.0 .0, Some(dmatrix![-5.684211; -5.894737]));
        assert_eq!(constants.1 .0, Some(dmatrix![0.07894737; 0.08421053]))
    }

    #[test]
    fn depthwise_conv_2d_to_tokens() {
        let layer = setup();
        let weights = &layer.weights;
        let fused_activation = layer.fused_activation;
        let view_padding = layer.view_padding;
        let (constants_0, constants_1) = &layer.constants;
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                let input: microflow::tensor::Tensor4D<_, 1usize, 2usize, 3usize, 2usize, 1usize> = input.into();
                let input: microflow::tensor::Tensor4D<_, 1usize, 2usize, 3usize, 2usize, 1usize> = microflow::ops::depthwise_conv_2d(
                    input,
                    #weights,
                    [0.19f32],
                    [20i8],
                    microflow::ops::DepthwiseConv2DOptions {
                        fused_activation: #fused_activation,
                        view_padding: #view_padding,
                        strides: (1usize, 1usize),
                    },
                    (#constants_0, #constants_1)
                );
            }.to_string()
        );
    }
}
