use crate::activation::TokenFusedActivation;
use crate::buffer::TokenBuffer2D;
use crate::quantize::TokenQuantized;
use crate::tensor::{TokenTensor2D, TokenTensor4D, TokenViewPadding};
use crate::tflite_flatbuffers::tflite::{Buffer, Operator, Tensor, TensorType};
use flatbuffers::{ForwardsUOffset, Vector};
use nalgebra::DMatrix;
use proc_macro2::TokenStream;
use quote::{quote, ToTokens};

pub(crate) struct TokenConv2D<T: TokenQuantized> {
    pub(crate) filters: TokenTensor4D<T>,
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
        TensorType::INT8 => Box::new(TokenConv2D::<i8>::new(operator, tensors, buffers)),
        TensorType::UINT8 => Box::new(TokenConv2D::<u8>::new(operator, tensors, buffers)),
        _ => unimplemented!(),
    }
}

impl<T: TokenQuantized> TokenConv2D<T> {
    pub(crate) fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
        buffers: Vector<ForwardsUOffset<Buffer>>,
    ) -> Self {
        let inputs = operator.inputs().unwrap();
        let input = TokenTensor4D::from_empty_tensor(tensors.get(inputs.get(0) as usize));
        let filters =
            TokenTensor4D::from_buffered_tensor(tensors.get(inputs.get(1) as usize), buffers);
        let biases =
            TokenTensor2D::from_buffered_tensor(tensors.get(inputs.get(2) as usize), buffers);
        let output = TokenTensor4D::from_empty_tensor(
            tensors.get(operator.outputs().unwrap().get(0) as usize),
        );
        let options = operator.builtin_options_as_conv_2_doptions().unwrap();
        let constants = Self::preprocess(&input, &filters, &biases, &output);
        Self {
            filters,
            output,
            fused_activation: options.fused_activation_function().into(),
            view_padding: options.padding().into(),
            strides: (options.stride_h() as usize, options.stride_w() as usize),
            constants,
        }
    }

    fn preprocess(
        input: &TokenTensor4D<T>,
        filters: &TokenTensor4D<T>,
        biases: &TokenTensor2D<i32>,
        output: &TokenTensor4D<T>,
    ) -> (TokenBuffer2D<f32>, TokenBuffer2D<f32>) {
        (
            TokenBuffer2D::from(DMatrix::from_fn(filters.shape[0], 1, |b, _| {
                biases.scale.get(b).copied().unwrap_or(biases.scale[0]) / output.scale[0]
                    * (biases.buffer[b]
                        - biases
                            .zero_point
                            .get(b)
                            .copied()
                            .unwrap_or(biases.zero_point[0])) as f32
            })),
            TokenBuffer2D::from(DMatrix::from_fn(filters.scale.len(), 1, |b, _| {
                input.scale[0] * filters.scale[b] / output.scale[0]
            })),
        )
    }
}

impl<T: TokenQuantized> ToTokens for TokenConv2D<T> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let mut output_shape = self.output.shape.clone();
        output_shape.push(self.output.scale.len());
        let filters = &self.filters;
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        let fused_activation = self.fused_activation;
        let view_padding = self.view_padding;
        let (strides_0, strides_1) = self.strides;
        let (constants_0, constants_1) = &self.constants;

        let output = quote! {
            let output: microflow::tensor::Tensor4D<_, #(#output_shape),*> = microflow::ops::conv_2d(
                output.into(),
                #filters,
                [#(#output_scale),*],
                [#(#output_zero_point),*],
                microflow::ops::Conv2DOptions {
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

    #[test]
    fn conv_2d_preprocess() {
        let input = TokenTensor4D {
            buffer: TokenBuffer4D::new(),
            shape: vec![1, 2, 3, 2],
            scale: vec![0.1],
            zero_point: vec![2],
        };
        let filters = TokenTensor4D {
            buffer: TokenBuffer4D::new(),
            shape: vec![2, 2, 3, 2],
            scale: vec![0.15, 0.16],
            zero_point: vec![17, 18],
        };
        let biases = TokenTensor2D {
            buffer: TokenBuffer2D::from(dmatrix![
                19; 20
            ]),
            shape: vec![2, 1],
            scale: vec![0.21, 0.22],
            zero_point: vec![23, 24],
        };
        let output = TokenTensor4D {
            buffer: TokenBuffer4D::new(),
            shape: vec![1, 2, 3, 2],
            scale: vec![0.25],
            zero_point: vec![26],
        };
        let constants = TokenConv2D::preprocess(&input, &filters, &biases, &output);
        assert_eq!(constants.0 .0, Some(dmatrix![-3.36; -3.52]));
        assert_eq!(constants.1 .0, Some(dmatrix![0.060000002; 0.064]));
    }

    #[test]
    fn conv_2d_to_tokens() {
        let layer = TokenConv2D {
            filters: TokenTensor4D {
                buffer: TokenBuffer4D::from(vec![
                    dmatrix![
                        vec![1i8, 2i8], vec![3i8, 4i8],  vec![5i8,  6i8];
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
            output: TokenTensor4D {
                buffer: TokenBuffer4D::new(),
                shape: vec![1, 2, 3, 2],
                scale: vec![0.33],
                zero_point: vec![34],
            },
            fused_activation: TokenFusedActivation::Relu6,
            view_padding: TokenViewPadding::Same,
            strides: (1, 1),
            constants: (
                TokenBuffer2D::from(dmatrix![35., 36.]),
                TokenBuffer2D::from(dmatrix![37., 38.]),
            ),
        };
        let filters = &layer.filters;
        let fused_activation = layer.fused_activation;
        let view_padding = layer.view_padding;
        let (constants_0, constants_1) = &layer.constants;
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                let output: microflow::tensor::Tensor4D<_, 1usize, 2usize, 3usize, 2usize, 1usize> = microflow::ops::conv_2d(
                    output.into(),
                    #filters,
                    [0.33f32],
                    [34i8],
                    microflow::ops::Conv2DOptions {
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
